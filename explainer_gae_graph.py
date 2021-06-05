import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import os
import sys
import shutil
import random
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import scipy.sparse as sp

from gae.model import GCNModelVAE, GCNModelVAE3
from gae.optimizer import loss_function as gae_loss
from gnnexp.utils import graph_utils

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--n_hops', type=int, default=3, help='Number of hops.')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Number of samples in a minibatch.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='syn1', help='Type of dataset.')
parser.add_argument('--distillation', type=str, default=None, help='Path of distillation.')
parser.add_argument('--output', type=str, default=None, help='Path of output dir.')
parser.add_argument('--load_checkpoint', default=None, help='Load parameters from checkpoint.')
parser.add_argument('--save_checkpoint', default=None, help='Save parameters to checkpoint.')

parser.add_argument('--exclude_non_label', action='store_true')
parser.add_argument('--early_stop', action='store_true')
parser.add_argument('--train_on_positive_label', action='store_true')
parser.add_argument('--label_feat', action='store_true')
parser.add_argument('--graph_labeling', action='store_true')
parser.add_argument('--degree_feat', action='store_true')
parser.add_argument('--neigh_degree_feat', type=int, default=0, help='Number of neighbors\' degree.')
parser.add_argument('--normalize_feat', action='store_true')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--weighted', action='store_true')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--gae3', action='store_true')
parser.add_argument('--explain_class', type=int, default=None, help='Number of training epochs.')
parser.add_argument('--loss', type=str, default='mse', help='Loss function')

args = parser.parse_args()

algo_conf = {
    "max_grad_norm": 1,
    "num_minibatch": 10
}

if args.gpu and torch.cuda.is_available():
    print("Use cuda")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)
if args.distillation is None:
    args.distillation = args.dataset

decimal_round = lambda x: round(x, 5)
color_map = ['gray', 'blue', 'purple', 'red', 'brown', 'green', 'orange', 'olive']
ckpt = torch.load('ckpt/%s_base_h20_o20.pth.tar'%(args.dataset))
cg_dict = ckpt["cg"]
label_onehot = torch.eye(100, dtype=torch.float)

class GraphSampler(torch.utils.data.Dataset):
    """ Sample graphs and nodes in graph
    """

    def __init__(
        self,
        graph_idxs
    ):
        self.graph_idxs = graph_idxs
        self.graph_data = [load_graph(graph_idx) for graph_idx in graph_idxs]

    def __len__(self):
        return len(self.graph_idxs)

    def __getitem__(self, idx):
        return self.graph_data[idx]


def load_graph(graph_idx):
    """Returns the neighborhood of a given ndoe."""
    data = torch.load("distillation/%s/graph_idx_%d.ckpt" %(args.distillation, graph_idx))
    sub_adj = torch.from_numpy(np.int64(data['adj']>0)).float()
    adj_norm = preprocess_graph(sub_adj.numpy())
    sub_feat = data['features'].squeeze(0)
    if args.normalize_feat:
        sub_feat = F.normalize(sub_feat, p=2, dim=1)
    if args.degree_feat:
        degree_feat = torch.sum(sub_adj, dim=0).unsqueeze(1)
        sub_feat = torch.cat((sub_feat, degree_feat), dim=1)
    if args.neigh_degree_feat > 0:
        degree_feat = torch.sum(sub_adj, dim=0)
        neigh_degree = degree_feat.repeat(100,1) * sub_adj
        v, _ = neigh_degree.sort(axis=1, descending=True)
        sub_feat = torch.cat((sub_feat, v[:,:args.neigh_degree_feat]), dim=1)
    if args.graph_labeling:
        # add graph labelling result
        graph_label = data['graph_label']
        graph_label_onehot = label_onehot[graph_label]
        sub_feat = torch.cat((sub_feat, graph_label_onehot), dim=1)
    sub_label = data['label']
    if args.weighted:
        sub_loss_diff = data['adj_y']
        # sub_loss_diff = data['adj_y']/np.max(data['adj_y'])
        # sub_loss_diff = np.sqrt(sub_loss_diff)
    else:
        sub_loss_diff = np.int64(data['adj_y']>0)
    sub_loss_diff = torch.from_numpy(sub_loss_diff).float()
    adj_label = sub_loss_diff + np.eye(sub_loss_diff.shape[0])
    n_nodes = sub_loss_diff.shape[0]
    pos_weight = float(sub_loss_diff.shape[0] * sub_loss_diff.shape[0] - sub_loss_diff.sum()) / sub_loss_diff.sum()
    pos_weight = torch.from_numpy(np.array(pos_weight))
    norm = torch.tensor(sub_loss_diff.shape[0] * sub_loss_diff.shape[0] / float((sub_loss_diff.shape[0] * sub_loss_diff.shape[0] - sub_loss_diff.sum()) * 2))
    return {
        "graph_idx": graph_idx, 
        "sub_adj": sub_adj.to(device), 
        "adj_norm": adj_norm.to(device).float(), 
        "sub_feat": sub_feat.to(device).float(), 
        "sub_label": sub_label.to(device).float(), 
        "sub_loss_diff": sub_loss_diff.to(device).float(),
        "adj_label": adj_label.to(device).float(),
        "n_nodes": n_nodes,
        "pos_weight": pos_weight.to(device),
        "norm": norm.to(device)
    }

def get_edges(adj_dict, edge_dict, node, hop, edges=set(), visited=set()):
    for neighbor in adj_dict[node]:
        edges.add(edge_dict[node, neighbor])
        visited.add(neighbor)
    if hop <= 1:
        return edges, visited
    for neighbor in adj_dict[node]:
        edges, visited = get_edges(adj_dict, edge_dict, neighbor, hop-1, edges, visited)
    return edges, visited

def preprocess_graph(adj):
    adj_ = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return torch.from_numpy(adj_normalized).float()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def main():
    
    def graph_labeling(G):
        for node in G:
            G.nodes[node]['string'] = 1
        old_strings = tuple([G.nodes[node]['string'] for node in G])
        for iter_num in range(100):
            for node in G:
                string = sorted([G.nodes[neigh]['string'] for neigh in G.neighbors(node)])
                G.nodes[node]['concat_string'] =  tuple([G.nodes[node]['string']] + string)
            d = nx.get_node_attributes(G,'concat_string')
            nodes,strings = zip(*{k: d[k] for k in sorted(d, key=d.get)}.items())
            map_string = dict([[string, i+1] for i, string in enumerate(sorted(set(strings)))])
            for node in nodes:
                G.nodes[node]['string'] = map_string[G.nodes[node]['concat_string']]
            new_strings = tuple([G.nodes[node]['string'] for node in G])
            if old_strings == new_strings:
                break
            else:
                old_strings = new_strings
        return G

    def eval_model(dataset):
        with torch.no_grad():
            losses = []
            for data in dataset:
                recovered, mu, logvar = model(data['sub_feat'], data['adj_norm'])
                loss = criterion(recovered, mu, logvar, data)
                losses += [loss.view(-1)]
        return (torch.cat(losses)).mean().item()

    def plot_graph(graph_idx, sub_adj, sub_loss_diff, recovered):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
        G1 = nx.from_numpy_array(sub_loss_diff.numpy())
        d = nx.get_edge_attributes(G1,'weight')
        largest_cc = max(nx.connected_components(G1), key=len)
        sub_G1 = G1.subgraph(largest_cc)
        pos = nx.spring_layout(sub_G1)
        node_labels = dict([(n,n) for n in sub_G1])
        edge_labels=dict([((u,v,),decimal_round(d['weight']))
            for u,v,d in sub_G1.edges(data=True)])
        nx.draw(sub_G1,pos,node_size=400,ax=ax1)
        nx.draw_networkx_labels(sub_G1,pos,node_labels,font_size=10, font_color='w',ax=ax1)
        nx.draw_networkx_edge_labels(sub_G1,pos,edge_labels=edge_labels, font_size=6,ax=ax1)
        ax1.set_title("Important Edges")
        
        pred_adj = recovered.numpy() * sub_adj.numpy()
        np.fill_diagonal(pred_adj, 0)
        G2 = nx.from_numpy_array(pred_adj)
        d = nx.get_edge_attributes(G2,'weight')
        largest_cc = max(nx.connected_components(G2), key=len)
        sub_G2 = G2.subgraph(largest_cc)
        fixed_nodes = pos.keys()
        fixed_nodes = [n for n in sub_G2 if n in fixed_nodes]
        if fixed_nodes == []:
            pos = nx.spring_layout(sub_G2)
        else:
            pos = nx.spring_layout(sub_G2, pos=pos, fixed=fixed_nodes)
        node_labels = dict([(n,n) for n in sub_G2])
        edge_labels=dict([((u,v,),decimal_round(d['weight']))
            for u,v,d in sub_G2.edges(data=True)])
        nx.draw(sub_G2,pos,node_size=400,ax=ax2)
        nx.draw_networkx_labels(sub_G2,pos,node_labels,font_size=10, font_color='w',ax=ax2)
        nx.draw_networkx_edge_labels(sub_G2,pos,edge_labels=edge_labels, font_size=6,ax=ax2)
        ax2.set_title("Predict Edges")
        fig.suptitle("pred for node %d, loss = %f" %(graph_idx, loss.item()))
        plt.savefig("explanation/%s/graph_idx_%d_pred.pdf" %(args.output, graph_idx))
        plt.clf()
        plt.close(fig)

    def eval_graph(dataset):
        start_time = time.time()
        org_adjs = []
        extracted_adjs = []
        pred_adjs = []
        graph_idxs = []
        for data in dataset:
            sub_adj = data['sub_adj']
            adj_norm = data['adj_norm']
            sub_feat = data['sub_feat']
            sub_loss_diff = data['sub_loss_diff']
            recovered, mu, logvar = model(sub_feat, adj_norm)
            graph_idxs += [data['graph_idx']]
            org_adjs += [sub_adj]
            extracted_adjs += [sub_loss_diff]
            pred_adjs += [recovered]
        print("Inference time:", time.time() - start_time)
        graph_idxs = torch.cat(graph_idxs).cpu().numpy()
        org_adjs = torch.cat(org_adjs).cpu().numpy()
        extracted_adjs = torch.cat(extracted_adjs).cpu().numpy()
        pred_adjs = torch.cat(pred_adjs).cpu().numpy()
        for idx, graph_idx in enumerate(graph_idxs):
            np.savetxt("explanation/%s/graph_idx_%d_label.csv"%(args.output, graph_idx), extracted_adjs[idx], delimiter=",")
            np.savetxt("explanation/%s/graph_idx_%d_pred.csv"%(args.output, graph_idx), pred_adjs[idx], delimiter=",")
            if args.plot:
                plot_graph(graph_idx, org_adjs[idx], extracted_adjs[idx], pred_adjs[idx])
        

    def save_checkpoint(filename):
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
            'epoch': epoch
        }, filename)
        print("Checkpoint saved to %s!" % filename)

    data = torch.load("distillation/%s/graph_idx_0.ckpt" %(args.distillation), map_location=device)
    feat_dim = data['features'].shape[-1]
    # add graph labeling as feature
    if args.graph_labeling:
        feat_dim += 100
    if args.degree_feat:
        feat_dim += 1
    feat_dim += args.neigh_degree_feat
    if args.gae3:
        model = GCNModelVAE3(feat_dim, args.hidden1, args.hidden2, args.dropout).to(device)
    else:
        model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    label = cg_dict['label'].numpy()
    pretrain_gnn_pred = cg_dict['pred']
    num_graphs = label.shape[0]
    graph_idxs = np.array(range(num_graphs))
    train_idxs = np.array(cg_dict['train_idx'])
    val_idxs = np.array(cg_dict['val_idx'])
    test_idxs = np.array(cg_dict['test_idx'])
    print("args.explain_class", args.explain_class)
    if args.explain_class is not None:
        train_idxs = train_idxs[np.where(label[train_idxs] == args.explain_class)[0]]
        val_idxs = val_idxs[np.where(label[val_idxs] == args.explain_class)[0]]
        test_idxs = test_idxs[np.where(label[test_idxs] == args.explain_class)[0]]
    # Only train on samples with correct prediction
    pred_label = np.argmax(pretrain_gnn_pred[0], axis=1)
    train_idxs = train_idxs[np.where(pred_label[train_idxs] == label[train_idxs])[0]]
    print("Num of train:", len(train_idxs))
    print("Num of val:", len(val_idxs))
    print("Num of test:", len(test_idxs))

    # MSE
    def mse(x,mu,logvar,data):
        return F.mse_loss(x.view(x.shape[0], -1), data['adj_label'].view(x.shape[0], -1))
    
    # GVAE
    def gaeloss(x,mu,logvar,data):
        return gae_loss(preds=x, labels=data['adj_label'],
                        mu=mu, logvar=logvar, n_nodes=data['n_nodes'],
                        norm=data['norm'], pos_weight=data['pos_weight'])
    if args.loss == 'mse':
        criterion = mse
    elif args.loss == 'gae':
        criterion = gaeloss
    else:
        raise("Loss function %s is not implemented" % args.loss)


    start_epoch = 1
    best_loss = 10000
    if args.load_checkpoint is not None and os.path.exists(args.load_checkpoint):
        print("Load checkpoint from {}".format(args.load_checkpoint))
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']

    train_graphs = GraphSampler(train_idxs)
    train_dataset = torch.utils.data.DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_graphs = GraphSampler(val_idxs)
    val_dataset = torch.utils.data.DataLoader(
        val_graphs,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    test_graphs = GraphSampler(test_idxs)
    test_dataset = torch.utils.data.DataLoader(
        test_graphs,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    print("Initial train loss:", eval_model(train_dataset))
    print("Initial val loss:", eval_model(val_dataset))
    print("Initial test loss:", eval_model(test_dataset))
    shutil.rmtree('explanation/%s' % args.output, ignore_errors=True)
    os.makedirs('explanation/%s' % args.output, exist_ok=True)
    import time
    model.train()
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs+1):
        print("------- Epoch %2d ------" % epoch)
        train_losses = []
        for batch_idx, data in enumerate(train_dataset):
            optimizer.zero_grad()
            recovered, mu, logvar = model(data['sub_feat'], data['adj_norm'])
            loss = criterion(recovered, mu, logvar, data)
            loss.mean().backward()
            nn.utils.clip_grad_norm_(model.parameters(), algo_conf['max_grad_norm'])
            optimizer.step()
            train_losses += [loss.view(-1)]
            sys.stdout.flush()
        
        train_loss = (torch.cat(train_losses)).mean().item()
        val_loss = eval_model(val_dataset)
        test_loss = eval_model(test_dataset)
        if args.early_stop and val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint('explanation/%s/model.ckpt' % args.output)
        print("Train loss:", train_loss)
        print("Val loss:", val_loss)
        print("test loss:", test_loss)
    train_time = time.time() - start_time
    print("Train time:", train_time)
    if epoch % 100 == 0:
        filename = 'explanation/%s/model.ckpt' % args.output
        shutil.copy(filename, filename[:-5]+'-%depoch-best.ckpt' % epoch)
    checkpoint = torch.load('explanation/%s/model.ckpt' % args.output)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    model.eval()
    with torch.no_grad():
        eval_graph(test_dataset)

if __name__ == "__main__":
    main()
