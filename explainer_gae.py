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
parser.add_argument('--lr', type=float, default=0.003, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='syn1', help='Type of dataset.')
parser.add_argument('--distillation', type=str, default=None, help='Path of distillation.')
parser.add_argument('--output', type=str, default=None, help='Path of output dir.')
parser.add_argument('--load_checkpoint', default=None, help='Load parameters from checkpoint.')
parser.add_argument('--save_checkpoint', default=None, help='Save parameters to checkpoint.')
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()

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

decimal_round = lambda x: round(x, 5)
color_map = ['gray', 'blue', 'purple', 'red', 'brown', 'green', 'orange', 'olive']

def get_edges(adj_dict, edge_dict, node, hop, edges=set(), visited=set()):
    # print("get_edges(%s)"%node)
    for neighbor in adj_dict[node]:
        edges.add(edge_dict[node, neighbor])
        visited.add(neighbor)
    if hop <= 1:
        # print("exit, hop", hop)
        return edges, visited
    # print("Go neighbors:", adj_dict[node])
    for neighbor in adj_dict[node]:
        edges, visited = get_edges(adj_dict, edge_dict, neighbor, hop-1, edges, visited)
    return edges, visited

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to_dense()

def main():
    with torch.no_grad():
        ckpt = torch.load('ckpt/%s_base_h20_o20.pth.tar'%(args.dataset))
        cg_dict = ckpt["cg"] # get computation graph
        input_dim = cg_dict["feat"].shape[2]
        adj = cg_dict["adj"][0]
        label = cg_dict["label"][0]
        features = torch.tensor(cg_dict["feat"][0], dtype=torch.float)
        num_class = max(label)+1

    def extract_neighborhood(node_idx):
        """Returns the neighborhood of a given ndoe."""
        data = torch.load("distillation/%s/node_%d.ckpt" %(args.distillation, node_idx))
        mapping = data['mapping']
        node_idx_new = np.where(mapping == node_idx)[0][0]
        sub_adj = torch.tensor(data['adj'], dtype=torch.float)
        # Calculate hop_feat:
        pow_adj = ((sub_adj @ sub_adj >=1).float() - np.eye(sub_adj.shape[0]) - sub_adj >=1).float()
        sub_feat = features[mapping]
        one_hot = torch.zeros((sub_adj.shape[0], ), dtype=torch.float)
        one_hot[node_idx_new] = 1
        hop_feat = [one_hot, sub_adj[node_idx_new], pow_adj[node_idx_new]]
        if args.n_hops == 3:
            pow3_adj = ((pow_adj @ pow_adj >=1).float() - np.eye(pow_adj.shape[0]) - pow_adj >=1).float()
            hop_feat += [pow3_adj[node_idx_new]]
            hop_feat = torch.stack(hop_feat).t()
            sub_feat = torch.cat((sub_feat, hop_feat), dim=1)
        sub_label = torch.from_numpy(data['label'])
        sub_loss_diff = torch.from_numpy(data['adj_y'])
        sub_loss_diff += torch.eye(sub_adj.shape[-1])
        adj_norm = preprocess_graph(sub_adj)
        return {
            "node_idx_new": node_idx_new,
            "sub_adj": sub_adj.to(device),
            "adj_norm": adj_norm.unsqueeze(0).to(device),
            "sub_feat": sub_feat.unsqueeze(0).to(device),
            "sub_label": sub_label.to(device),
            "sub_loss_diff": sub_loss_diff.float().to(device),
            "mapping": mapping
        }


    def eval_model():
        with torch.no_grad():
            val_loss = []
            for node_idx in val_idxs:
                data = dataset[node_idx]
                recovered, mu, logvar = model(data['sub_feat'], data['adj_norm'])
                loss = criterion(recovered, data['sub_loss_diff'])
                val_loss += [loss.item()]
        return np.mean(val_loss)

    def plot_node(node_idx):
        with torch.no_grad():
            data = dataset[node_idx]
            mapping = data['mapping']
            node_idx_new = data['node_idx_new']
            sub_loss_diff = data['sub_loss_diff']
            recovered, mu, logvar = model(data['sub_feat'], data['adj_norm'])
            loss = criterion(recovered, sub_loss_diff)
            np.savetxt("explanation/%s/node%d_label.csv"%(args.output, node_idx), sub_loss_diff.cpu().squeeze(0).numpy(), delimiter=",")
            np.savetxt("explanation/%s/node%d_pred.csv"%(args.output, node_idx), recovered.cpu().squeeze(0).numpy(), delimiter=",")

    def save_checkpoint(filename):
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, filename)
        print("Checkpoint saved to %s!" % filename)

    feat_dim = features.shape[-1]
    # hop feature
    feat_dim += args.n_hops + 1
    model = GCNModelVAE3(feat_dim, args.hidden1, args.hidden2, args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_nodes = label.shape[0]
    train_idxs = np.array(cg_dict['train_idx'])
    test_idxs = np.array([i for i in range(num_nodes) if i not in train_idxs])

    # exclude nodes labeled as class 0 and 4
    train_label = label[train_idxs]
    test_label = label[test_idxs]
    if args.dataset == 'syn2':
        train_idxs = train_idxs[np.where(np.logical_and(train_label != 0, train_label != 4))[0]]
        test_idxs = test_idxs[np.where(np.logical_and(test_label != 0, test_label != 4))[0]]
    else:
        train_idxs = train_idxs[np.where(train_label != 0)[0]]
        test_idxs = test_idxs[np.where(test_label != 0)[0]]

    num_train = len(train_idxs)
    num_test = len(test_idxs)
    dataset = dict([[node_idx,extract_neighborhood(node_idx)] for node_idx in train_idxs])
    dataset.update(dict([[node_idx,extract_neighborhood(node_idx)] for node_idx in test_idxs]))
    val_idxs = test_idxs[:num_test//2]
    test_idxs = test_idxs[num_test//2:]

    # MSE:
    criterion = lambda x,y : F.mse_loss(x.flatten(), y.flatten())

    start_epoch = 1
    if args.load_checkpoint is not None and os.path.exists(args.load_checkpoint):
        print("Load checkpoint from {}".format(args.load_checkpoint))
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

    print("Initial test loss:", eval_model())
    shutil.rmtree('explanation/%s' % args.output, ignore_errors=True)
    os.makedirs('explanation/%s' % args.output)

    model.train()
    batch_size = args.batch_size
    best_loss = 100
    import time
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs+1):
        print("------- Epoch %2d ------" % epoch)
        batch = 0
        perm = np.random.permutation(num_train)
        for beg_ind in range(0, num_train, batch_size):
            batch += 1
            # print("------- Batch %2d ------" % batch)
            end_ind = min(beg_ind+batch_size, num_train)
            perm_train_idxs = train_idxs[perm[beg_ind: end_ind]]
            losses = []
            optimizer.zero_grad()
            for node_idx in perm_train_idxs:
                data = dataset[node_idx]
                recovered, mu, logvar = model(data['sub_feat'], data['adj_norm'])
                loss = criterion(recovered, data['sub_loss_diff'])
                losses += [loss]
            loss = torch.mean(torch.stack(losses))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            train_loss = loss.item()
            sys.stdout.flush()
        
        val_loss = eval_model()
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint('explanation/%s/model.ckpt' % args.output)
        print("Train loss:", train_loss)
        print("Val loss:", val_loss)
    print("Train time:", time.time() - start_time)
    # Load checkpoint with lowest val loss
    checkpoint = torch.load('explanation/%s/model.ckpt' % args.output)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()
    start_time = time.time()
    for i in test_idxs:
        plot_node(i)
    print("Inference time:", time.time() - start_time)

if __name__ == "__main__":
    main()
