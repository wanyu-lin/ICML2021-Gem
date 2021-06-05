""" explainer_main.py

     Main user interface for the explainer module.
"""
import argparse
import os

import sklearn.metrics as metrics

from tensorboardX import SummaryWriter

import pickle
import shutil
import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F

import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'gnnexp'))

import models
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
from explainer import explain



def arg_parse():
    parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument(
        "--bmname", dest="bmname", help="Name of the benchmark dataset"
    )
    io_parser.add_argument("--pkl", dest="pkl_fname", help="Name of the pkl data file")

    parser_utils.parse_optimizer(parser)

    parser.add_argument("--clean-log", action="store_true", help="If true, cleans the specified log directory before running.")
    parser.add_argument("--logdir", dest="logdir", help="Tensorboard log directory")
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_const",
        const=True,
        default=False,
        help="whether to use GPU.",
    )
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument("--dropout", dest="dropout", type=float, help="Dropout rate.")
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--no-writer",
        dest="writer",
        action="store_const",
        const=False,
        default=True,
        help="Whether to do SummaryWriter. Default to True.",
    )
    # Explainer
    parser.add_argument("--mask-act", dest="mask_act", type=str, help="sigmoid, ReLU.")
    parser.add_argument(
        "--mask-bias",
        dest="mask_bias",
        action="store_const",
        const=True,
        default=False,
        help="Whether to add mask bias. Default to True.",
    )
    parser.add_argument(
        "--explain-node", dest="explain_node", type=int, help="Node to explain."
    )
    parser.add_argument(
        "--graph-idx", dest="graph_idx", type=int, help="Graph to explain."
    )
    parser.add_argument(
        "--graph-mode",
        dest="graph_mode",
        action="store_const",
        const=True,
        default=False,
        help="whether to run Explainer on Graph Classification task.",
    )
    parser.add_argument(
        "--multigraph-class",
        dest="multigraph_class",
        type=int,
        help="whether to run Explainer on multiple Graphs from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--multinode-class",
        dest="multinode_class",
        type=int,
        help="whether to run Explainer on multiple nodes from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--align-steps",
        dest="align_steps",
        type=int,
        help="Number of iterations to find P, the alignment matrix.",
    )

    parser.add_argument(
        "--method", dest="method", type=str, help="Method. Possible values: base, att."
    )
    parser.add_argument(
        "--name-suffix", dest="name_suffix", help="suffix added to the output filename"
    )
    parser.add_argument(
        "--explainer-suffix",
        dest="explainer_suffix",
        help="suffix added to the explainer log",
    )
    parser.add_argument('--n_hops', type=int, default=3, help='Number of hops.')
    parser.add_argument('--top_k', type=int, default=None, help='Keep k edges of prediction.')
    parser.add_argument('--threshold', type=float, default=None, help='Keep k edges of prediction.')
    parser.add_argument('--output', type=str, default=None, help='output path.')

    # TODO: Check argument usage
    parser.set_defaults(
        logdir="log",
        ckptdir="ckpt",
        dataset="Mutagenicity",
        opt="adam",  
        opt_scheduler="none",
        cuda="cuda:0",
        lr=0.1,
        clip=2.0,
        batch_size=20,
        num_epochs=100,
        hidden_dim=20,
        output_dim=20,
        num_gc_layers=3,
        dropout=0.0,
        method="base",
        name_suffix="",
        explainer_suffix="",
        align_steps=1000,
        explain_node=None,
        graph_idx=-1,
        graph_mode=False,
        mask_act="sigmoid",
        multigraph_class=-1,
        multinode_class=-1,
    )
    return parser.parse_args()

def get_edges(adj_dict, edge_dict, node, hop, edges=set(), visited=set()):
    for neighbor in adj_dict[node]:
        edges.add(edge_dict[node, neighbor])
        visited.add(neighbor)
    if hop <= 1:
        return edges, visited
    for neighbor in adj_dict[node]:
        edges, visited = get_edges(adj_dict, edge_dict, neighbor, hop-1, edges, visited)
    return edges, visited

def main():
    # Load a configuration
    prog_args = arg_parse()
    
    if prog_args.output is None:
        prog_args.output = prog_args.dataset
    if prog_args.top_k is not None:
        prog_args.output += '_top%d' % (prog_args.top_k)
    elif prog_args.threshold is not None:
        prog_args.output += '_threshold%s' % (prog_args.threshold)
    os.makedirs("distillation/%s" %prog_args.output, exist_ok=True)

    device = torch.device(prog_args.cuda if prog_args.gpu and torch.cuda.is_available() else "cpu")
    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        print("CUDA", prog_args.cuda)
    else:
        print("Using CPU")

    # Load a model checkpoint
    ckpt = io_utils.load_ckpt(prog_args)
    cg_dict = ckpt["cg"] # get computation graph
    input_dim = cg_dict["feat"].shape[2] 
    num_classes = cg_dict["pred"].shape[2]
    print("Loaded model from {}".format(prog_args.ckptdir))
    print("input dim: ", input_dim, "; num classes: ", num_classes)

    # Determine explainer mode
    graph_mode = (
        prog_args.graph_mode
        or prog_args.multigraph_class >= 0
        or prog_args.graph_idx >= 0
    )

    # build model
    print("Method: ", prog_args.method)
    if graph_mode: 
        # Explain Graph prediction
        model = models.GcnEncoderGraph(
            input_dim=input_dim,
            hidden_dim=prog_args.hidden_dim,
            embedding_dim=prog_args.output_dim,
            label_dim=num_classes,
            num_layers=prog_args.num_gc_layers,
            bn=prog_args.bn,
            args=prog_args,
        )
    else:
        if prog_args.dataset == "ppi_essential":
            # class weight in CE loss for handling imbalanced label classes
            prog_args.loss_weight = torch.tensor([1.0, 5.0], dtype=torch.float).to(device) 
        # Explain Node prediction
        model = models.GcnEncoderNode(
            input_dim=input_dim,
            hidden_dim=prog_args.hidden_dim,
            embedding_dim=prog_args.output_dim,
            label_dim=num_classes,
            num_layers=prog_args.num_gc_layers,
            bn=prog_args.bn,
            args=prog_args,
        )
    if prog_args.gpu:
        model = model.to(device) 
    # load state_dict (obtained by model.state_dict() when saving checkpoint)
    model.load_state_dict(ckpt["model_state"])

    feat = torch.from_numpy(cg_dict["feat"]).float()
    adj = torch.from_numpy(cg_dict["adj"]).float()
    label = torch.from_numpy(cg_dict["label"]).long()
    model.eval()
    preds, _ = model(feat, adj)
    # ce = torch.nn.CrossEntropyLoss(reduction='none')
    ce = lambda x,y: F.cross_entropy(x, y, reduction='none')
    loss = ce(preds[0], label[0])
    G = nx.from_numpy_matrix(cg_dict["adj"][0])
    masked_loss = []
    sorted_edges = sorted(G.edges)
    edge_dict = np.zeros(adj.shape[1:], dtype=np.int)
    adj_dict = {}
    for node in G:
        adj_dict[node] = list(G.neighbors(node))
    for edge_idx, (x,y) in enumerate(sorted_edges):
        edge_dict[x,y] = edge_idx
        edge_dict[y,x] = edge_idx
        masked_adj = torch.from_numpy(cg_dict["adj"]).float()
        masked_adj[0,x,y] = 0
        masked_adj[0,y,x] = 0
        m_preds, _ = model(feat, masked_adj)
        m_loss = ce(m_preds[0], label[0])
        masked_loss += [m_loss]

    masked_loss = torch.stack(masked_loss)
    loss_diff = masked_loss - loss
    loss_diff_t = loss_diff.t()

    graphs = []
    def extract(node):
        if cg_dict["label"][0][node] in [0, 4]:
            return
        weights = loss_diff_t[node].detach().numpy()
        sub_edge_idxs, visited = get_edges(adj_dict, edge_dict, node, prog_args.n_hops, edges=set(), visited=set({node}))
        sub_edge_idxs = np.array(list(sub_edge_idxs))
        sub_weights = weights[sub_edge_idxs]
        edges =[]
        for e in sub_edge_idxs:
            x, y = sorted_edges[e]
            edges.append((x,y))
            G[x][y]['weight'] = weights[e]
        sub_G = G.edge_subgraph(edges)
        sorted_idxs = np.argsort(sub_weights)
        edges = [edges[sorted_idx] for sorted_idx in sorted_idxs]
        sub_edge_idxs = sub_edge_idxs[sorted_idxs]

        if prog_args.top_k is not None:
            top_k = prog_args.top_k
        elif prog_args.threshold is not None:
            top_k = math.ceil(sub_G.number_of_edges() * prog_args.threshold)
            top_k = max(3, top_k)
        else:
            top_k = None

        node_loss = loss[node].item()
        best_loss = node_loss
        sub_G_y = sub_G.copy()
        for idx, e in enumerate(sub_edge_idxs):
            sub_G_y.remove_edge(*sorted_edges[e])
            largest_cc = max(nx.connected_components(sub_G_y), key=len)
            sub_G2 = sub_G_y.subgraph(largest_cc)
            sub_G3 = nx.Graph()
            sub_G3.add_nodes_from(list(G.nodes))
            sub_G3.add_edges_from(sub_G2.edges)
            masked_adj = torch.from_numpy(nx.to_numpy_matrix(sub_G3, weight=None)).unsqueeze(0).float()
            m_preds, _ = model(feat, masked_adj)
            m_loss = ce(m_preds[0], label[0])
            x,y = sorted_edges[e]
            if m_loss[node] > best_loss:
                sub_G_y.add_edge(*sorted_edges[e])
                sub_G_y[x][y]['weight'] = (m_loss[node] - best_loss).item()
            else:
                best_loss = m_loss[node]

        d = nx.get_edge_attributes(sub_G_y, 'weight')

        if d and top_k is not None:
            edges,weights = zip(*{k: d[k] for k in sorted(d, key=d.get)}.items())
            sorted_weight_idxs = np.argsort(weights)
            for idx, sorted_idx in enumerate(sorted_weight_idxs):
                sub_G_y.remove_edge(*edges[sorted_idx])
                largest_cc = max(nx.connected_components(sub_G_y), key=len)
                sub_G2 = sub_G_y.subgraph(largest_cc)
                if sub_G2.number_of_edges() < top_k:
                    sub_G_y.add_edge(*edges[sorted_idx])
            

        save_dict = {
            "adj": np.asarray(nx.to_numpy_matrix(sub_G, weight=None)),
            "adj_y": nx.to_numpy_matrix(sub_G_y),
            "mapping": np.asarray(list(sub_G.nodes)),
            "label": np.asarray([cg_dict["label"][0][n] for n in sub_G]),
            "features": feat[0][list(sub_G.nodes)]
        }
        assert save_dict['adj'].shape[0] == save_dict['adj_y'].shape[0], "{}, {}".format(save_dict['adj'].shape[0], save_dict['adj_y'].shape[0])
        assert save_dict['adj'].shape[0] == save_dict['mapping'].shape[0]
        torch.save(save_dict, "distillation/%s/node_%d.ckpt" % (prog_args.output, node))

    with torch.no_grad():
        for node in G:
            extract(node)

if __name__ == "__main__":
    main()

