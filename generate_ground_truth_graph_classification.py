""" explainer_main.py

     Main user interface for the explainer module.
"""
import argparse
import os
from networkx.algorithms.components.connected import connected_components

import sklearn.metrics as metrics

from tensorboardX import SummaryWriter

import sys
import math
import pickle
import shutil
import torch
import numpy as np
import networkx as nx

import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'gnnexp'))

import models
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
from explainer import explain


decimal_round = lambda x: round(x, 5)
color_map = ['gray', 'blue', 'purple', 'red', 'brown', 'green', 'orange', 'olive']

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
    parser.add_argument('--top_k', type=int, default=None, help='Keep k edges of prediction.')
    parser.add_argument('--threshold', type=float, default=None, help='Keep k edges of prediction.')
    parser.add_argument('--output', type=str, default=None, help='output path.')
    parser.add_argument('--disconnected', action="store_true", help='Allow distill disconnected subgraphs.')

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

def main():
    # Load a configuration
    prog_args = arg_parse()
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
    ce = torch.nn.CrossEntropyLoss(reduction='none')
    # load state_dict (obtained by model.state_dict() when saving checkpoint)
    threshold = 0
    if prog_args.output is None:
        prog_args.output = prog_args.dataset
    if prog_args.top_k is not None:
        prog_args.output += '_top%d' % (prog_args.top_k)
    elif prog_args.threshold is not None:
        threshold = prog_args.threshold
        prog_args.output += '_threshold%s' % (prog_args.threshold)
    shutil.rmtree('distillation/%s' % (prog_args.output), ignore_errors=True)
    os.mkdir('distillation/%s' % (prog_args.output))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("Number of graphs:", cg_dict["adj"].shape[0])
    # for graph_idx in range(cg_dict["adj"].shape[0]):
    def run(graph_idx):
        feat = cg_dict["feat"][graph_idx, :].float().unsqueeze(0)
        adj = cg_dict["adj"][graph_idx].float().unsqueeze(0)
        label = cg_dict["label"][graph_idx].long().unsqueeze(0)
        preds, _ = model(feat, adj)
        loss = ce(preds, label)
        G = nx.from_numpy_matrix(adj[0].numpy())
        if prog_args.top_k is not None:
            top_k = prog_args.top_k
        elif prog_args.threshold is not None:
            top_k = math.ceil(G.number_of_edges() * prog_args.threshold)
            top_k = max(3, top_k)
        else:
            top_k = None
        sorted_edges = sorted(G.edges)
        masked_loss = []
        edge_dict = np.zeros(adj.shape[1:], dtype=np.int)
        for edge_idx, (x,y) in enumerate(sorted_edges):
            edge_dict[x,y] = edge_idx
            edge_dict[y,x] = edge_idx
            masked_adj = adj
            masked_adj[0,x,y] = 0
            masked_adj[0,y,x] = 0
            m_preds, _ = model(feat, masked_adj)
            m_loss = ce(m_preds, label)
            masked_loss += [m_loss]
            G[x][y]['weight'] = (m_loss - loss).item()

        masked_loss = torch.stack(masked_loss)
        loss_diff = (masked_loss - loss).squeeze(-1)

        if prog_args.disconnected:
            best_loss = loss.detach()
            masked_loss = []
            weights = loss_diff.detach().numpy()
            largest_cc = max(nx.connected_components(G), key=len)
            sub_G = G.copy()
            sorted_weight_idxs = np.argsort(weights)
            highest_weights = sum(weights)
            extracted_adj = np.zeros(adj.shape[1:])
            for idx, sorted_idx in enumerate(sorted_weight_idxs):
                sub_G.remove_edge(*sorted_edges[sorted_idx])
                masked_adj = torch.tensor(nx.to_numpy_matrix(sub_G, weight=None)).unsqueeze(0).float()
                m_preds, _ = model(feat, masked_adj)
                m_loss = ce(m_preds, label)
                x,y = sorted_edges[sorted_idx]
                masked_loss += [m_loss]
                if m_loss > best_loss:
                    extracted_adj[x,y] = (m_loss - best_loss).item()
                    sub_G.add_edge(*sorted_edges[sorted_idx])
                else:
                    best_loss = m_loss
            masked_loss = torch.stack(masked_loss)
            loss_diff = (masked_loss - best_loss).squeeze(-1)
            
            
            G2 = nx.from_numpy_array(extracted_adj)
            d = nx.get_edge_attributes(G2,'weight')

            if d and top_k is not None:
                edges,weights = zip(*{k: d[k] for k in sorted(d, key=d.get)}.items())
                weights = torch.tensor(weights)
                largest_cc = max(nx.connected_components(G2), key=len)
                sub_G = G2.copy()
                sorted_weight_idxs = np.argsort(weights)
                highest_weights = sum(weights)
                for idx, sorted_idx in enumerate(sorted_weight_idxs):
                    sub_G.remove_edge(*edges[sorted_idx])
                    # largest_cc = max(nx.connected_components(sub_G), key=len)
                    # sub_G2 = sub_G.subgraph(largest_cc)
                    if sub_G.number_of_edges() < top_k:
                        sub_G.add_edge(*edges[sorted_idx])
                G3 = nx.Graph()
                G3.add_nodes_from(list(G2.nodes))
                G3.add_weighted_edges_from([[*e, d[e]] for e in sub_G.edges])
                extracted_adj = nx.to_numpy_matrix(G3)
        else:
            best_loss = loss.detach()
            masked_loss = []
            weights = loss_diff.detach().numpy()
            largest_cc = max(nx.connected_components(G), key=len)
            sub_G = G.copy()
            sorted_weight_idxs = np.argsort(weights)
            highest_weights = sum(weights)
            extracted_adj = np.zeros(adj.shape[1:])
            for idx, sorted_idx in enumerate(sorted_weight_idxs):
                sub_G.remove_edge(*sorted_edges[sorted_idx])
                largest_cc = max(nx.connected_components(sub_G), key=len)
                sub_G2 = sub_G.subgraph(largest_cc)
                sub_G3 = nx.Graph()
                sub_G3.add_nodes_from(list(G.nodes))
                sub_G3.add_edges_from(sub_G2.edges)
                masked_adj = torch.tensor(nx.to_numpy_matrix(sub_G3, weight=None)).unsqueeze(0).float()
                m_preds, _ = model(feat, masked_adj)
                m_loss = ce(m_preds, label)
                x,y = sorted_edges[sorted_idx]
                masked_loss += [m_loss]
                if m_loss > best_loss:
                    extracted_adj[x,y] = (m_loss - best_loss).item()
                    sub_G.add_edge(*sorted_edges[sorted_idx])
                else:
                    best_loss = m_loss
            masked_loss = torch.stack(masked_loss)
            loss_diff = (masked_loss - loss).squeeze(-1)
            
            
            G2 = nx.from_numpy_array(extracted_adj)
            d = nx.get_edge_attributes(G2,'weight')

            if d and top_k is not None:
                edges,weights = zip(*{k: d[k] for k in sorted(d, key=d.get)}.items())
                weights = torch.tensor(weights)
                largest_cc = max(nx.connected_components(G2), key=len)
                sub_G = G2.copy()
                sorted_weight_idxs = np.argsort(weights)
                highest_weights = sum(weights)
                for idx, sorted_idx in enumerate(sorted_weight_idxs):
                    sub_G.remove_edge(*edges[sorted_idx])
                    largest_cc = max(nx.connected_components(sub_G), key=len)
                    sub_G2 = sub_G.subgraph(largest_cc)
                    if sub_G2.number_of_edges() < top_k:
                        sub_G.add_edge(*edges[sorted_idx])
                        break
                G3 = nx.Graph()
                G3.add_nodes_from(list(G2.nodes))
                G3.add_weighted_edges_from([[*e, d[e]] for e in sub_G.edges])
                extracted_adj = nx.to_numpy_matrix(G3)
        G = graph_labeling(G)
        graph_label = np.array([G.nodes[node]['string'] for node in G])
        save_dict = {
            "adj": cg_dict["adj"][graph_idx].float().numpy(),
            "adj_y": extracted_adj,
            "mapping": np.asarray(list(G.nodes)),
            "label": label,
            "features": feat,
            "graph_label": graph_label
        }
        assert highest_weights >= sum(weights)
        assert save_dict['adj'].shape[0] == save_dict['adj_y'].shape[0]
        assert save_dict['adj'].shape[0] == save_dict['mapping'].shape[0]
        torch.save(save_dict, "distillation/%s/graph_idx_%d.ckpt" % (prog_args.output, graph_idx))

    import multiprocessing
    pool = multiprocessing.pool.ThreadPool(processes=2)
    pool.map(run, cg_dict['test_idx']+cg_dict['train_idx']+cg_dict['val_idx'])
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()

