""" explainer_main.py

     Main user interface for the explainer module.
"""
import argparse
import os

import sklearn.metrics as metrics

from tensorboardX import SummaryWriter

import math
import pickle
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'gnnexp'))

import models
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
from explainer import explain


color_label = ['C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca']
cNorm  = matplotlib.colors.Normalize(vmin=0, vmax=14)
scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=plt.cm.tab20)
color_map = [scalarMap.to_rgba(i) for i in range(14)]
# color_map = ['gray', 'blue', 'purple', 'red', 'slategray', 'green', 'orange', 'olive']

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
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--exclude_non_label', action='store_true')
    parser.add_argument('--top_k', type=int, default=None, help='Keep k edges of prediction.')
    parser.add_argument('--threshold', type=float, default=None, help='Keep k edges of prediction.')
    parser.add_argument('--distillation', type=str, default=None, help='Path of distillation.')
    parser.add_argument('--exp_out', type=str, default=None, help='Path of explainer output.')
    parser.add_argument('--test_out', type=str, default=None, help='Path of test output.')
    parser.add_argument('--explain_class', type=int, default=None, help='Only explain specific class.')

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

def preprocess_graph(adj):
    adj_ = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return torch.from_numpy(adj_normalized).float()

def main():
    # Load a configuration
    prog_args = arg_parse()
    device = torch.device(prog_args.cuda if prog_args.gpu and torch.cuda.is_available() else "cpu")
    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        print("CUDA", prog_args.cuda)
    else:
        print("Using CPU")

    # Configure the logging directory 
    if prog_args.writer:
        path = os.path.join(prog_args.logdir, io_utils.gen_explainer_prefix(prog_args))
        if os.path.isdir(path) and prog_args.clean_log:
           print('Removing existing log dir: ', path)
           if not input("Are you sure you want to remove this directory? (y/n): ").lower().strip()[:1] == "y": sys.exit(1)
           shutil.rmtree(path)
        writer = SummaryWriter(path)
    else:
        writer = None

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
    ce = torch.nn.CrossEntropyLoss(reduction='none')
    softmax = torch.nn.Softmax(dim=0)
    def log_odd(p):
        return (torch.log(p) - torch.log(1-p)).item()

    def evaluate_adj(feat, adj, label, losses, corrects):
        # if prog_args.normalize:
        #     adj = preprocess_graph(adj.squeeze(0).numpy()).unsqueeze(0)
        with torch.no_grad():
            pred, _ = model(feat, adj)
            loss = ce(pred, label)
            _, pred_label = torch.max(pred, 1)
            correct = (pred_label== label).float().sum()
            losses.append(loss.item())
            corrects.append(correct.item())
        return pred_label, softmax(pred[0])

    def plot(adj, feat, pred_label, pos, ax):
        global colors
        sub_G = nx.from_numpy_matrix(adj[0].numpy())
        largest_cc = max(nx.connected_components(sub_G), key=len)
        sub_G = sub_G.subgraph(largest_cc).copy()
        node_labels = dict([(n, n) for n in sub_G])
        fixed_nodes = pos.keys()
        fixed_nodes = [n for n in sub_G if n in fixed_nodes]
        if fixed_nodes:
            _pos = nx.spring_layout(sub_G, pos=pos, fixed=fixed_nodes)
        else:
            _pos = nx.spring_layout(sub_G)
        pos = _pos
        if "PROTEINS_full" in prog_args.dataset:
            node_color = [color_map[np.where(feat[0][n][-3:]==1)[0][0]] for n in sub_G]
            colors += [np.where(feat[0][n][-3:]==1)[0][0] for n in sub_G]
        else:
            node_color = [color_map[np.where(feat[0][n]==1)[0][0]] if len(np.where(feat[0][n]==1)[0]) > 0 else color_map[0] for n in sub_G]
            colors += [np.where(feat[0][n]==1)[0][0] if len(np.where(feat[0][n]==1)[0]) > 0 else 0 for n in sub_G]
            
        nc = nx.draw_networkx_nodes(sub_G, pos, alpha=1, node_color=node_color, node_size=100, ax=ax)
        # nx.draw_networkx_labels(sub_G,pos,node_labels,font_size=8, font_color='w', ax=ax)
        nx.draw_networkx_edges(sub_G, pos, alpha=0.4, ax=ax)
        return pos


    org_losses = []
    extracted_losses = []
    ours_losses = []
    gnnexp_losses = []
    org_corrects = []
    extracted_corrects = []
    ours_corrects = []
    gnnexp_corrects = []
    our_ground_truth_odd = []
    our_pred_label_odd = []
    gnnexp_ground_truth_odd = []
    gnnexp_pred_label_odd = []
    pred_prob = []
    label = cg_dict['label'].numpy()
    graph_idxs = np.array(cg_dict['test_idx'])
    if prog_args.explain_class is not None:
        graph_idxs = graph_idxs[np.where(label[graph_idxs] == prog_args.explain_class)[0]]
    print("Num of graphs:", len(graph_idxs))
    valid_graph_idxs =[]
    suffix = ''
    if prog_args.top_k is not None:
        suffix = "(top_k=%s)" % prog_args.top_k
    elif prog_args.threshold is not None:
        suffix = "(threshold=%s)" % prog_args.threshold

    dir_name = "explanation/%s"%(prog_args.exp_out)
    test = os.listdir(dir_name)
    for item in test:
        if item.endswith(".pdf"):
            os.remove(os.path.join(dir_name, item))
    if prog_args.test_out is None:
        prog_args.test_out = prog_args.exp_out
    shutil.rmtree('log/%s' % prog_args.test_out, ignore_errors=True)
    os.makedirs('log/%s' % prog_args.test_out, exist_ok=True)
    gnnexp_result_path = os.path.join('explanation', 'gnnexp', io_utils.gen_explainer_prefix(prog_args))
    for graph_idx in graph_idxs:
        if not os.path.exists("explanation/%s/graph_idx_%d_pred.csv"%(prog_args.exp_out, graph_idx)):
            continue
        data = torch.load("distillation/%s/graph_idx_%d.ckpt" %(prog_args.distillation, graph_idx), map_location=device)
        sub_label = data['label'].to(device)
        org_adj = torch.from_numpy(np.int64(data['adj']>0)).unsqueeze(0).to(device).float()
        G = nx.from_numpy_matrix(data['adj'])
        if prog_args.top_k is not None:
            top_k = prog_args.top_k
        elif prog_args.threshold is not None:
            top_k = math.ceil(G.number_of_edges() * prog_args.threshold)
            top_k = max(3, top_k)
        else:
            top_k = None

        extracted_adj = torch.from_numpy(np.int64(data['adj_y']>0)).unsqueeze(0).to(device).float()
        sub_feat = data['features']
        norm_sub_feat = F.normalize(data['features'], p=2, dim=2)
        
        recovered = np.loadtxt("explanation/%s/graph_idx_%d_pred.csv"%(prog_args.exp_out, graph_idx), delimiter=',')
        ours_adj = recovered * data['adj']

        ### Top K or threshold ###
        num_nodes = ours_adj.shape[-1]
        if top_k is not None:
            adj_threshold_num = top_k * 2
            neigh_size = len(ours_adj[ours_adj > 0])
            if neigh_size == 0:
                continue
            threshold_num = min(neigh_size, adj_threshold_num)
            threshold = np.sort(ours_adj[ours_adj > 0])[-threshold_num]
        else:
            threshold = 1e-6
            
        weighted_edge_list = [
            (i, j, ours_adj[i, j])
            for i in range(num_nodes)
            for j in range(num_nodes)
            if ours_adj[i, j] >= threshold
        ]
        G2 = nx.Graph()
        G2.add_nodes_from(range(num_nodes))
        G2.add_weighted_edges_from(weighted_edge_list)
        ours_adj = nx.to_numpy_matrix(G2, weight=None)

        ours_adj = torch.from_numpy(ours_adj).unsqueeze(0).float()

        org_pred, org_p = evaluate_adj(sub_feat, org_adj, sub_label, org_losses, org_corrects)
        extracted_pred, extracted_p = evaluate_adj(sub_feat, extracted_adj, sub_label, extracted_losses, extracted_corrects)
        ours_pred, ours_p = evaluate_adj(sub_feat, ours_adj, sub_label, ours_losses, ours_corrects)
        gnnexp_data = torch.load(os.path.join(gnnexp_result_path, 'masked_adj_node_idx_0graph_idx_%s.ckpt' % graph_idx), map_location=device)
        # gnnexp_data = torch.load("log/%s_base_h20_o20_explain/masked_adj_node_idx_0graph_idx_%d.ckpt" %(prog_args.dataset, graph_idx), map_location=device)
        gnnexp_adj = torch.from_numpy(gnnexp_data['adj']).float().unsqueeze(0)
        gnnexp_feat = sub_feat
        gnnexp_label = sub_label
        gnnexp_adj = gnnexp_data['adj']

        ### Top K or threshold ###
        num_nodes = gnnexp_adj.shape[-1]
        if top_k is not None:
            adj_threshold_num = top_k * 2
            neigh_size = len(gnnexp_adj[gnnexp_adj > 0])
            threshold_num = min(neigh_size, adj_threshold_num)
            threshold = np.sort(gnnexp_adj[gnnexp_adj > 0])[-threshold_num]
        else:
            threshold = 1e-6
            
        weighted_edge_list = [
            (i, j, gnnexp_adj[i, j])
            for i in range(num_nodes)
            for j in range(num_nodes)
            if gnnexp_adj[i, j] >= threshold
        ]
        G2 = nx.Graph()
        G2.add_nodes_from(range(num_nodes))
        G2.add_weighted_edges_from(weighted_edge_list)
        gnnexp_adj = nx.to_numpy_matrix(G2, weight=None)

        gnnexp_adj = torch.from_numpy(gnnexp_adj).unsqueeze(0).float()
        gnnexp_pred, gnnexp_p = evaluate_adj(gnnexp_feat, gnnexp_adj, gnnexp_label, gnnexp_losses, gnnexp_corrects)
        graph_label = data["label"]
        if prog_args.plot:
            if org_pred == extracted_pred == ours_pred == gnnexp_pred == graph_label:
                prefix = 'correct_'
            else:
                prefix = 'wrong_'
            fig, axes = plt.subplots(1, 4, figsize=(20,5))
            fig.suptitle("Graph %d, label=%d" % (graph_idx, graph_label))
            axes[0].set_title("Original adj", color="Black" if (org_pred==graph_label).item() else "Red")
            axes[1].set_title("Extracted adj", color="Black" if (extracted_pred==graph_label).item() else "Red")
            axes[2].set_title("Our Explainer adj", color="Black" if (ours_pred==graph_label).item() else "Red")
            axes[3].set_title("GNNExplainer adj", color="Black" if (gnnexp_pred==graph_label).item() else "Red")
            global colors
            colors = []
            pos = dict({})
            pos = plot(org_adj, sub_feat, org_pred, pos, axes[0])
            _pos = plot(extracted_adj, sub_feat, extracted_pred, pos, axes[1])
            _pos = plot(ours_adj, sub_feat, ours_pred, pos, axes[2])
            _pos = plot(gnnexp_adj, sub_feat, gnnexp_pred, pos, axes[3])
            patches = []
            for i in set(colors):
                patches += [mpatches.Patch(color=color_map[i], label=color_label[i])]
            plt.axis('off')
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 0.0), loc="lower right", fontsize=10)
            plt.savefig("log/%s/%sgraph_idx_%d.pdf" %(prog_args.test_out, prefix, graph_idx))
            plt.clf()
            plt.close(fig)

        pred_prob += [[graph_idx] + list(org_p) + list(extracted_p) + list(ours_p) + list(gnnexp_p)]
        if np.inf in [log_odd(org_p[graph_label]), log_odd(ours_p[graph_label]), log_odd(gnnexp_p[graph_label]),
                        log_odd(org_p[org_pred]), log_odd(ours_p[org_pred]), log_odd(gnnexp_p[org_pred])]:
            valid_graph_idxs += [graph_idx]
            our_ground_truth_odd += [np.nan]
            gnnexp_ground_truth_odd += [np.nan]
            our_pred_label_odd += [np.nan]
            gnnexp_pred_label_odd += [np.nan]
        else:
            valid_graph_idxs += [graph_idx]
            our_ground_truth_odd += [log_odd(org_p[graph_label]) - log_odd(ours_p[graph_label])]
            gnnexp_ground_truth_odd += [log_odd(org_p[graph_label]) - log_odd(gnnexp_p[graph_label])]
            our_pred_label_odd += [log_odd(org_p[org_pred]) - log_odd(ours_p[org_pred])]
            gnnexp_pred_label_odd += [log_odd(org_p[org_pred]) - log_odd(gnnexp_p[org_pred])]
    pred_prob = np.stack(pred_prob, axis=0)
    np.savetxt("log/%s/prob.csv"%(prog_args.test_out), pred_prob, delimiter=",")
    merged_loss = np.stack([valid_graph_idxs, org_losses, extracted_losses, ours_losses, gnnexp_losses], axis=1)
    np.savetxt("log/%s/loss.csv"%(prog_args.test_out), merged_loss, delimiter=",")
    merged_correct = np.stack([valid_graph_idxs, org_corrects, extracted_corrects, ours_corrects, gnnexp_corrects], axis=1)
    np.savetxt("log/%s/acc.csv"%(prog_args.test_out), merged_correct, delimiter=",")
    merged_odd = np.stack([valid_graph_idxs, our_ground_truth_odd, gnnexp_ground_truth_odd, our_pred_label_odd, gnnexp_pred_label_odd])
    np.savetxt("log/%s/log_odd.csv"%(prog_args.test_out), merged_odd, delimiter=",")
    print("{0:<25}{1:<25}{2:<25}{3:<25}".format("name", "loss", "accuracy", "f1_score"))
    print("{0:<25}{1:<25}{2:<25}{3:<25}".format("org_adj", np.mean(org_losses), np.mean(org_corrects), metrics.f1_score(np.ones(len(org_corrects)), org_corrects)))
    print("{0:<25}{1:<25}{2:<25}{3:<25}".format("extracted_adj", np.mean(extracted_losses), np.mean(extracted_corrects), metrics.f1_score(np.ones(len(extracted_corrects)), extracted_corrects)))
    print("{0:<25}{1:<25}{2:<25}{3:<25}".format("ours_adj%s" % suffix, np.mean(ours_losses), np.mean(ours_corrects), metrics.f1_score(np.ones(len(ours_corrects)), ours_corrects)))
    print("{0:<25}{1:<25}{2:<25}{3:<25}".format("gnnexp_adj%s"% suffix, np.mean(gnnexp_losses), np.mean(gnnexp_corrects), metrics.f1_score(np.ones(len(gnnexp_corrects)), gnnexp_corrects)))
    print()
    print("{0:<25}{1:<25}{2:<25}".format("name", "mean", "std"))
    print("{0:<25}{1:<25}{2:<25}".format("our_ground_truth_odd", np.nanmean(our_ground_truth_odd), np.nanstd(our_ground_truth_odd)))
    print("{0:<25}{1:<25}{2:<25}".format("gnnexp_ground_truth_odd", np.nanmean(gnnexp_ground_truth_odd), np.nanstd(gnnexp_ground_truth_odd)))
    print("{0:<25}{1:<25}{2:<25}".format("our_pred_label_odd", np.nanmean(our_pred_label_odd), np.nanstd(our_pred_label_odd)))
    print("{0:<25}{1:<25}{2:<25}".format("gnnexp_pred_label_odd", np.nanmean(gnnexp_pred_label_odd), np.nanstd(gnnexp_pred_label_odd)))

    with open("log/%s/output.log"%(prog_args.test_out), 'w') as f:
        f.write("{0:<25}{1:<25}{2:<25}{3:<25}\n".format("name", "loss", "accuracy", "f1_score"))
        f.write("{0:<25}{1:<25}{2:<25}{3:<25}\n".format("org_adj", np.mean(org_losses), np.mean(org_corrects), metrics.f1_score(np.ones(len(org_corrects)), org_corrects)))
        f.write("{0:<25}{1:<25}{2:<25}{3:<25}\n".format("extracted_adj", np.mean(extracted_losses), np.mean(extracted_corrects), metrics.f1_score(np.ones(len(extracted_corrects)), extracted_corrects)))
        f.write("{0:<25}{1:<25}{2:<25}{3:<25}\n".format("ours_adj%s" % suffix, np.mean(ours_losses), np.mean(ours_corrects), metrics.f1_score(np.ones(len(ours_corrects)), ours_corrects)))
        f.write("{0:<25}{1:<25}{2:<25}{3:<25}\n".format("gnnexp_adj%s"% suffix, np.mean(gnnexp_losses), np.mean(gnnexp_corrects), metrics.f1_score(np.ones(len(gnnexp_corrects)), gnnexp_corrects)))
        f.write("\n")
        f.write("{0:<25}{1:<25}{2:<25}\n".format("name", "mean", "std"))
        f.write("{0:<25}{1:<25}{2:<25}\n".format("our_ground_truth_odd", np.nanmean(our_ground_truth_odd), np.nanstd(our_ground_truth_odd)))
        f.write("{0:<25}{1:<25}{2:<25}\n".format("gnnexp_ground_truth_odd", np.nanmean(gnnexp_ground_truth_odd), np.nanstd(gnnexp_ground_truth_odd)))
        f.write("{0:<25}{1:<25}{2:<25}\n".format("our_pred_label_odd", np.nanmean(our_pred_label_odd), np.nanstd(our_pred_label_odd)))
        f.write("{0:<25}{1:<25}{2:<25}\n".format("gnnexp_pred_label_odd", np.nanmean(gnnexp_pred_label_odd), np.nanstd(gnnexp_pred_label_odd)))

if __name__ == "__main__":
    main()

