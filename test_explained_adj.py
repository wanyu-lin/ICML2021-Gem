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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'gnnexp'))

import models
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils


color_map = ['gray', 'blue', 'purple', 'red', 'slategray', 'green', 'orange', 'olive']

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
        "--writer",
        dest="writer",
        action="store_const",
        const=True,
        default=False,
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
    parser.add_argument('--top_k', type=int, default=None, help='Keep k edges of prediction.')
    parser.add_argument('--distillation', type=str, default=None, help='Path of distillation.')
    parser.add_argument('--exp_out', type=str, default=None, help='Path of explainer output.')
    parser.add_argument('--test_out', type=str, default=None, help='Path of test output.')

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
    # load state_dict (obtained by model.state_dict() when saving checkpoint)
    model.load_state_dict(ckpt["model_state"])

    if prog_args.test_out is None:
        prog_args.test_out = prog_args.exp_out
    features = torch.from_numpy(cg_dict["feat"]).float()
    adj = torch.from_numpy(cg_dict["adj"]).float()
    label = torch.from_numpy(cg_dict["label"][0]).long()
    model.eval()
    preds, _ = model(features, adj)
    _, pred_label = torch.max(preds[0], dim=1)
    neg_nodes = np.where(pred_label.detach().numpy() != label)[0]
    ce = torch.nn.CrossEntropyLoss(reduction='none')
    G = nx.from_numpy_matrix(cg_dict["adj"][0])
    softmax = torch.nn.Softmax(dim=0)
    def log_odd(p):
        return (torch.log(p) - torch.log(1-p)).item()

    def evaluate_adj(node_idx_new, feat, adj, _label, losses, corrects):
        with torch.no_grad():
            pred, _ = model(feat, adj)
            loss = ce(pred[0], _label)
            _, pred_label = torch.max(pred[0], 1)
            correct = (pred_label[node_idx_new] == _label[node_idx_new]).float().sum()
            losses.append(loss[node_idx_new].item())
            corrects.append(correct.item())
        return pred_label[node_idx_new], softmax(pred[0][node_idx_new])

    def plot(node_idx_new, adj, mapping, pred_label, pos, ax):
        node_idx = mapping[node_idx_new]
        sub_G = nx.from_numpy_matrix(adj[0].numpy())
        sub_G = nx.relabel_nodes(sub_G, dict(enumerate(mapping)))
        sub_G = sub_G.subgraph(nx.node_connected_component(sub_G, node_idx))
        node_labels = dict([(n, n) for n in sub_G])
        fixed_nodes = pos.keys()
        fixed_nodes = [n for n in sub_G if n in fixed_nodes]
        if fixed_nodes:
            _pos = nx.spring_layout(sub_G, pos=pos, fixed=fixed_nodes)
        else:
            _pos = nx.spring_layout(sub_G)
        pos = _pos
        node_list = set(list(sub_G.nodes)) - set([node_idx])
        node_color = [color_map[cg_dict["label"][0][n]] for n in node_list]
        nx.draw_networkx_nodes(sub_G, pos, nodelist=node_list, alpha=0.6, node_color=node_color, node_size=100, ax=ax)
        nx.draw_networkx_nodes(sub_G, pos, nodelist=[node_idx], node_color=[color_map[pred_label]], edgecolors='YellowGreen', linewidths=2, node_size=100, ax=ax)
        nx.draw_networkx_labels(sub_G,pos,node_labels,font_size=8, font_color='w', ax=ax)
        nx.draw_networkx_edges(sub_G, pos, alpha=0.4, ax=ax)
        return pos
    
    if prog_args.dataset == 'syn2':
        node_idxs = np.where(np.logical_and(label != 0, label != 4))[0]
    else:
        node_idxs = np.where(label != 0)[0]

    org_losses = []
    extracted_losses = []
    ours_losses = []
    gnnexp_losses = []
    num_nodes = adj.shape[-1]
    org_corrects = []
    extracted_corrects = []
    ours_corrects = []
    gnnexp_corrects = []
    pred_prob = []
    our_ground_truth_odd = []
    our_pred_label_odd = []
    gnnexp_ground_truth_odd = []
    gnnexp_pred_label_odd = []
    # node_idxs = [709]
    shutil.rmtree('log/%s' % prog_args.test_out, ignore_errors=True)
    os.makedirs('log/%s' % prog_args.test_out, exist_ok=True)
    gnnexp_result_path = os.path.join('explanation', 'gnnexp', io_utils.gen_explainer_prefix(prog_args))
    valid_node_idxs = []
    for node_idx in node_idxs:
        if not os.path.exists("explanation/%s/node%d_pred.csv"%(prog_args.exp_out, node_idx)):
            continue
        valid_node_idxs += [node_idx]
        data = torch.load("distillation/%s/node_%d.ckpt" %(prog_args.distillation, node_idx), map_location=device)
        mapping = data['mapping']
        node_idx_new = np.where(mapping == node_idx)[0][0]
        sub_label = torch.from_numpy(data['label']).to(device).long()
        sub_adj = torch.tensor(data['adj'], dtype=torch.float, device=device)
        org_adj = sub_adj.unsqueeze(0)
        extracted_adj = torch.from_numpy(data['adj_y']).to(device).unsqueeze(0).float()
        sub_feat = data['features'].unsqueeze(0)
        
        recovered = np.loadtxt("explanation/%s/node%d_pred.csv"%(prog_args.exp_out, node_idx), delimiter=',')
        ours_adj = recovered * sub_adj.numpy()
        np.fill_diagonal(ours_adj, 0)
        G2 = nx.from_numpy_array(ours_adj)
        d = nx.get_edge_attributes(G2,'weight')
        edges,weights = zip(*{k: d[k] for k in sorted(d, key=d.get)}.items())
        if prog_args.top_k is not None:
            sub_edges = edges[-prog_args.top_k:]
        else:
            sub_edges = edges
        G3 = nx.Graph()
        G3.add_nodes_from(list(G2.nodes))
        G3.add_edges_from(sub_edges)
        ours_adj = torch.from_numpy(nx.to_numpy_matrix(G3, weight=None)).unsqueeze(0).float()

        org_pred, org_p = evaluate_adj(node_idx_new, sub_feat, org_adj, sub_label, org_losses, org_corrects)
        extracted_pred, extracted_p = evaluate_adj(node_idx_new, sub_feat, extracted_adj, sub_label, extracted_losses, extracted_corrects)
        ours_pred, ours_p = evaluate_adj(node_idx_new, sub_feat, ours_adj, sub_label, ours_losses, ours_corrects)

        fname = 'masked_adj_' + (
                        'node_idx_'+str()+'graph_idx_'+str(0)+'.ckpt')
        gnnexp_data = torch.load(os.path.join(gnnexp_result_path, 'masked_adj_node_idx_%sgraph_idx_0.ckpt' % node_idx), map_location=device)
        # gnnexp_adj = torch.from_numpy(gnnexp_data['adj']).float().unsqueeze(0)
        gnnexp_mapping = gnnexp_data['node_idx']
        gnnexp_feat = features[0][gnnexp_mapping]
        gnnexp_label = gnnexp_data['label'][0]

        gnnexp_adj = gnnexp_data['adj']

        ### Top K or threshold ###
        num_nodes = gnnexp_adj.shape[-1]
        gnnexp_adj_topk = np.zeros((num_nodes,num_nodes))
        if prog_args.top_k is not None:
            adj_threshold_num = prog_args.top_k * 2
            neigh_size = len(gnnexp_adj[gnnexp_adj > 0])
            threshold_num = min(neigh_size, adj_threshold_num)
            threshold = np.sort(gnnexp_adj[gnnexp_adj > 0])[-threshold_num]
        else:
            threshold = 1e-6
        _m = []
        for i, n in enumerate(mapping):
            _m += [np.where(gnnexp_mapping == n)[0][0]]
        _m_dict = dict([(n, i) for i, n in enumerate(_m)])
        for i in range(num_nodes):
            for j in range(num_nodes):
                if gnnexp_adj[i,j] >= threshold:
                    gnnexp_adj_topk[_m_dict[i], _m_dict[j]] = 1
        gnnexp_adj = torch.from_numpy(gnnexp_adj_topk).unsqueeze(0).float()
        gnnexp_mapping = gnnexp_mapping[_m]
        gnnexp_label = gnnexp_label[_m]
        gnnexp_feat = gnnexp_feat[_m].unsqueeze(0)
        gnnexp_node_idx_new = np.where(gnnexp_mapping == node_idx)[0][0]
        # gnnexp_pred, gnnexp_p = evaluate_adj(node_idx, features, gnnexp_adj, label, gnnexp_losses, gnnexp_corrects)
        gnnexp_pred, gnnexp_p = evaluate_adj(gnnexp_node_idx_new, gnnexp_feat, gnnexp_adj, gnnexp_label, gnnexp_losses, gnnexp_corrects)
        pred_prob += [[node_idx] + list(org_p) + list(extracted_p) + list(ours_p) + list(gnnexp_p)]
        node_label = cg_dict["label"][0][node_idx]
        our_ground_truth_odd += [log_odd(org_p[node_label]) - log_odd(ours_p[node_label])]
        gnnexp_ground_truth_odd += [log_odd(org_p[node_label]) - log_odd(gnnexp_p[node_label])]
        our_pred_label_odd += [log_odd(org_p[org_pred]) - log_odd(ours_p[org_pred])]
        gnnexp_pred_label_odd += [log_odd(org_p[org_pred]) - log_odd(gnnexp_p[org_pred])]
        if prog_args.plot:
            fig, axes = plt.subplots(1, 4, figsize=(19,5))
            fig.suptitle("Node %d, label=%d" % (node_idx, cg_dict["label"][0][node_idx]))
            axes[0].set_title("Original adj", color="Black" if (org_pred==node_label).item() else "Red")
            axes[1].set_title("Extracted adj", color="Black" if (extracted_pred==node_label).item() else "Red")
            axes[2].set_title("Our Explainer adj", color="Black" if (ours_pred==node_label).item() else "Red")
            axes[3].set_title("GNNExplainer adj", color="Black" if (gnnexp_pred==node_label).item() else "Red")
            pos = dict({})
            pos = plot(node_idx_new, org_adj, mapping, org_pred, pos, axes[0])
            pos = plot(node_idx_new, extracted_adj, mapping, extracted_pred, pos, axes[1])
            pos = plot(node_idx_new, ours_adj, mapping, ours_pred, pos, axes[2])
            pos = plot(gnnexp_node_idx_new, gnnexp_adj, gnnexp_mapping, gnnexp_pred, pos, axes[3])
            patches = []
            for i in range(num_classes):
                patches += [mpatches.Patch(color=color_map[i], label=i)]
            plt.axis('off')
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 0.0), loc="lower right", fontsize=10)
            
            if org_pred == extracted_pred == ours_pred == gnnexp_pred == node_label:
                plt.savefig("log/%s/correct_pred_node%d.pdf" %(prog_args.test_out, node_idx))
            else:
                plt.savefig("log/%s/wrong_pred_node%d.pdf" %(prog_args.test_out, node_idx))
            plt.clf()
            plt.close(fig)

    merged_loss = np.stack([valid_node_idxs, org_losses, extracted_losses, ours_losses, gnnexp_losses], axis=1)
    merged_correct = np.stack([valid_node_idxs, org_corrects, extracted_corrects, ours_corrects, gnnexp_corrects], axis=1)
    pred_prob = np.stack(pred_prob, axis=0)
    np.savetxt("log/%s/prob.csv"%(prog_args.test_out), pred_prob, delimiter=",")
    np.savetxt("log/%s/loss.csv"%(prog_args.test_out), merged_loss, delimiter=",")
    np.savetxt("log/%s/acc.csv"%(prog_args.test_out), merged_correct, delimiter=",")
    merged_odd = np.stack([our_ground_truth_odd, gnnexp_ground_truth_odd, our_pred_label_odd, gnnexp_pred_label_odd])
    np.savetxt("log/%s/log_odd.csv"%(prog_args.test_out), merged_odd, delimiter=",")
    print("{0:<25}{1:<25}{2:<25}{3:<25}".format("name", "loss", "accuracy", "f1_score"))
    print("{0:<25}{1:<25}{2:<25}{3:<25}".format("org_adj", np.mean(org_losses), np.mean(org_corrects), metrics.f1_score(np.ones(len(org_corrects)), org_corrects)))
    print("{0:<25}{1:<25}{2:<25}{3:<25}".format("extracted_adj", np.mean(extracted_losses), np.mean(extracted_corrects), metrics.f1_score(np.ones(len(extracted_corrects)), extracted_corrects)))
    print("{0:<25}{1:<25}{2:<25}{3:<25}".format("ours_adj(top_k=%s)" % prog_args.top_k, np.mean(ours_losses), np.mean(ours_corrects), metrics.f1_score(np.ones(len(ours_corrects)), ours_corrects)))
    print("{0:<25}{1:<25}{2:<25}{3:<25}".format("gnnexp_adj(top_k=%s)" % prog_args.top_k, np.mean(gnnexp_losses), np.mean(gnnexp_corrects), metrics.f1_score(np.ones(len(gnnexp_corrects)), gnnexp_corrects)))
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
        f.write("{0:<25}{1:<25}{2:<25}{3:<25}\n".format("ours_adj(top_k=%s)" % prog_args.top_k, np.mean(ours_losses), np.mean(ours_corrects), metrics.f1_score(np.ones(len(ours_corrects)), ours_corrects)))
        f.write("{0:<25}{1:<25}{2:<25}{3:<25}\n".format("gnnexp_adj(top_k=%s)" % prog_args.top_k, np.mean(gnnexp_losses), np.mean(gnnexp_corrects), metrics.f1_score(np.ones(len(gnnexp_corrects)), gnnexp_corrects)))
        f.write("\n")
        f.write("{0:<25}{1:<25}{2:<25}\n".format("name", "mean", "std"))
        f.write("{0:<25}{1:<25}{2:<25}\n".format("our_ground_truth_odd", np.nanmean(our_ground_truth_odd), np.nanstd(our_ground_truth_odd)))
        f.write("{0:<25}{1:<25}{2:<25}\n".format("gnnexp_ground_truth_odd", np.nanmean(gnnexp_ground_truth_odd), np.nanstd(gnnexp_ground_truth_odd)))
        f.write("{0:<25}{1:<25}{2:<25}\n".format("our_pred_label_odd", np.nanmean(our_pred_label_odd), np.nanstd(our_pred_label_odd)))
        f.write("{0:<25}{1:<25}{2:<25}\n".format("gnnexp_pred_label_odd", np.nanmean(gnnexp_pred_label_odd), np.nanstd(gnnexp_pred_label_odd)))

if __name__ == "__main__":
    main()

