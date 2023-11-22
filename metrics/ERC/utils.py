import os
import math 

import sklearn
import numpy as np 
from sklearn import metrics
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties

from metrics.ERC.roc import * 


def load_feat_pair(pair_path, root):
    pairs = {}
    with open(pair_path, "r") as f:
        lines = f.readlines()
        for idex in range(len(lines)):
            name1 = lines[idex].rstrip().split()[0]
            name1 = name1.split(".")[0]
            name2 = lines[idex].rstrip().split()[1]
            name2 = name2.split(".")[0]
            is_same = int(lines[idex].rstrip().split()[2])
            feat_a = np.load(os.path.join(root, f"{name1}.npy"))
            feat_b = np.load(os.path.join(root, f"{name2}.npy"))
            pairs[idex] = [feat_a, feat_b, is_same]
    print("All features are loaded")
    return pairs

def load_quality(path_score):
    quality = {}
    with open(path_score, "r") as f:
        lines = f.readlines()
        for l in lines:
            scores = l.split()[1].strip()
            name_wo_ext = l.split()[0].strip()
            name_wo_ext = os.path.basename(name_wo_ext)
            name_wo_ext = name_wo_ext.split(".")[0]
            quality[name_wo_ext] = scores
    return quality


def load_quality_pair(pair_path, path_score, dataset, args):
    pairs_quality = []
    quality = load_quality(path_score)
    with open(pair_path, "r") as f:
        lines = f.readlines()
        for idex in range(len(lines)):
            name1 = lines[idex].rstrip().split()[0]
            name1 = name1.split(".")[0]
            name2 = lines[idex].rstrip().split()[1]
            name2 = name2.split(".")[0]
            qlt = min(
                float(quality.get(name1)),
                float(quality.get(name2)),
            )
            pairs_quality.append(qlt)
    return pairs_quality


def distance_(embeddings0, embeddings1, dist="cosine"):
    # Distance based on cosine similarity
    if dist == "cosine":
        dot = np.sum(np.multiply(embeddings0, embeddings1), axis=1)
        norm = np.linalg.norm(embeddings0, axis=1) * np.linalg.norm(embeddings1, axis=1)
        # shaving
        similarity = np.clip(dot / norm, -1.0, 1.0)
        dist = np.arccos(similarity) / math.pi
    else:
        embeddings0 = sklearn.preprocessing.normalize(embeddings0)
        embeddings1 = sklearn.preprocessing.normalize(embeddings1)
        diff = np.subtract(embeddings0, embeddings1)
        dist = np.sum(np.square(diff), 1)

    return dist

def calc_score(
    embeddings0, embeddings1, actual_issame, subtract_mean=False, dist_type="cosine"
):
    assert embeddings0.shape[0] == embeddings1.shape[0]
    assert embeddings0.shape[1] == embeddings1.shape[1]

    if subtract_mean:
        mean = np.mean(np.concatenate([embeddings0, embeddings1]), axis=0)
    else:
        mean = 0.0

    dist = distance_(embeddings0, embeddings1, dist=dist_type)
    # sort in a desending order
    pos_scores = np.sort(dist[actual_issame == 1])
    neg_scores = np.sort(dist[actual_issame == 0])
    return pos_scores, neg_scores



def getFNMRFixedTH(feat_pairs, qlts, dist_type="cosine", desc=True):
    embeddings0, embeddings1, targets = [], [], []
    pair_qlt_list = []  # store the min qlt
    for k, v in feat_pairs.items():
        feat_a = v[0]
        feat_b = v[1]
        ab_is_same = int(v[2])
        # convert into np
        np_feat_a = np.asarray(feat_a, dtype=np.float64)
        np_feat_b = np.asarray(feat_b, dtype=np.float64)
        # append
        embeddings0.append(np_feat_a)
        embeddings1.append(np_feat_b)
        targets.append(ab_is_same)

    # evaluate
    embeddings0 = np.vstack(embeddings0)
    embeddings1 = np.vstack(embeddings1)
    targets = np.vstack(targets).reshape(
        -1,
    )
    qlts = np.array(qlts)
    if desc:
        qlts_sorted_idx = np.argsort(qlts)
    else:
        qlts_sorted_idx = np.argsort(qlts)[::-1]

    num_pairs = len(targets)
    unconsidered_rates = np.arange(0, 0.98, 0.05)

    fnmrs_list_2 = []
    fnmrs_list_3 = []
    fnmrs_list_4 = []
    for u_rate in unconsidered_rates:
        hq_pairs_idx = qlts_sorted_idx[int(u_rate * num_pairs) :]
        pos_dists, neg_dists = calc_score(
            embeddings0[hq_pairs_idx],
            embeddings1[hq_pairs_idx],
            targets[hq_pairs_idx],
            dist_type=dist_type,
        )
        fmr100_th, fmr1000_th, fmr10000_th = get_eer_threshold(
            pos_dists, neg_dists, ds_scores=True
        )

        g_true = [g for g in pos_dists if g < fmr100_th]
        fnmrs_list_2.append(1 - len(g_true) / (len(pos_dists)))
        g_true = [g for g in pos_dists if g < fmr1000_th]
        fnmrs_list_3.append(1 - len(g_true) / (len(pos_dists)))
        g_true = [g for g in pos_dists if g < fmr10000_th]
        fnmrs_list_4.append(1 - len(g_true) / (len(pos_dists)))

    return fnmrs_list_2, fnmrs_list_3, fnmrs_list_4, unconsidered_rates



def save_pdf(fnmrs_lists, method_labels, model, output_dir, fmr, db):
    fontsize = 20
    colors = [
        "green",
        "black",
        "orange",
        "plum",
        "cyan",
        "gold",
        "gray",
        "salmon",
        "deepskyblue",
        "red",
        "blue",
        "darkseagreen",
        "seashell",
        "hotpink",
        "indigo",
        "lightseagreen",
        "khaki",
        "brown",
        "teal",
        "darkcyan",
    ]
    STYLES = [
        "--",
        "-.",
        ":",
        "v--",
        "^--",
        ",--",
        "<--",
        ">--",
        "1--",
        "-",
        "-",
        "2--",
        "3--",
        "4--",
        ".--",
        "p--",
        "*--",
        "h--",
        "H--",
        "+--",
        "x--",
        "d--",
        "|--",
        "---",
    ]
    unconsidered_rates = 100 * np.arange(0, 0.98, 0.05)

    fig, ax1 = plt.subplots()  # added
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for i in range(len(fnmrs_lists)):
        print(fnmrs_lists[i])
        plt.plot(
            unconsidered_rates[: len(fnmrs_lists[i])],
            fnmrs_lists[i],
            STYLES[i],
            color=colors[i],
            label=method_labels[i],
        )
        auc_value = metrics.auc(
            np.array(unconsidered_rates[: len(fnmrs_lists[i])] / 100),
            np.array(fnmrs_lists[i]),
        )
        os.makedirs(os.path.join(output_dir, db), exist_ok= True)
        with open(os.path.join(output_dir, db, str(fmr) + "_auc.txt"), "a") as f:
            f.write(
                db + ":" + model + ":" + method_labels[i] + ":" + str(auc_value) + "\n"
            )
    plt.xlabel("Ratio of unconsidered image [%]")

    plt.xlabel("Ratio of unconsidered image [%]", fontsize=fontsize)
    plt.xlim([0, 98])
    plt.xticks(np.arange(0, 98, 10), fontsize=fontsize)
    plt.title(
        f"Testing on {db}, FMR={fmr}" + f" ({model})", fontsize=fontsize
    )  # update : -3
    plt.ylabel("FNMR", fontsize=fontsize)

    axbox = ax1.get_position()
    fig.legend(
        bbox_to_anchor=(axbox.x0 + 0.5 * axbox.width, axbox.y0 - 0.22),
        prop=FontProperties(size=12),
        loc="lower center",
        ncol=6,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, db, db + "_" + str(fmr) + "_" + model + ".png"),
        bbox_inches="tight",
    )
