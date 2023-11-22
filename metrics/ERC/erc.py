import os

import numpy as np


from config.cfg import config
from metrics.ERC.utils import *


def quality_eval(
    pair_list_path, embedding_dir, path_score, output_dir="output_dir", FMR=1e-3
):
    method_names = ["ABCXYZ"]
    fnmrs_list_2 = []
    method_labels = []

    for method_name in method_names:
        desc = False if method_name == "PFE" else True

        feat_pairs = load_feat_pair(
            pair_list_path,
            embedding_dir,
        )

        quality_scores = load_quality_pair(pair_list_path, path_score, None, None)

        fnmr, unconsidered_rates = getFNMRFixedFMR(
            feat_pairs=feat_pairs,
            qlts=quality_scores,
            FMR=FMR,
            dist_type="cosine",
            desc=desc,
        )
        fnmrs_list_2.append(fnmr)
        method_labels.append(f"{method_name}")

        os.makedirs(
            os.path.join(output_dir, "fnmr"),
            exist_ok=True,
        )
        np.save(
            os.path.join(
                output_dir,
                "fnmr",
                f"{method_name}_fnmr.npy",
            ),
            fnmr,
        )

    save_pdf(
        fnmrs_list_2,
        method_labels,
        model="ABC_XYS",
        output_dir=output_dir,
        fmr=FMR,
        db="DATASET",
    )
