import os

import numpy as np


from config.cfg import config
from metrics.ERC.utils import *


def quality_eval(pair_list_path, embedding_dir, path_score, output_dir="output_dir", FMR=1e-3):
    method_names = ["ABCXYZ"]
    fnmrs_list_2 = []
    fnmrs_list_3 = []
    fnmrs_list_4 = []
    method_labels = []

    for method_name in method_names:
        desc = False if method_name == "PFE" else True

        feat_pairs = load_feat_pair(
            pair_list_path,
            embedding_dir,
        )

        quality_scores = load_quality_pair(pair_list_path, path_score, None, None)

        fnmr2, fnmr3, fnmr4, unconsidered_rates = getFNMRFixedTH(
            feat_pairs,
            quality_scores,
            dist_type="cosine",
            desc=desc,
        )
        fnmrs_list_2.append(fnmr2)
        fnmrs_list_3.append(fnmr3)
        fnmrs_list_4.append(fnmr4)
        method_labels.append(f"{method_name}")

        os.makedirs(
            os.path.join(
                output_dir,
                "fnmr"
            ),
            exist_ok=True,
        )
        np.save(
            os.path.join(
                output_dir,
                "fnmr",
                f"{method_name}_fnmr2.npy",
            ),
            fnmr2,
        )
        # np.save(
        #     os.path.join(
        #         output_dir,
        #         "fnmr",
        #         f"{method_name}_fnmr3.npy",
        #     ),
        #     fnmr3,
        # )
        # np.save(
        #     os.path.join(
        #         output_dir,
        #         "fnmr",
        #         f"{method_name}_fnmr4.npy",
        #     ),
        #     fnmr4,
        # )

    save_pdf(
        fnmrs_list_2,
        method_labels,
        model="ABC_XYS",
        output_dir=output_dir,
        fmr=1e-2,
        db="DATASET",
    )
