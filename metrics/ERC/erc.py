import os

import numpy as np


from config.cfg import config
from metrics.ERC.utils import *


def quality_eval(
    list_method_compares, 
    pair_list_path, embedding_dir, list_path_score, output_dir="output_dir", FMR=1e-3
):
    method_names = list_method_compares
    fnmrs_list_2 = []
    method_labels = []

    for index, method_name in enumerate(method_names):
        desc = False if method_name == "PFE" else True

        path_score = list_path_score[index]

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
                f"{method_name}_{config.name_dataset}_{config.name_face_recog_model}_fnmr.npy",
            ),
            fnmr,
        )
    print(len(fnmrs_list_2))
    save_pdf(
        fnmrs_list_2,
        method_labels,
        model= config.name_face_recog_model,
        output_dir=output_dir,
        fmr=FMR,
        db= config.name_dataset,
    )
