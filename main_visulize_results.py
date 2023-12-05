from metrics.ERC.erc import quality_eval, quality_eval_nist
from config.cfg import config

quality_eval(
    ["FaceQAN", "CR-QIFA", "FaceQAN-r160", "DiffQIFA", "CR-FIQA-Custom"],
    config.pair_list_path,
    # embedding_dir= "/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_XQLFW/Elastic_embedding",
    embedding_dir= "/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_XQLFW/IMINT_V5_PLUS_embedding",
    list_path_score=["/data/disk2/tanminh/FaceQAN/src/FaceQAN_XQLFW.txt",
                     "/data/disk2/tanminh/CR-FIQA/data/quality_data/XQLFW/CRFIQAS_XQLFW.txt", \
                     "/data/disk2/tanminh/Evaluate_FIQA_EVR/output_result/FaceQAN_XQLFW_r160.txt",
                    #  "/data/disk2/tanminh/Evaluate_FIQA_EVR/output_result/testing_val_1.txt",
                    
                     "/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_XQLFW/DiffFIQA_scores.txt",
                      f"/data/disk2/tanminh/IFQA/output_file_score.txt",
                    #   "/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_XQLFW/FIQ_IMINT_scores.txt"
                      ],
    output_dir="output_dir_7",
    FMR = 1e-3
)
# quality_eval_nist(
#     ["FaceQAN", "CR-QIFA", "FaceQAN-r160", "DiffQIFA", "CR-FIQA-Custom"],
#     config.pair_list_path,
#     # embedding_dir= "/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_XQLFW/Elastic_embedding",
#     embedding_dir= "/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_XQLFW/IMINT_V5_PLUS_embedding",
#     list_path_score=["/data/disk2/tanminh/FaceQAN/src/FaceQAN_XQLFW.txt",
#                      "/data/disk2/tanminh/CR-FIQA/data/quality_data/XQLFW/CRFIQAS_XQLFW.txt", \
#                      "/data/disk2/tanminh/Evaluate_FIQA_EVR/output_result/FaceQAN_XQLFW_r160.txt",
#                     #  "/data/disk2/tanminh/Evaluate_FIQA_EVR/output_result/testing_val_1.txt",
                    
#                      "/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_XQLFW/DiffFIQA_scores.txt",
#                       f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_XQLFW/CR_ONTOP_10K_9KITER_scores.txt",
#                     #   "/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_XQLFW/FIQ_IMINT_scores.txt"
#                       ],
#     output_dir="output_dir_nist",
# )
# quality_eval(
#     ["CR-QIFA", "DIFF-FIQA_"],
#     config.pair_list_path,
#     embedding_dir= config.embeding_folder,
#     list_path_score=["/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_CPLFW/CR-FIQA_scores.txt",
#                      "/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_CPLFW/DIFF-FIQA_scores.txt"
#                     ],
#     output_dir="output_dir_5",
#     FMR = 1e-3
# )


# name_dataset = config.name_dataset

# # list_name_model = ["CR-QIFA", "DiffFIQA", "IMINT"]
# # list_name_model = ["CR-QIFA", "DiffFIQA", "IMINT", "CR-FIQA-ONTOP", "CR-FIQA-ONTOP-4K", "CR-FIQA-ONTOP-10K"]
# list_name_model = ["CR-QIFA", "DiffFIQA", "IMINT", "CR-ONTOP-4K", "CR-ONTOP-9K", "CR-ONTOP-12K", "CR-ONTOP-14K"]
# list_path_score = [
#     f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_{name_dataset}/CR-FIQA_scores.txt",
#     f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_{name_dataset}/DiffFIQA_scores.txt",
#     f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_{name_dataset}/FIQ_IMINT_scores.txt",
#     # f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_XQLFW/score_image_XQLFW_Elastic_ensemble.txt",
#     # f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_{name_dataset}/score_image_CALFW_Enet1.txt",
#     # f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_{name_dataset}/new_cr_fiqa_ori_scores.txt",
#     # f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_{name_dataset}/new_cr_fiqa_ori_fi_scores.txt",
#     # "/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_CFP_FP/score_image_CFP_FP_Enet.txt",
#     # f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_{name_dataset}/cr_fiqa_ontop_scores.txt",
#     f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_{name_dataset}/new_cr_fiqa_ori_fi_scores.txt",
#     f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_{name_dataset}/CR_ONTOP_10K_9KITER_scores.txt",
#     f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_{name_dataset}/CR_ONTOP_10K_12KITER_scores.txt",
#     f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_{name_dataset}/CR_ONTOP_10K_14KITER_scores.txt",
#     # f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_{name_dataset}/cr_fiqa_ontop_4K_scores.txt",
#     # f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_{name_dataset}/cr_fiqa_ontop_10k_scores.txt",
#     # f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_{name_dataset}/cr_fiqa_ontop_14k_scores.txt",
#     # f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_{name_dataset}/cr_fiqa_ontop_16k_scores.txt",
#     # f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_{name_dataset}/cr_fiqa_ontop_18k_scores.txt",
#     # f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_{name_dataset}/custom_crfiqa_scores.txt",
#     # f"/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_{name_dataset}/custom_crfiqa_260000_4_scores.txt",
# ]
# save_output_dir = "output_dir_6"
# assert len(list_name_model) == len(list_path_score)

# quality_eval(
#     list_name_model,
#     config.pair_list_path,
#     embedding_dir=config.embeding_folder,
#     list_path_score=list_path_score,
#     output_dir= save_output_dir,
#     FMR=1e-3,
# )
