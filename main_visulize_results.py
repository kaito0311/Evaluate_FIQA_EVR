from metrics.ERC.erc import quality_eval
from config.cfg import config 

quality_eval(
    ["FaceQAN", "CR-QIFA", "FaceQAN-r160", "hungpham-iresnet_100", "DiffQIFA"],
    config.pair_list_path,
    embedding_dir= "/data/disk2/tanminh/CR-FIQA/data/quality_embeddings/XQLFW_ElasticFaceModel",
    list_path_score=["/data/disk2/tanminh/FaceQAN/src/FaceQAN_XQLFW.txt", 
                     "/data/disk2/tanminh/CR-FIQA/data/quality_data/XQLFW/CRFIQAS_XQLFW.txt", \
                     "/data/disk2/tanminh/Evaluate_FIQA_EVR/FaceQAN_XQLFW_r160.txt",
                     "testing_val_1.txt",
                     "/data/disk2/tanminh/Evaluate_FIQA_EVR/log/testing_val.txt"],
    output_dir="output_dir_3",
    FMR = 1e-3
)