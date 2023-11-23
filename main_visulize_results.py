from metrics.ERC.erc import quality_eval
from config.cfg import config 

quality_eval(
    ["old", "new"],
    config.pair_list_path,
    embedding_dir= "/data/disk2/tanminh/CR-FIQA/data/quality_embeddings/XQLFW_ElasticFaceModel",
    list_path_score=["/data/disk2/tanminh/CR-FIQA/data/quality_data/XQLFW/CRFIQAS_XQLFW.txt", "data/processed_XQLFW/model_scores.txt"],
    output_dir="output_dir_3",
    FMR = 1e-3
)