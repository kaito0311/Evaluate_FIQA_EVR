from metrics.ERC.utils import load_quality_pair, load_feat_pair
from metrics.ERC.erc import quality_eval


# output = load_quality_pair(
#     pair_path= "/data/disk2/tanminh/CR-FIQA/data/quality_data/XQLFW/pair_list.txt",
#     path_score= "/data/disk2/tanminh/CR-FIQA/data/quality_data/XQLFW/CRFIQAS_XQLFW.txt",
#     dataset= None,
#     args= None
# )

# output = load_feat_pair(
#     root=pair_path= "/data/disk2/tanminh/CR-FIQA/data/quality_data/XQLFW/pair_list.txt",
#     pair_path= "/data/disk2/tanminh/CR-FIQA/data/quality_data/XQLFW/pair_list.txt"
# )

# print(output)

quality_eval(
    "/data/disk2/tanminh/CR-FIQA/data/quality_data/XQLFW/pair_list.txt",
    embedding_dir= "/data/disk2/tanminh/CR-FIQA/data/quality_embeddings/XQLFW_ElasticFaceModel",
    path_score="/data/disk2/tanminh/CR-FIQA/data/quality_data/XQLFW/CRFIQAS_XQLFW.txt",
    output_dir="output_dir",
    FMR = 1e-3
)