from metrics.ERC.utils import load_quality_pair, load_feat_pair
from metrics.ERC.erc import quality_eval
from process_dataset.xqlfw_dataset import process_xqlf_pairs

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
    "/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_XQLFW/pairs_image_list.txt",
    embedding_dir= "features_temp",
    path_score="score_file.txt",
    output_dir="output_dir_3",
    FMR = 1e-3
)

# process_xqlf_pairs(
#     output_path_dir="data/processed_XQLFW/images/",
#     image_path_list="data/processed_XQLFW/image_path_list.txt",
#     pair_list= "data/processed_XQLFW/pairs_image_list.txt",
#     dataset_folder= "/data/disk2/tanminh/CR-FIQA/data/XQLFW/xqlfw_aligned_112",
#     pairs_list_path= "/data/disk2/tanminh/CR-FIQA/data/XQLFW/xqlfw_pairs.txt"
# )

# import torchvision.transforms as T 
# from PIL import Image 

# transform = T.Compose([
#     T.Resize((112, 112)),
#     T.ToTensor(),
#     T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
# ])

# img = Image.open("data/processed_XQLFW/images/Aaron_Eckhart_0001.jpg")
# out = transform(img) 
# print(out.shape)


# from models.recognition_model.Elastic_model.elasticface import ElasticFaceModel
# import torch 
# from models.recognition_model.extract_embed import extract_features, save_features
# import glob 

# model = ElasticFaceModel(pretrained="/data/disk2/tanminh/CR-FIQA/pretrained/295672backbone.pth")
# # print(output.shape)
# image_path_list= glob.glob("data/processed_XQLFW/images/*.jpg")

# features = extract_features(
#     model= model, 
#     image_path_list= image_path_list,
#     batch_size= 16,
#     device= "cuda"
# )

# save_features(
#     output_dir="features_temp", 
#     ls_features=features, 
#     list_name= image_path_list
# )


# import torch 
# import glob 
# from models.quality_model.CRFIQA.crfiqa import CR_FIQA_Model 
# from models.quality_model.extract_score import extract_scores, save_scores

# model = CR_FIQA_Model(pretrained="/data/disk2/tanminh/CR-FIQA/pretrained/32572backbone.pth")

# image_path_list= glob.glob("data/processed_XQLFW/images/*.jpg")

# output_scores = extract_scores(
#     model= model, 
#     image_path_list= image_path_list,
#     batch_size=16,
#     device= "cuda"
# )

# print(output_scores)

# save_scores(
#     output_file="score_file.txt",
#     ls_scores= output_scores,
#     list_name= image_path_list
# )

