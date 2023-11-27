import glob

import torch 

from config.cfg import config 
from models.recognition_model.Elastic_model.elasticface import ElasticFaceModel
from models.recognition_model.extract_embed import extract_features, save_features



model = ElasticFaceModel(pretrained=config.pretrained_face_recog)
# print(output.shape)

image_path_list= glob.glob(config.output_path_dir_images + "/*.jpg")
print("[INFO]: Prcess image : ",config.output_path_dir_images,  len(image_path_list))
features = extract_features(
    model= model, 
    image_path_list= image_path_list,
    batch_size= 16,
    device= "cuda"
)

save_features(
    output_dir= config.embeding_folder, 
    ls_features=features, 
    list_name= image_path_list
)
