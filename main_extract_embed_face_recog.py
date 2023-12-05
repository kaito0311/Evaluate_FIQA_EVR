import glob

import torch

from config.cfg import config
from models.recognition_model.Elastic_model.elasticface import ElasticFaceModel
from models.recognition_model.imintv5_plus.imint import ONNX_IMINT
from models.recognition_model.extract_embed import extract_features, save_features


print("[INFO] Face Recognize model: ", config.name_face_recog_model.lower())
if config.name_face_recog_model.lower() == "imintv5":
    model = ONNX_IMINT()
elif config.name_face_recog_model.lower() == "elastic":
    model = ElasticFaceModel(pretrained=config.pretrained_face_recog)
else:
    raise ValueError("please assign true name model")

# print(output.shape)

image_path_list = glob.glob(config.output_path_dir_images + "/*.jpg")
print("[INFO]: Prcess image : ", config.output_path_dir_images, len(image_path_list))
features = extract_features(
    model=model, image_path_list=image_path_list, batch_size=16, device="cuda"
)
print("[INFO] Save to ", config.embeding_folder)
save_features(
    output_dir=config.embeding_folder, ls_features=features, list_name=image_path_list
)
