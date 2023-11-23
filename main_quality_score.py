import torch 
import glob 
from models.quality_model.CRFIQA.crfiqa import CR_FIQA_Model 
from models.quality_model.extract_score import extract_scores, save_scores
from config.cfg import config 

model = CR_FIQA_Model(pretrained= config.pretrain_quality_model)

image_path_list= glob.glob(config.output_path_dir_images + "/*.jpg")

output_scores = extract_scores(
    model= model, 
    image_path_list= image_path_list,
    batch_size=16,
    device= "cuda"
)

print(output_scores)

save_scores(
    output_file= config.path_score,
    ls_scores= output_scores,
    list_name= image_path_list
)

