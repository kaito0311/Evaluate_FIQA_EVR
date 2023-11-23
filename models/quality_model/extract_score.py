import os

import cv2
import torch 
import numpy as np
from tqdm import tqdm 
import torchvision.transforms as T 
from sklearn.preprocessing import normalize
from PIL import Image



def extract_scores(model, image_path_list, img_size=(112, 112), batch_size=4, device="cpu"):
    count = 0 
    num_batch = int(len(image_path_list) // batch_size)
    scores = [] 
    model.to(device)

    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def convert_to_tensor(path_image):
        image = Image.open(path_image)
        tensor = transform(image) 
        tensor = torch.unsqueeze(tensor, 0) 
        tensor = tensor.to(device)
        return tensor 

    for i in tqdm(range(0, len(image_path_list), batch_size)):
        if count < num_batch: 
            tmp_list = image_path_list[i:i+batch_size]
        else:
            tmp_list = image_path_list[i:]
        
        count += 1 

        list_image_tensor = [] 

        for image_path in tmp_list:
            list_image_tensor.append(convert_to_tensor(image_path)) 
        
        batch_tensor = torch.cat(list_image_tensor, dim=0) 
        # NOTE: Can change depends on output of model. 
        batch_scores = model(batch_tensor) 
        if isinstance(batch_scores, torch.Tensor):
            batch_scores = batch_scores.detach().cpu().numpy() 
        scores.append(batch_scores) 

    scores = np.vstack(scores)
    
    return scores 


def save_scores(output_file, ls_scores, list_name): 
    assert len(list_name) == len(ls_scores) 
    if len(os.path.dirname(output_file)) > 0:
        os.makedirs(os.path.dirname(output_file), exist_ok= True) 
    file = open(output_file, "w")
    for count, name in tqdm(enumerate(list_name)): 
        basename  = os.path.basename(name) 
        basename = basename.split(".")[0]

        # np.save(os.path.join(output_dir, basename), ls_scores[count])
        file.write(
            basename + ".jpg" + " " + str(ls_scores[count][0]) + "\n"
        )
    file.close() 