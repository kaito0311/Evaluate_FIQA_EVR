import os

import cv2
import torch 
import numpy as np
from tqdm import tqdm 
import torchvision.transforms as T 
from sklearn.preprocessing import normalize
from PIL import Image



def extract_features(model, image_path_list, img_size=(112, 112), batch_size=4, device="cpu"):
    count = 0 
    num_batch = int(len(image_path_list) // batch_size)
    features = [] 
    try:
        model.to(device)
    except Exception as e: 
        print("[ERROR] ", str(e))
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
        batch_feature = model(batch_tensor) 
        if isinstance(batch_feature, torch.Tensor):
            batch_feature = batch_feature.detach().cpu().numpy() 
        features.append(batch_feature) 

    features = np.vstack(features)
    features = normalize(features)
    
    return features 


def save_features(output_dir, ls_features, list_name): 
    assert len(list_name) == len(ls_features) 

    os.makedirs(output_dir, exist_ok= True) 

    for count, name in tqdm(enumerate(list_name)): 
        basename  = os.path.basename(name) 
        basename = basename.split(".")[0]

        np.save(os.path.join(output_dir, basename), ls_features[count])
    
    
        