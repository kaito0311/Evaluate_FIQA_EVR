import sys

import argparse
import os
import pickle
import glob 

import cv2
import numpy as np 
from PIL import Image
from tqdm import tqdm 
import torch
from torchvision.transforms import transforms

from util import F, load_cosface, add_norm_to_model, load_imintv5
from bim_attack import basic_iterative_method as BIM
from symmetry import symmetry_estimation


PATH_25 = "eval_results/0_5/"
PATH_25_50 = "eval_results/5_15"
PATH_50_60 = "eval_results/15_20"
PATH_60_65 = "eval_results/20_25"
PATH_65 = "eval_results/25_30" 

os.makedirs(PATH_25, exist_ok= True)
os.makedirs(PATH_25_50, exist_ok= True)
os.makedirs(PATH_50_60, exist_ok= True)
os.makedirs(PATH_60_65, exist_ok= True)
os.makedirs(PATH_65, exist_ok= True)



def __calculate_score_single__(model: torch.nn.Module, image_transform: transforms.Compose, image_path: str, eps: float, l: int, k: int, p: int) -> float:
    """Method performs FaceQAN over individual images

    Args:
        model (torch.nn.Module): FR model used for adversarial attack.
        image_transform (transforms.Compose): Image transformation, without normalization.
        image_path (str): Path to image, for which the quality score will be predicted.
        eps (float): ε-parameter from the paper.
        l (int): l-parameter from the paper, controling the number of FGSM iterations to perform.
        k (int): k-parameter from the paper, controling the number of adversarial examples generated.
        p (int): p-parameter from the paper, used in the final quality score calculation as the exponent.

    Returns:
        float: The calculated quality score 
    """

    assert os.path.exists(
        image_path), f" Image path {image_path} does not exist! "

    # print(f" => Extracting quality for {image_path}")

    image = Image.open(image_path).convert("RGB")

    S_i = BIM(model, image_transform(image), eps=eps, iter=l)

    q_s = symmetry_estimation(model, image_transform, image)

    quality_score = F(S_i, q_s)

    os.makedirs("results_unit/", exist_ok= True)
    with open("results_unit/" + os.path.basename(image_path).split(".")[0] + ".txt", 'w') as file:
        file.write(str(quality_score))
    file.close() 

    return quality_score


def __calculate_score_batch__(image_paths: list, eps: float, l: int, k: int, p: int) -> dict:
    """Performs all the steps of the FaceQAN approach based on the input parameters

    Args:
        image_path (str): Path to the image for which we want to generate the quality score.
        eps (float): ε-parameter from the paper, controlling both the amount of adversarial noise as well as the uniform distribution.
        l (int): l-parameter from the paper, controlling the number of FGSM iterations to perform.
        k (int): k-parameter from the paper, controlling the number of adversarial examples generated.
        p (int): p-parameter from the paper, used in the final quality score calculation as the exponent.

    Returns:
        dict: Dictionary where key=image_path, value=generated quality score
    """

    """
        In case you wish to use a different FR model simply replace the load_cosface function with a custom function 
        that loads your desired FR model, additionally change the mean, st.deviation and transform used by your custom FR model
    """
    # model: torch.nn.Module = load_cosface().eval().cuda()
    model: torch.nn.Module = load_imintv5("/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/r160_imintv4_statedict.pth").eval().cuda()
    mean, std = [.5, .5, .5], [.5, .5, .5]
    image_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])

    model = add_norm_to_model(model, mean=mean, std=std)
    quality_scores = dict()
    for image_path in tqdm(image_paths):
        quality_scores[image_path] = __calculate_score_single__(
        model, image_transforms, image_path, eps, l, k, p)
    # quality_scores = dict(map(lambda image_path: (image_path, __calculate_score_single__(
    #     model, image_transforms, image_path, eps, l, k, p)), image_paths))

    return quality_scores

dict_qs = dict() 

def inference(list_images):
    class config:
        epsilon = 0.001
        bim_iterations = 5
        batch_size = 4
        exponent = 5

    dict_score = __calculate_score_batch__(
        list_images, config.epsilon, config.bim_iterations, config.batch_size, config.exponent)

    for key, score in dict_score.items():
        dict_score[key] = score 
        name = os.path.basename(key) 

        path_save = None 

        if score < 0.25: 
            path_save = PATH_25 
        elif score < 0.5: 
            path_save = PATH_25_50 
        elif score < 0.75: 
            path_save = PATH_50_60 
        else:
            path_save = PATH_65

        new_name = str(int(score * 100)) + "-" + name 
        
        cv2.imwrite(
            os.path.join(path_save, new_name), 
            img = cv2.imread(key) 
        )
    np.save("dict_score", dict_score)

list_images = glob.glob("/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_XQLFW/images/*.jpg")
# list_images = glob.glob("/home/data2/tanminh/FIQA/CR-FIQA/xqlfw_aligned_112/*/*.jpg")
print(len(list_images))
inference(list_images= list_images)