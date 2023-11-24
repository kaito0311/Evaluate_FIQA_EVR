import os 
import json 
import time 
import requests 
import base64 
import io 
import glob 


import cv2 
from PIL import Image 

list_images = list_images = glob.glob("/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_XQLFW/images/*.jpg")

def get_quality(image, api= "http://172.16.10.240:8899/dif-face/quality"):
    tik = time.time()
    byte_content = (cv2.imencode('.jpg', image)[1]).tobytes()
    request_header = {"file": byte_content}
    r = requests.post(api, files= request_header)
    # print(r.json())
    return r.json()["res"][0]["score"]

file_output = open("XQLFW_DiffFIQA.txt", "a") 

for image_path in list_images:
    quaility = get_quality(
        image= cv2.imread(image_path)
    )
    file_output.write(
        os.path.basename(
            image_path
        ) + " " + str(quaility) + "\n"
    )
file_output.close() 



