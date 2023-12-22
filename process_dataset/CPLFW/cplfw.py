import os 
import time 

import cv2
import numpy as np 
import mxnet as mx 
from tqdm import tqdm
from mxnet import ndarray as nd
import pickle 

# root_dir = "/home1/data/tanminh/faces_emore"
# path_imgrec = os.path.join(root_dir, 'train.rec')
# path_imgidx = os.path.join(root_dir, 'train.idx')

# imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

# s = imgrec.read_idx(0)
# header, _ = mx.recordio.unpack(s)
# if header.flag > 0:
#     header0 = (int(header.label[0]), int(header.label[1]))
#     imgidx = np.array(range(1, int(header.label[0])))
# else:
#     imgidx = np.array(list(imgrec.keys))

class CPLFW_Dataset:
    def __init__(self, path_bin) -> None:
        self.path_bin = path_bin 
        with open(self.path_bin, "rb") as file: 
            self.bins, self.issame_list= pickle.load(file, encoding="bytes")
        file.close()
    
    def convert(self, img_mx_decode):
        img_rgb = mx.image.imdecode(img_mx_decode) 
        img_rgb = img_rgb.asnumpy() 
        return img_rgb

    def process(self, dir_save_images:str, path_file_image_list:str, path_file_image_pair:str):
        os.makedirs(dir_save_images, exist_ok= True)
        os.makedirs(os.path.dirname(path_file_image_list), exist_ok=True)
        os.makedirs(os.path.dirname(path_file_image_pair), exist_ok= True) 

        file_image_list = open(path_file_image_list, "w") 
        file_image_pair = open(path_file_image_pair, "w") 


        for idx in tqdm(range(0, len(self.bins), 2)):
            name1 = str(idx).zfill(4) + ".jpg" 
            name2 = str(idx+1).zfill(4) + ".jpg"

            image1 = self.convert(self.bins[idx])
            image2 = self.convert(self.bins[idx+1]) 


            is_same = (self.issame_list[idx//2])
            if isinstance(is_same, bool):
                is_same = (is_same == 1)
            else: is_same = (is_same)


            # Save image 
            path_save_1 = os.path.join(dir_save_images, name1)
            path_save_1 = os.path.abspath(path_save_1)
            path_save_2 = os.path.join(dir_save_images, name2)
            path_save_2 = os.path.abspath(path_save_2)

            cv2.imwrite(path_save_1, cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))
            cv2.imwrite(path_save_2, cv2.cvtColor(image2, cv2.COLOR_RGB2BGR))

            file_image_list.write(
                str(path_save_1) + "\n"
            )
            file_image_list.write(
                str(path_save_2) + "\n" 
            )

            file_image_pair.write(
                str(name1) + " " + str(name2) + " " + str(int(is_same)) + "\n"
            )

        
        file_image_list.close()
        file_image_pair.close() 

            

# calfw_bin = "/home1/data/tanminh/faces_emore/cplfw.bin"

# with open(calfw_bin, 'rb') as file:
#     bins, issame_list = pickle.load(file, encoding="bytes")
# file.close() 

# print(len(bins))
# print(len(issame_list))
# print(issame_list[:10])
# idx = 16
# img = mx.image.imdecode(bins[idx])
# img = img.asnumpy() 
# cv2.imwrite(
#     "test.jpg", 
#     img
# )

# img = mx.image.imdecode(bins[idx+1])
# img = img.asnumpy() 
# cv2.imwrite(
#     "test_1.jpg", 
#     img
# )