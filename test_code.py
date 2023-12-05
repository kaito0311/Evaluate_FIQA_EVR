from process_dataset.CALFW.calfw import CALFW_Dataset
import torch


from models.quality_model.Custom.crfiqa_ontop import CRFIQA_ontop 

















import numpy as np 

fnmr = np.load("/data/disk2/tanminh/Evaluate_FIQA_EVR/output_dir_6/fnmr/CR-ONTOP-14K_fnmr.npy")
print(fnmr)






exit() 
model = CRFIQA_ontop(pretrained_backbone="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/55000backbone.pth",
                     pretrained_head="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/55000header.pth", device="cuda")

model(torch.rand(4,3,112,112).to("cuda"))


# dataset = CPLFW_Dataset("/data/disk2/tanminh/Evaluate_FIQA_EVR/dataset_evaluate/cplfw.bin")

# dataset.process(
#     "calfw/save_images_temp", 
#     "calfw/file_name.txt",
#     "calfw/file_pair.txt"
# )

# value = True

# print(str(int(bool(value) == 1)))

# import cv2
# import numpy as np
# import onnxruntime 

# path = "/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/stacking_avg_r160+ada-unnorm-stacking-ada-1.6.onnx"

# session = onnxruntime.InferenceSession(path, providers = ['CUDAExecutionProvider'])

# IN_IMAGE_H = session.get_inputs()[0].shape[2]
# IN_IMAGE_W = session.get_inputs()[0].shape[3]

# # Input
# image_src = cv2.imread("/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_XQLFW/images/Abdoulaye_Wade_0004.jpg")
# resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
# img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
# img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
# img_in = np.expand_dims(img_in, axis=0)
# img_in /= 255.0
# img_in = img_in - 0.5 
# img_in /= 0.5
# # img_in = -1 + img_in * 2 
# # img_in = np.clip(img_in, -1, 1) 
# print(np.max(img_in))

# print("Shape of the network input: ", img_in.shape)



# input = session.get_inputs()

# input_name = session.get_inputs()[0].name 
# outputs = session.run(None, {input_name: img_in})

# print(type(outputs))
# print(outputs)
# np.save("file2", outputs[0])


# feature1 = np.load("/data/disk2/tanminh/Evaluate_FIQA_EVR/file1.npy")
# feature2 = np.load("/data/disk2/tanminh/Evaluate_FIQA_EVR/file2.npy")
# feature1 = feature1 / np.linalg.norm(feature1)
# feature2 = feature2 / np.linalg.norm(feature2) 
# dis = np.sqrt(np.sum((feature1 - feature2) ** 2))
# print("dis: ", dis)

import torch 

from models.quality_model.Imintv5.imint import ONNX_FIQ_IMINT
model = ONNX_FIQ_IMINT(
    path="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/model_fiq_imintv5.onnx"
)

output = model(torch.rand(4, 3, 112, 112))
print(output.shape)
print(output)