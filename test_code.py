import numpy as np
from models.quality_model.Imintv5.imint import ONNX_FIQ_IMINT
from process_dataset.CALFW.calfw import CALFW_Dataset
import torch


from models.quality_model.Custom.crfiqa_ontop import CRFIQA_ontop


import os

import cv2
import matplotlib.pyplot as plt






result = np.load("/data/disk2/tanminh/Evaluate_FIQA_EVR/models/quality_model/DiffFIQA/DifFIQA/diffiqa_r/result.npy", allow_pickle= True).item()

file = open("DiffFIQA_scores.txt", "w") 

for key in result.keys():
    file.write(
        str(key) + " " + str(result[key]) + "\n"
    )

file.close()




exit() 


from models.quality_model.ViT.vit import VisionTransformer

class Head_Cls(torch.nn.Module):
    def __init__(self, in_features=512, out_features=1) -> None:
        super().__init__()
        self.middle = torch.nn.Linear(in_features, 128)
        self.leaky = torch.nn.LeakyReLU(negative_slope=0.1)
        self.dropout = torch.nn.Dropout(p=0.4)
        self.qs = torch.nn.Linear(128, out_features)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.middle(x)
        x = self.leaky(x)
        x = self.dropout(x)
        return self.qs(x)



class VIT_FIQA(torch.nn.Module):
    def __init__(self, pretrained_vit=None, pretrained_head=None, freeze_backbone=True) -> None:
        super().__init__()
        self.backbone_vit = VisionTransformer(
            input_size=224, patch_size=32, in_channels=3, dim=768, embedding_size=512,
            depth=12, num_heads=12, drop_path_rate=0.1, using_checkpoint=False)
        self.head = Head_Cls(512, 1)
        self.sigmoid = torch.sigmoid

        if pretrained_head is not None:
            print("[INFO] Loading pretrained : ", pretrained_head)
            self.head.load_state_dict(torch.load(pretrained_head))

        if pretrained_vit is not None:
            print("[INFO] Loading pretrained : ", pretrained_vit)
            self.backbone_vit.load_state_dict(torch.load(pretrained_vit))

        if freeze_backbone:
            self.backbone_vit.eval()
            for p in self.backbone_vit.parameters():
                p.requires_grad = False

    def forward(self, x):
        features = self.backbone_vit(x)
        output = self.head(features)
        return self.sigmoid(output)




model_fiqa = VIT_FIQA()
model_fiqa.load_state_dict(torch.load("/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/vit_1000header.pth"))




exit()

def plot_images_with_scores(images, scores, num_cols=10, name_save="yeah.jpg"):
    # Số lượng ảnh
    num_images = len(images)

    # Tính số dòng cần thiết
    num_rows = num_images // num_cols + (num_images % num_cols > 0)

    # Tạo figure và axes
    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))

    # Duyệt qua từng ảnh và score tương ứng
    for i, (image, score) in enumerate(zip(images, scores)):
        # Tính toán chỉ số dòng và cột tương ứng
        row = i // num_cols
        col = i % num_cols

        # Hiển thị ảnh
        axs[row, col].imshow(image)
        axs[row, col].axis('off')  # Tắt các trục

        # Hiển thị score dưới ảnh
        axs[row, col].set_title(f'Score: {score}')

    # Xóa các axes không sử dụng
    for i in range(num_images, num_rows * num_cols):
        fig.delaxes(axs.flatten()[i])

    # Hiển thị figure
    plt.show()

    plt.savefig(name_save)


path_list_score = "/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_XQLFW/CR_ONTOP_10K_MULTI_FC_KL_scores.txt"
folder_images = "/data/disk2/tanminh/Evaluate_FIQA_EVR/data/processed_XQLFW/images"

file = open(path_list_score, "r")
data = file.readlines()
file.close()
list_images = []
list_scores = []
for line in data:
    name_images, score = line.split(" ")
    score = float(score)
    image = cv2.imread(os.path.join(folder_images, name_images))
    list_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    list_scores.append(score)

arg = np.argsort(list_scores)
list_images = [list_images[i] for i in arg]
list_scores = [list_scores[i] for i in arg]

for start in range(0, len(list_images), 500):
    plot_images_with_scores(list_images[start:min(
        start + 500, len(list_images))], list_scores[start:min(start + 500, len(list_images))], name_save= f"{start}.jpg")


exit()


fnmr = np.load(
    "/data/disk2/tanminh/Evaluate_FIQA_EVR/output_dir_6/fnmr/CR-ONTOP-14K_fnmr.npy")
print(fnmr)


exit()
model = CRFIQA_ontop(pretrained_backbone="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/55000backbone.pth",
                     pretrained_head="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/55000header.pth", device="cuda")

model(torch.rand(4, 3, 112, 112).to("cuda"))


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


model = ONNX_FIQ_IMINT(
    path="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/model_fiq_imintv5.onnx"
)

output = model(torch.rand(4, 3, 112, 112))
print(output.shape)
print(output)
