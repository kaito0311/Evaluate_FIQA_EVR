import os

import torch

from models.backbones.iresnet_imintv5 import iresnet160
from models.recognition_model.FaceModel import FaceModel
from models.recognition_model.imintv5_plus.imint import ONNX_IMINT


class Head_Cls(torch.nn.Module):
    def __init__(self, in_features=1024, out_features=1) -> None:
        super().__init__()
        self.qs = torch.nn.Linear(in_features, out_features)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).cuda()
        return self.qs(x)


class Head_Multi_FC_Cls(torch.nn.Module):
    def __init__(self, in_features=1024, out_features=1) -> None:
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
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).cuda()
        x = self.middle(x)
        x = self.leaky(x)
        x = self.dropout(x)
        return self.qs(x)


class CRFIQA_ontop_multi_fc(FaceModel):
    def __init__(self, image_size=(112, 112), pretrained_backbone=None, pretrained_head=None, device="cpu"):
        super().__init__(image_size, pretrained_backbone, device)

        self.backbone = ONNX_IMINT()

        self.head = Head_Multi_FC_Cls() 
        self.head.to(device)
        self.head.eval() 
        if pretrained_head is not None:
            print("[INFO] Loading pretrained head ....")
            self.head.load_state_dict(torch.load(pretrained_head))

    def _get_model(self):
        print("[INFO] Loading pretrained backbone....")
        backbone = iresnet160(False).to(self.device)
        if self.pretrained is not None:
            weight = torch.load(self.pretrained)
            backbone.load_state_dict(weight)
        # model = torch.nn.DataParallel(backbone, )
        model = backbone
        model.eval()
        self.backbone = model
        return model

    @torch.no_grad()
    def _getFeatureBlob(self, input_blob):
        imgs = torch.Tensor(input_blob).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat = self.model(imgs)
        # feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()

    @torch.no_grad()
    def forward(self, x):

        return self.head(self.backbone(x))


class CRFIQA_ontop(FaceModel):
    def __init__(self, image_size=(112, 112), pretrained_backbone=None, pretrained_head=None, device="cpu"):
        super().__init__(image_size, pretrained_backbone, device)

        self.backbone = ONNX_IMINT()

        self.head = Head_Cls()
        self.head.to(device)
        if pretrained_head is not None:
            print("[INFO] Loading pretrained head ....")
            self.head.load_state_dict(torch.load(pretrained_head))

    def _get_model(self):
        print("[INFO] Loading pretrained backbone....")
        backbone = iresnet160(False).to(self.device)
        if self.pretrained is not None:
            weight = torch.load(self.pretrained)
            backbone.load_state_dict(weight)
        # model = torch.nn.DataParallel(backbone, )
        model = backbone
        model.eval()
        self.backbone = model
        return model

    @torch.no_grad()
    def _getFeatureBlob(self, input_blob):
        imgs = torch.Tensor(input_blob).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat = self.model(imgs)
        # feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()

    @torch.no_grad()
    def forward(self, x):

        return self.head(self.backbone(x))


