import os

import torch

from models.backbones.irestnet_quality import iresnet50, iresnet100
from models.recognition_model.FaceModel import FaceModel


class CR_FIQA_Model(FaceModel):
    def __init__(self, image_size=(112, 112), pretrained=None, device="cpu", arch="r50"):
        super().__init__(image_size, pretrained, device)
        self.arch = arch
        self.model = self._get_model()

    def _get_model(self):
        if self.arch == "r50":
            backbone = iresnet50(num_features=512, qs=1, use_se=False).to(self.device)
        elif self.arch == "r100":
            backbone= iresnet100(num_features=512, qs=1, use_se=False).to(self.device)
        if self.pretrained is not None:
            weight = torch.load(self.pretrained)
            backbone.load_state_dict(weight)
        # model = torch.nn.DataParallel(backbone, )
        model = backbone
        model.eval()
        self.model = model 
        return model

    @torch.no_grad()
    def _getFeatureBlob(self, input_blob):
        imgs = torch.Tensor(input_blob).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat, qs = self.model(imgs)
        # feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()

    @torch.no_grad()
    def forward(self, x):
        return self.model(x)[1]
