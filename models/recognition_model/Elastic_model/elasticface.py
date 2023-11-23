import os

import torch

from models.backbones.iresnet import iresnet100
from models.recognition_model.FaceModel import FaceModel


class ElasticFaceModel(FaceModel):
    def __init__(self, image_size=(112, 112), pretrained=None, device="cpu"):
        super().__init__(image_size, pretrained, device)

        self.model = self._get_model()

    def _get_model(self):
        backbone = iresnet100().to(self.device)
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
        feat = self.model(imgs)
        # feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()

    def forward(self, x):
        return self.model(x)
