
import cv2
import torch
import numpy as np
from sklearn.preprocessing import normalize


class FaceModel(torch.nn.Module):
    def __init__(self, image_size=(112, 112), pretrained=None, device="cpu"):
        
        super(FaceModel, self).__init__()
        
        self.device = device
        self.image_size = image_size
        self.pretrained = pretrained
        self.model = None

    def _get_model(self, ctx, image_size, prefix, epoch, layer):
        pass

    def _getFeatureBlob(self, input_blob):
        pass

    def forward(self, x):
        pass 
