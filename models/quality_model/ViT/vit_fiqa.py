
import torch

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
