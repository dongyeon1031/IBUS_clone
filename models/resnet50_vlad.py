# @Time : 2022-07-13 11:20
# @Author : Wang Zhen
# @Email : frozenzhencola@163.com
# @File : resnet50_vlad.py
# @Project : GPR-R2-0701
import torch
from torch import nn
from .NetVLAD import NetVLAD
from torchvision.models import resnet50
class EmbedNet(nn.Module):
    def __init__(self):
        super(EmbedNet, self).__init__()
        # Discard layers at the end of base network
        encoder = resnet50(pretrained=True)
        base_model = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4
        )
        dim = list(base_model.parameters())[-1].shape[0]  # last channels (512)
        self.base_model = base_model
        # Define model for embedding
        self.net_vlad = NetVLAD(num_clusters=32, dim=dim, alpha=1.0)

    def forward(self, x):
        x = self.base_model(x)
        embedded_x = self.net_vlad(x)
        if not self.training:
            return embedded_x
        else:
            return embedded_x,embedded_x
def resnet50_vlad(num_classes=1000, pretrained=True, loss='softmax',feature_dim=500,dropout_p=0.5,use_gpu=True):
    model=EmbedNet().cuda()
    return model