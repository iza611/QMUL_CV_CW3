import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
# from dataset.mnist import MNISTDataLoader


import torch.nn as nn
import torch
from torchvision.models import resnet18
# import matplotlib.pyplot as plt

class ResNet(nn.Module):
    """
    Wrapper class around PyTorch's ResNet18 model to provide custom output layer.
    """
    def __init__(self, num_classes, in_channels=3, pretrained=False):
        super(ResNet, self).__init__()
        if(pretrained):
            self.model = resnet18(weights='IMAGENET1K_V1') # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html#torchvision.models.ResNet18_Weights
        else:
            self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False # replace the nn.Conv2d instance in the ResNet model's conv1 layer to make it flexible for different datasets. The rest of the model is unchanged
        )
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)