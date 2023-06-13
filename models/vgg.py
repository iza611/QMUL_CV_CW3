import torch.nn as nn
from torchvision.models import vgg11_bn

class VGG11(nn.Module):
    """
    Wrapper class around PyTorch's ResNet18 model to provide custom output layer.
    """
    def __init__(self, num_classes, in_channels=3, pretrained=False):
        super(VGG11, self).__init__()
        if(pretrained):
            self.model = vgg11_bn(weights='IMAGENET1K_V1') 
        else:
            self.model = vgg11_bn(weights=None)
        self.model.features[0] = nn.Conv2d(
            in_channels, 64, kernel_size=3, padding=1, bias=False 
        ) # stride=1 (default)
        
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)