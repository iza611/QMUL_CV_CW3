from torchvision.datasets import Caltech101
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Lambda, ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import random_split
# from PIL.Image import convert
from PIL import Image

class Caltech101DataLoader:
    """
    Wrapper class around PyTorch's Caltech101 dataset to provide data loading functionality.
    """
    def __init__(self, batch_size, data_dir='data/', download=False, train_size=0.8):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = Compose([
            Resize((224, 224)), 
            Lambda(lambda x: x.convert("RGB")),
            ToTensor()
            ])

        self.dataset = Caltech101(self.data_dir, download=download, transform=self.transform)
        self.train_dataset, self.val_dataset = random_split(self.dataset, [
            int(train_size * len(self.dataset)),
            len(self.dataset) - int(train_size * len(self.dataset))
        ])

    def get_train_loader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def get_val_loader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        return val_loader