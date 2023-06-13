from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class MNISTDataLoader:
    """
    Wrapper class around PyTorch's MNIST dataset to provide data loading functionality.
    """
    def __init__(self, batch_size, data_dir='data/'):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = ToTensor()

    def get_train_loader(self, download=False):
        train_data = MNIST(self.data_dir, train=True, download=download, transform=self.transform)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def get_val_loader(self, download=False):
        val_data = MNIST(self.data_dir, train=False, download=download, transform=self.transform)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        return val_loader