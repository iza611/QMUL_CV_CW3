from sys import path
from os.path import dirname, abspath, exists
path.append(dirname(abspath(__file__)) + '/../')

import torch
# import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from os import makedirs
from time import time
from pickle import dump
from datetime import datetime

def get_time():
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

class Trainer:
    """
    Trainer class that handles the training loop.
    """
    def __init__(self, model, data_loader, optimizer, criterion, device, pretrained):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.pretrained = pretrained

        self.losses = []
        self.accuracies = []
        self.epoch_times = []
        self.epoch_it_per_sec = []

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            # Train for one epoch
            print(f"{get_time()} Epoch {epoch+1}/{num_epochs}")

            start_time = time()
            loss, accuracy = self._train_epoch(epoch)
            end_time = time()

            # Save performance
            self.epoch_times.append(end_time - start_time)
            self.epoch_it_per_sec.append(len(self.data_loader.get_train_loader())/(end_time - start_time))
            self.losses.append(loss)
            self.accuracies.append(accuracy)
            print(f"{get_time()} Accuracy: {accuracy}\nLoss: {loss}")
        
        # Display progress
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the loss on the first subplot
        ax1.plot(self.losses)
        ax1.set_title('Loss')
        ax1.set_ylim([0.0, 1.0])

        # Plot the accuracy on the second subplot
        ax2.plot(self.accuracies)
        ax2.set_title('Accuracy')
        ax2.set_ylim([0.0, 1.0])

        # Show the plots
        plt.show()

        # Save the trained model
        save_dir = 'results/models/'
        if not exists(save_dir):
            makedirs(save_dir)
        dataset_name = type(self.data_loader).__name__.replace("DataLoader", "")
        if(self.pretrained): name = f"trained_{type(self.model).__name__}(pretrained)_on_{dataset_name}_{time()}.pth"
        else: name = f"trained_{type(self.model).__name__}(NOT_pretrained)_on_{dataset_name}_{time()}.pth"
        # name = f"trained_{type(self.model).__name__}_on_{dataset_name}_{time()}.pth"
        save_path = save_dir + name
        torch.save(self.model.state_dict(), save_path)
        print(f"{get_time()} Model saved at {save_path}")

        # Save the progress
        save_dir = 'results/training/'
        if not exists(save_dir):
            makedirs(save_dir)
        name = name.replace(".pth", ".pkl")
        save_path = save_dir + name
        
        with open(save_path, 'wb') as f:
            data = {
                'losses': self.losses,
                'accuracies': list([float(t) for t in self.accuracies]),
                'epoch_times': self.epoch_times,
                'epoch_it_per_sec': self.epoch_it_per_sec
            }
            dump(data, f)
        
        print(f"{get_time()} Training logs saved at {save_path}")


    def _train_epoch(self, epoch):
        # Set model to train mode
        self.model.train()

        # Initialize metrics
        running_loss = 0.0
        num_correct = 0
        num_samples = 0

        # Iterate over data loader
        for i, (images, labels) in tqdm(enumerate(self.data_loader.get_train_loader()), total=len(self.data_loader.get_train_loader())):
            # Move data to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass and update weights
            loss.backward()
            self.optimizer.step()

            # Update metrics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            num_correct += torch.sum(preds == labels.data)
            num_samples += labels.size(0)

        # Calculate epoch-level metrics
        epoch_loss = running_loss / num_samples
        epoch_accuracy = num_correct.double() / num_samples

        return epoch_loss, epoch_accuracy
    

# from sys import path
# from os.path import dirname, abspath
# path.append(dirname(abspath(__file__)) + '/../')

# from dataset.mnist import MNISTDataLoader
# from models.resnet import ResNet
# from torch.optim import Adam

# if __name__ == '__main__':
#     # Set device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Load data
#     data_loader = MNISTDataLoader(batch_size=32)

#     # Load model
#     model = ResNet(num_classes=10, in_channels=1)

#     # Set optimizer and criterion
#     optimizer = Adam(model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()

#     # Initialize trainer
#     trainer = Trainer(model, data_loader, optimizer, criterion, device)

#     # Train model
#     trainer.train(num_epochs=2)