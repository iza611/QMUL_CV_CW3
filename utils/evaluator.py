from sys import path
from os.path import dirname, abspath, exists
path.append(dirname(abspath(__file__)) + '/../')

from dataset.mnist import MNISTDataLoader
from dataset.caltech101 import Caltech101DataLoader
from models.resnet import ResNet
from models.vgg import VGG11
from torch import load
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from random import randint, random
from sklearn.metrics import confusion_matrix
import seaborn as sns

def get_training_data(path):
    with open(path, 'rb') as f:
        training_data = pickle.load(f)

    losses = training_data['losses']
    accuracies = training_data['accuracies']
    epoch_times = training_data['epoch_times']
    epoch_it_per_sec = training_data['epoch_it_per_sec']

    return losses, accuracies, epoch_times, epoch_it_per_sec

class Evaluation():
    def __init__(self, batch_size=16):
        self.batch_size = batch_size

        # load mnist and tiny imagement data loader (both tran and val loaders)
        self.MNIST_data_loader = MNISTDataLoader(batch_size=self.batch_size)
        self.Caltech101_data_loader = Caltech101DataLoader(batch_size=self.batch_size)

        # load resnet and vgg nets (both pretrained and not) & training data
        
        # MNIST
        ResNet_pretrained_for_MNIST_model = ResNet(num_classes=10, in_channels=1)
        
        ResNet_pretrained_for_MNIST_model.load_state_dict(load('results/models/trained_ResNet(pretrained)_on_MNIST_1683453407.0334976.pth'))
        ResNet_NOT_pretrained_for_MNIST_model = ResNet(num_classes=10, in_channels=1)
        ResNet_NOT_pretrained_for_MNIST_model.load_state_dict(load('results/models/trained_ResNet(NOT_pretrained)_on_MNIST_1683453860.8321037.pth'))
        
        VGG11_pretrained_for_MNIST_model = VGG11(num_classes=10, in_channels=1)
        VGG11_pretrained_for_MNIST_model.load_state_dict(load('results/models/trained_VGG11(pretrained)_on_MNIST_1683461182.578489.pth'))
        VGG11_NOT_pretrained_for_MNIST_model = VGG11(num_classes=10, in_channels=1)
        VGG11_NOT_pretrained_for_MNIST_model.load_state_dict(load('results/models/trained_VGG11(NOT_pretrained)_on_MNIST_1683470357.5373356.pth'))

        self.MNIST_models = {'ResNet_pretrained': ResNet_pretrained_for_MNIST_model, 
                             'ResNet_NOT_pretrained': ResNet_NOT_pretrained_for_MNIST_model,
                             'VGG11_pretrained': VGG11_pretrained_for_MNIST_model,
                             'VGG11_NOT_pretrained': VGG11_NOT_pretrained_for_MNIST_model}

        ResNet_pretrained_for_MNIST_losses, ResNet_pretrained_for_MNIST_accuracies, ResNet_pretrained_for_MNIST_epoch_times, ResNet_pretrained_for_MNIST_epoch_it_per_sec = get_training_data('results/training/trained_ResNet(pretrained)_on_MNIST_1683453407.0334976.pkl')
        ResNet_NOT_pretrained_for_MNIST_losses, ResNet_NOT_pretrained_for_MNIST_accuracies, ResNet_NOT_pretrained_for_MNIST_epoch_times, ResNet_NOT_pretrained_for_MNIST_epoch_it_per_sec = get_training_data('results/training/trained_ResNet(NOT_pretrained)_on_MNIST_1683453860.8321037.pkl')
    
        VGG11_pretrained_for_MNIST_losses, VGG11_pretrained_for_MNIST_accuracies, VGG11_pretrained_for_MNIST_epoch_times, VGG11_pretrained_for_MNIST_epoch_it_per_sec = get_training_data('results/training/trained_VGG11(pretrained)_on_MNIST_1683461182.578489.pkl')
        VGG11_NOT_pretrained_for_MNIST_losses, VGG11_NOT_pretrained_for_MNIST_accuracies, VGG11_NOT_pretrained_for_MNIST_epoch_times, VGG11_NOT_pretrained_for_MNIST_epoch_it_per_sec = get_training_data('results/training/trained_VGG11(NOT_pretrained)_on_MNIST_1683470357.5373356.pkl')

        self.MNIST_metrics_pretrained = {'loss': [ResNet_pretrained_for_MNIST_losses, VGG11_pretrained_for_MNIST_losses], 
                                        'accuracy': [ResNet_pretrained_for_MNIST_accuracies, VGG11_pretrained_for_MNIST_accuracies],
                                        'epoch_time': [ResNet_pretrained_for_MNIST_epoch_times, VGG11_pretrained_for_MNIST_epoch_times],
                                        'it_per_sec': [ResNet_pretrained_for_MNIST_epoch_it_per_sec, VGG11_pretrained_for_MNIST_epoch_it_per_sec]}
        
        self.MNIST_metrics_NOT_pretrained = {'loss': [ResNet_NOT_pretrained_for_MNIST_losses, VGG11_NOT_pretrained_for_MNIST_losses], 
                                        'accuracy': [ResNet_NOT_pretrained_for_MNIST_accuracies, VGG11_NOT_pretrained_for_MNIST_accuracies],
                                        'epoch_time': [ResNet_NOT_pretrained_for_MNIST_epoch_times, VGG11_NOT_pretrained_for_MNIST_epoch_times],
                                        'it_per_sec': [ResNet_NOT_pretrained_for_MNIST_epoch_it_per_sec, VGG11_NOT_pretrained_for_MNIST_epoch_it_per_sec]}

        # Caltech101
        ResNet_pretrained_for_Caltech101_model = ResNet(num_classes=101, in_channels=3)
        ResNet_pretrained_for_Caltech101_model.load_state_dict(load('results/models/trained_ResNet(pretrained)_on_Caltech101_1683479581.2552748.pth'))
        ResNet_NOT_pretrained_for_Caltech101_model = ResNet(num_classes=101, in_channels=3)
        ResNet_NOT_pretrained_for_Caltech101_model.load_state_dict(load('results/models/trained_ResNet(NOT_pretrained)_on_Caltech101_1683483710.2781956.pth'))
        
        VGG11_pretrained_for_Caltech101_model = VGG11(num_classes=101, in_channels=3)
        VGG11_pretrained_for_Caltech101_model.load_state_dict(load('results/models/trained_VGG11(pretrained)_on_Caltech101_1683488118.611486.pth'))
        VGG11_NOT_pretrained_for_Caltech101_model = VGG11(num_classes=101, in_channels=3)
        VGG11_NOT_pretrained_for_Caltech101_model.load_state_dict(load('results/models/trained_VGG11(NOT_pretrained)_on_Caltech101_1683548034.838974.pth'))

        self.Caltech101_models = {'ResNet_pretrained': ResNet_pretrained_for_Caltech101_model, 
                                  'ResNet_NOT_pretrained': ResNet_NOT_pretrained_for_Caltech101_model, 
                                  'VGG11_pretrained': VGG11_pretrained_for_Caltech101_model,
                                  'VGG11_NOT_pretrained': VGG11_NOT_pretrained_for_Caltech101_model}

        ResNet_pretrained_for_Caltech101_losses, ResNet_pretrained_for_Caltech101_accuracies, ResNet_pretrained_for_Caltech101_epoch_times, ResNet_pretrained_for_Caltech101_epoch_it_per_sec = get_training_data('results/training/trained_ResNet(pretrained)_on_Caltech101_1683479581.2552748.pkl')
        ResNet_NOT_pretrained_for_Caltech101_losses, ResNet_NOT_pretrained_for_Caltech101_accuracies, ResNet_NOT_pretrained_for_Caltech101_epoch_times, ResNet_NOT_pretrained_for_Caltech101_epoch_it_per_sec = get_training_data('results/training/trained_ResNet(NOT_pretrained)_on_Caltech101_1683483710.2781956.pkl')
        
        VGG11_pretrained_for_Caltech101_losses, VGG11_pretrained_for_Caltech101_accuracies, VGG11_pretrained_for_Caltech101_epoch_times, VGG11_pretrained_for_Caltech101_epoch_it_per_sec = get_training_data('results/training/trained_VGG11(pretrained)_on_Caltech101_1683488118.611486.pkl')
        VGG11_NOT_pretrained_for_Caltech101_losses, VGG11_NOT_pretrained_for_Caltech101_accuracies, VGG11_NOT_pretrained_for_Caltech101_epoch_times, VGG11_NOT_pretrained_for_Caltech101_epoch_it_per_sec = get_training_data('results/training/trained_VGG11(NOT_pretrained)_on_Caltech101_1683548034.838974.pkl')

        self.Caltech101_metrics_pretrained = {'loss': [ResNet_pretrained_for_Caltech101_losses, VGG11_pretrained_for_Caltech101_losses], 
                                            'accuracy': [ResNet_pretrained_for_Caltech101_accuracies, VGG11_pretrained_for_Caltech101_accuracies],
                                            'epoch_time': [ResNet_pretrained_for_Caltech101_epoch_times, VGG11_pretrained_for_Caltech101_epoch_times],
                                            'it_per_sec': [ResNet_pretrained_for_Caltech101_epoch_it_per_sec, VGG11_pretrained_for_Caltech101_epoch_it_per_sec]}
        
        self.Caltech101_metrics_NOT_pretrained = {'loss': [ResNet_NOT_pretrained_for_Caltech101_losses, VGG11_NOT_pretrained_for_Caltech101_losses], 
                                                'accuracy': [ResNet_NOT_pretrained_for_Caltech101_accuracies, VGG11_NOT_pretrained_for_Caltech101_accuracies],
                                                'epoch_time': [ResNet_NOT_pretrained_for_Caltech101_epoch_times, VGG11_NOT_pretrained_for_Caltech101_epoch_times],
                                                'it_per_sec': [ResNet_NOT_pretrained_for_Caltech101_epoch_it_per_sec, VGG11_NOT_pretrained_for_Caltech101_epoch_it_per_sec]}

    def visualise_dataset(self, dataset_name):
        if dataset_name == 'mnist':
            data_loader = self.MNIST_data_loader
        elif dataset_name == 'caltech':
            data_loader = self.Caltech101_data_loader
        else:
            raise 

       # Get a random batch of images and labels
        images, labels = next(iter(data_loader.get_val_loader()))

        # Reshape the images from [batch_size, num_channels, height, width] to [batch_size, height, width, num_channels]
        images = images.permute(0, 2, 3, 1)

        # Create a 4x4 grid of images
        fig, ax = plt.subplots(4, 4, figsize=(8, 8))
        fig.suptitle(f'{dataset_name.upper()} dataset')
        for i in range(4):
            for j in range(4):
                # Display the image and its corresponding label
                image_idx = i * 4 + j
                image = images[image_idx]
                label = labels[image_idx].item()
                if(dataset_name=='mnist'): 
                    ax[i, j].imshow(image, cmap='gray')
                    ax[i, j].set_title(f'Label: {label}')
                elif(dataset_name=='caltech'): 
                    ax[i, j].imshow(image)
                    ax[i, j].set_title(f'{data_loader.dataset.categories[label]}')
                ax[i, j].axis('off')
        plt.show()

    def plot_training_progress(self, metric_name, dataset_name):
        if metric_name != 'loss' and metric_name != 'accuracy': raise

        if dataset_name == 'mnist':
            ResNet_data_pretrained = self.MNIST_metrics_pretrained[metric_name][0]
            VGG11_data_pretrained = self.MNIST_metrics_pretrained[metric_name][1]
            ResNet_data_NOT_pretrained = self.MNIST_metrics_NOT_pretrained[metric_name][0]
            VGG11_data_NOT_pretrained = self.MNIST_metrics_NOT_pretrained[metric_name][1]
        elif dataset_name == 'caltech':
            ResNet_data_pretrained = self.Caltech101_metrics_pretrained[metric_name][0]
            VGG11_data_pretrained = self.Caltech101_metrics_pretrained[metric_name][1]
            ResNet_data_NOT_pretrained = self.Caltech101_metrics_NOT_pretrained[metric_name][0]
            VGG11_data_NOT_pretrained = self.Caltech101_metrics_NOT_pretrained[metric_name][1]
        else:
            raise  

        # Plot data
        plt.plot(ResNet_data_pretrained, label='ResNet (fine-tuned)')
        plt.plot(VGG11_data_pretrained, label='VGG11 (fine-tuned)')
        plt.plot(ResNet_data_NOT_pretrained, label='ResNet (baseline)')
        plt.plot(VGG11_data_NOT_pretrained, label='VGG11 (baseline)')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.capitalize())
        plt.title(f"{metric_name.capitalize()} vs Epoch for {dataset_name.upper()}")
        plt.legend()
        plt.show()

    def compare_computational_performace(self, metric_name, dataset_name):
        # metric_name can be 'epoch_time' or 'it_per_sec'
        if dataset_name == 'mnist':
            ResNet_data_pretrained = self.MNIST_metrics_pretrained[metric_name][0]
            VGG11_data_pretrained = self.MNIST_metrics_pretrained[metric_name][1]
            ResNet_data_NOT_pretrained = self.MNIST_metrics_NOT_pretrained[metric_name][0]
            VGG11_data_NOT_pretrained = self.MNIST_metrics_NOT_pretrained[metric_name][1]
        elif dataset_name == 'caltech':
            ResNet_data_pretrained = self.Caltech101_metrics_pretrained[metric_name][0]
            VGG11_data_pretrained = self.Caltech101_metrics_pretrained[metric_name][1]
            ResNet_data_NOT_pretrained = self.Caltech101_metrics_NOT_pretrained[metric_name][0]
            VGG11_data_NOT_pretrained = self.Caltech101_metrics_NOT_pretrained[metric_name][1]
        else:
            raise  

        # Create box plot
        data = [ResNet_data_pretrained, ResNet_data_NOT_pretrained, VGG11_data_pretrained, VGG11_data_NOT_pretrained]
        labels = ['ResNet (fine-tuned)', 'ResNet (baseline)', 'VGG11 (fine-tuned)', 'VGG11 (baseline)']
        plt.figure(figsize=(12, 6))
        plt.boxplot(data, labels=labels)
        plt.ylabel(metric_name.capitalize())
        plt.title(f"{metric_name.capitalize()} Comparison for {dataset_name.capitalize()}")
        plt.show()
          
        print(metric_name.capitalize())
        print(f"ResNet (fine-tuned) = {ResNet_data_pretrained}\nResNet (baseline) = {ResNet_data_NOT_pretrained}\nVGG11 (fine-tuned) = {VGG11_data_pretrained}\nVGG11 (baseline) = {VGG11_data_NOT_pretrained}")

    def plot_train_vs_test(self, dataset_name):
        # only 'accuracy'
        if dataset_name == 'mnist':
            ResNet_pretrained_model = self.MNIST_models['ResNet_pretrained']
            VGG11_pretrained_model = self.MNIST_models['VGG11_pretrained']
            ResNet_NOT_pretrained_model = self.MNIST_models['ResNet_NOT_pretrained']
            VGG11_NOT_pretrained_model = self.MNIST_models['VGG11_NOT_pretrained']

            ResNet_data_pretrained = self.MNIST_metrics_pretrained['accuracy'][0]
            VGG11_data_pretrained = self.MNIST_metrics_pretrained['accuracy'][1]
            ResNet_data_NOT_pretrained = self.MNIST_metrics_NOT_pretrained['accuracy'][0]
            VGG11_data_NOT_pretrained = self.MNIST_metrics_NOT_pretrained['accuracy'][1]

            data_loader = self.MNIST_data_loader
        elif dataset_name == 'caltech':
            ResNet_pretrained_model = self.Caltech101_models['ResNet_pretrained']
            VGG11_pretrained_model = self.Caltech101_models['VGG11_pretrained']
            ResNet_NOT_pretrained_model = self.Caltech101_models['ResNet_NOT_pretrained']
            VGG11_NOT_pretrained_model = self.Caltech101_models['VGG11_NOT_pretrained']

            ResNet_data_pretrained = self.Caltech101_metrics_pretrained['accuracy'][0]
            VGG11_data_pretrained = self.Caltech101_metrics_pretrained['accuracy'][1]
            ResNet_data_NOT_pretrained = self.Caltech101_metrics_NOT_pretrained['accuracy'][0]
            VGG11_data_NOT_pretrained = self.Caltech101_metrics_NOT_pretrained['accuracy'][1]
            
            data_loader = self.Caltech101_data_loader
        else:
            raise

        # Calculate testing accuracy/loss with models and data loaders
        ResNet_pretrained_test_metric = self._calculate_metric(ResNet_pretrained_model, data_loader.get_val_loader())
        VGG11_pretrained_test_metric = self._calculate_metric(VGG11_pretrained_model, data_loader.get_val_loader())
        ResNet_NOT_pretrained_test_metric = self._calculate_metric(ResNet_NOT_pretrained_model, data_loader.get_val_loader())
        VGG11_NOT_pretrained_test_metric = self._calculate_metric(VGG11_NOT_pretrained_model, data_loader.get_val_loader())
        
        # mock
        # ResNet_pretrained_test_metric = 
        # VGG11_pretrained_test_metric = 0.3
        # ResNet_NOT_pretrained_test_metric = 0.4
        # VGG11_NOT_pretrained_test_metric = 0.2
        print(f"ResNet (fine-tuned) test accuracy: {ResNet_pretrained_test_metric}\nResNet (baseline) test accuracy: {ResNet_NOT_pretrained_test_metric}\nVGG1 (fine-tuned) test accuracy: {VGG11_pretrained_test_metric}\nVGG1 (baseline) test accuracy: {VGG11_NOT_pretrained_test_metric}")

        # Plot train and test accuracies/losses for both models
        fig, ax = plt.subplots()
        bar_width = 0.35
        opacity = 0.8

        # ResNet bars
        index = [0, 1, 2, 3]
        train_data = [ResNet_data_pretrained[-1], ResNet_data_NOT_pretrained[-1], VGG11_data_pretrained[-1], VGG11_data_NOT_pretrained[-1]]
        test_data = [ResNet_pretrained_test_metric, ResNet_NOT_pretrained_test_metric, VGG11_pretrained_test_metric, VGG11_NOT_pretrained_test_metric]
        ax.bar(index, train_data, bar_width, alpha=opacity, label='Train')
        ax.bar([i + bar_width for i in index], test_data, bar_width, alpha=opacity, label='Test')

        ax.set_xlabel('Model')
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Train vs Test Accuracy for {dataset_name.capitalize()}")
        ax.set_xticks([i + bar_width/2 for i in index])
        ax.set_xticklabels(('ResNet (fine-tuned)', 'ResNet (baseline)', 'VGG11 (fine-tuned)', 'VGG11 (baseline)'))
        ax.legend()

        plt.tight_layout()
        plt.show()

    def _calculate_metric(self, model, data_loader):
        total_metric = 0
        total_batches = 0
        criterion = nn.CrossEntropyLoss()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(data_loader):
                # images = images.to(device)
                # labels = labels.to(device)
                images = images
                labels = labels
                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)
                total_metric += (predicted == labels).sum().item()

                total_batches += 1

        return total_metric / len(data_loader.dataset)

    def get_quanti_metrics(self, dataset_name):
        if dataset_name == 'mnist':
            ResNet_pretrained_model = self.MNIST_models['ResNet_pretrained']
            VGG11_pretrained_model = self.MNIST_models['VGG11_pretrained']
            ResNet_NOT_pretrained_model = self.MNIST_models['ResNet_NOT_pretrained']
            VGG11_NOT_pretrained_model = self.MNIST_models['VGG11_NOT_pretrained']

            data_loader = self.MNIST_data_loader
        elif dataset_name == 'caltech':
            ResNet_pretrained_model = self.Caltech101_models['ResNet_pretrained']
            VGG11_pretrained_model = self.Caltech101_models['VGG11_pretrained']
            ResNet_NOT_pretrained_model = self.Caltech101_models['ResNet_NOT_pretrained']
            VGG11_NOT_pretrained_model = self.Caltech101_models['VGG11_NOT_pretrained']

            data_loader = self.Caltech101_data_loader
        else:
            raise

        # Get the predictions and labels for fine-tuned ResNet model
        print("Make predictions on the testing dataset for the fine-tuned ResNet model.")
        ResNet_pretrained_predictions, ResNet_pretrained_labels = [], []
        for images, labels in tqdm(data_loader.get_val_loader()):
            predictions = ResNet_pretrained_model(images)
            ResNet_pretrained_predictions.extend(torch.argmax(predictions, dim=1))
            ResNet_pretrained_labels.extend(labels)

        # Get the predictions and labels for baseline ResNet model
        print("Make predictions on the testing dataset for the baseline ResNet model.")
        ResNet_NOT_pretrained_predictions, ResNet_NOT_pretrained_labels = [], []
        for images, labels in tqdm(data_loader.get_val_loader()):
            predictions = ResNet_NOT_pretrained_model(images)
            ResNet_NOT_pretrained_predictions.extend(torch.argmax(predictions, dim=1))
            ResNet_NOT_pretrained_labels.extend(labels)

        # Get the predictions and labels for fine-tuned VGG11 model
        print("Make predictions on the testing dataset for the fine-tuned VGG11 model.")
        VGG11_pretrained_predictions, VGG11_pretrained_labels = [], []
        for images, labels in tqdm(data_loader.get_val_loader()):
            predictions = VGG11_pretrained_model(images)
            VGG11_pretrained_predictions.extend(torch.argmax(predictions, dim=1))
            VGG11_pretrained_labels.extend(labels)

        # Get the predictions and labels for baseline VGG11 model
        print("Make predictions on the testing dataset for the baseline VGG11 model.")
        VGG11_NOT_pretrained_predictions, VGG11_NOT_pretrained_labels = [], []
        for images, labels in tqdm(data_loader.get_val_loader()):
            predictions = VGG11_NOT_pretrained_model(images)
            VGG11_NOT_pretrained_predictions.extend(torch.argmax(predictions, dim=1))
            VGG11_NOT_pretrained_labels.extend(labels)

        # mock
        # ResNet_NOT_pretrained_predictions = [torch.tensor(randint(0, 9)) for _ in range(10000)]
        # ResNet_NOT_pretrained_labels = [torch.tensor(0) if random() < 0.6722 else t for t in ResNet_NOT_pretrained_predictions]
        # VGG11_pretrained_predictions = [torch.tensor(randint(0, 9)) for _ in range(10000)]
        # VGG11_pretrained_labels = [torch.tensor(0) if random() < 0.3722 else t for t in VGG11_pretrained_predictions]
        # VGG11_NOT_pretrained_predictions = [torch.tensor(randint(0, 9)) for _ in range(10000)]
        # VGG11_NOT_pretrained_labels = [torch.tensor(0) if random() < 0.4722 else t for t in VGG11_NOT_pretrained_predictions]

        # Calculate the precision, recall, F1 score, and support for ResNet model
        ResNet_pretrained_precision, ResNet_pretrained_recall, ResNet_pretrained_f1, ResNet_pretrained_support = precision_recall_fscore_support(ResNet_pretrained_labels, ResNet_pretrained_predictions, average='macro')
        ResNet_NOT_pretrained_precision, ResNet_NOT_pretrained_recall, ResNet_NOT_pretrained_f1, ResNet_NOT_pretrained_support = precision_recall_fscore_support(ResNet_NOT_pretrained_labels, ResNet_NOT_pretrained_predictions, average='macro')

        # Calculate the precision, recall, F1 score, and support for VGG11 model
        VGG11_pretrained_precision, VGG11_pretrained_recall, VGG11_pretrained_f1, VGG11_pretrained_support = precision_recall_fscore_support(VGG11_pretrained_labels, VGG11_pretrained_predictions, average='macro')
        VGG11_NOT_pretrained_precision, VGG11_NOT_pretrained_recall, VGG11_NOT_pretrained_f1, VGG11_NOT_pretrained_support = precision_recall_fscore_support(VGG11_NOT_pretrained_labels, VGG11_NOT_pretrained_predictions, average='macro')

        return {
            'ResNet (fine-tuned)': {
                'precision': ResNet_pretrained_precision,
                'recall': ResNet_pretrained_recall,
                'f1': ResNet_pretrained_f1,
                'support': ResNet_pretrained_support
            },
            'ResNet (baseline)': {
                'precision': ResNet_NOT_pretrained_precision,
                'recall': ResNet_NOT_pretrained_recall,
                'f1': ResNet_NOT_pretrained_f1,
                'support': ResNet_NOT_pretrained_support
            },
            'VGG11 (fine-tuned)': {
                'precision': VGG11_pretrained_precision,
                'recall': VGG11_pretrained_recall,
                'f1': VGG11_pretrained_f1,
                'support': VGG11_pretrained_support
            },
            'VGG11 (baseline)': {
                'precision': VGG11_NOT_pretrained_precision,
                'recall': VGG11_NOT_pretrained_recall,
                'f1': VGG11_NOT_pretrained_f1,
                'support': VGG11_NOT_pretrained_support
            }
        }

    def plot_successes(self, dataset_name, model_name):
        if dataset_name == 'mnist':
            if(model_name=='resnet'): model = self.MNIST_models['ResNet_pretrained']
            elif(model_name=='vgg'): model = self.MNIST_models['VGG11_pretrained']
            else: raise
            data_loader = self.MNIST_data_loader
        elif dataset_name == 'caltech':
            if(model_name=='resnet'): model = self.Caltech101_models['ResNet_pretrained']
            elif(model_name=='vgg'): model = self.Caltech101_models['VGG11_pretrained']
            else: raise
            data_loader = self.Caltech101_data_loader
        else:
            raise 

        successes = []
        successes_preds = []
        for images, labels in tqdm(data_loader.get_val_loader()):
            # Get predictions from the model
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct_indices = (predicted == labels).nonzero(as_tuple=True)[0]
            for index in correct_indices:
                image = images[index].permute(1, 2, 0).numpy()
                successes.append(image)
                successes_preds.append(predicted[index])

            # Stop after getting 4 examples for each model
            if len(successes) >= 16:
                break

        # Plot ResNet and VGG11 successes together
        fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(4):
            for j in range(4):
                if(dataset_name=='mnist'): 
                    axs[i, j].imshow(successes[i*4+j], cmap='gray')
                    axs[i, j].set_title(f'Predicted:{successes_preds[i*4+j]}')
                elif(dataset_name=='caltech'): 
                    axs[i, j].imshow(successes[i*4+j])
                    axs[i, j].set_title(f'{data_loader.dataset.categories[successes_preds[i*4+j]]}')
                axs[i, j].axis('off')
        if(model_name=='resnet'): fig.suptitle('ResNet Successes')
        elif(model_name=='vgg'): fig.suptitle('VGG11 Successes')
        plt.show()

    def plot_failures(self, dataset_name, model_name):
        if dataset_name == 'mnist':
            if(model_name=='resnet'): model = self.MNIST_models['ResNet_pretrained']
            elif(model_name=='vgg'): model = self.MNIST_models['VGG11_pretrained']
            else: raise
            data_loader = self.MNIST_data_loader
        elif dataset_name == 'caltech':
            if(model_name=='resnet'): model = self.Caltech101_models['ResNet_pretrained']
            elif(model_name=='vgg'): model = self.Caltech101_models['VGG11_pretrained']
            else: raise
            data_loader = self.Caltech101_data_loader
        else:
            raise 

        failures = []
        failures_preds = []
        for images, labels in tqdm(data_loader.get_val_loader()):
            # Get predictions from the model
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            incorrect_indices = (predicted != labels).nonzero(as_tuple=True)[0]
            for index in incorrect_indices:
                image = images[index].permute(1, 2, 0).numpy()
                failures.append(image)
                failures_preds.append(predicted[index])

            # Stop after getting 4 examples for each model
            if len(failures) >= 16:
                break

        # Plot ResNet and VGG11 failures together
        fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(4):
            for j in range(4):
                if(dataset_name=='mnist'): 
                    axs[i, j].imshow(failures[i*4+j], cmap='gray')
                    axs[i, j].set_title(f'Predicted:{failures_preds[i*4+j]}')
                elif(dataset_name=='caltech'): 
                    axs[i, j].imshow(failures[i*4+j])
                    axs[i, j].set_title(f'{data_loader.dataset.categories[failures_preds[i*4+j]]}')
                axs[i, j].axis('off')
        if(model_name=='resnet'): fig.suptitle('ResNet Failures')
        elif(model_name=='vgg'): fig.suptitle('VGG11 Failures')
        plt.show()

    def plot_confusion_matrix(self, dataset_name):
        if dataset_name == 'mnist':
            ResNet_pretrained_model = self.MNIST_models['ResNet_pretrained']
            VGG11_pretrained_model = self.MNIST_models['VGG11_pretrained']
            ResNet_NOT_pretrained_model = self.MNIST_models['ResNet_NOT_pretrained']
            VGG11_NOT_pretrained_model = self.MNIST_models['VGG11_NOT_pretrained']

            data_loader = self.MNIST_data_loader
        elif dataset_name == 'caltech':
            ResNet_pretrained_model = self.Caltech101_models['ResNet_pretrained']
            VGG11_pretrained_model = self.Caltech101_models['VGG11_pretrained']
            ResNet_NOT_pretrained_model = self.Caltech101_models['ResNet_NOT_pretrained']
            VGG11_NOT_pretrained_model = self.Caltech101_models['VGG11_NOT_pretrained']

            data_loader = self.Caltech101_data_loader
        else:
            raise

        # Get the predictions and labels for fine-tuned ResNet model
        print("Make predictions on the testing dataset for the fine-tuned ResNet model.")
        ResNet_pretrained_predictions, ResNet_pretrained_labels = [], []
        for images, labels in tqdm(data_loader.get_val_loader()):
            predictions = ResNet_pretrained_model(images)
            ResNet_pretrained_predictions.extend(torch.argmax(predictions, dim=1))
            ResNet_pretrained_labels.extend(labels)

        # Get the predictions and labels for baseline ResNet model
        print("Make predictions on the testing dataset for the baseline ResNet model.")
        ResNet_NOT_pretrained_predictions, ResNet_NOT_pretrained_labels = [], []
        for images, labels in tqdm(data_loader.get_val_loader()):
            predictions = ResNet_NOT_pretrained_model(images)
            ResNet_NOT_pretrained_predictions.extend(torch.argmax(predictions, dim=1))
            ResNet_NOT_pretrained_labels.extend(labels)

        # Get the predictions and labels for fine-tuned VGG11 model
        print("Make predictions on the testing dataset for the fine-tuned VGG11 model.")
        VGG11_pretrained_predictions, VGG11_pretrained_labels = [], []
        for images, labels in tqdm(data_loader.get_val_loader()):
            predictions = VGG11_pretrained_model(images)
            VGG11_pretrained_predictions.extend(torch.argmax(predictions, dim=1))
            VGG11_pretrained_labels.extend(labels)

        # Get the predictions and labels for baseline VGG11 model
        print("Make predictions on the testing dataset for the baseline VGG11 model.")
        VGG11_NOT_pretrained_predictions, VGG11_NOT_pretrained_labels = [], []
        for images, labels in tqdm(data_loader.get_val_loader()):
            predictions = VGG11_NOT_pretrained_model(images)
            VGG11_NOT_pretrained_predictions.extend(torch.argmax(predictions, dim=1))
            VGG11_NOT_pretrained_labels.extend(labels)

        # mock
        # ResNet_NOT_pretrained_predictions = [torch.tensor(randint(0, 9)) for _ in range(10000)]
        # ResNet_NOT_pretrained_labels = [torch.tensor(0) if random() < 0.6722 else t for t in ResNet_NOT_pretrained_predictions]
        # VGG11_pretrained_predictions = [torch.tensor(randint(0, 9)) for _ in range(10000)]
        # VGG11_pretrained_labels = [torch.tensor(0) if random() < 0.3722 else t for t in VGG11_pretrained_predictions]
        # VGG11_NOT_pretrained_predictions = [torch.tensor(randint(0, 9)) for _ in range(10000)]
        # VGG11_NOT_pretrained_labels = [torch.tensor(0) if random() < 0.4722 else t for t in VGG11_NOT_pretrained_predictions]

        ResNet_pretrained_confusion_matrix = confusion_matrix(ResNet_pretrained_labels, ResNet_pretrained_predictions)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(ResNet_pretrained_confusion_matrix, annot=True, cmap='Blues', fmt='g', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'ResNet (fine-tuned) for {dataset_name.upper()} - Confusion Matrix')

        plt.show()

        ResNet_NOT_pretrained_confusion_matrix = confusion_matrix(ResNet_NOT_pretrained_labels, ResNet_NOT_pretrained_predictions)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(ResNet_NOT_pretrained_confusion_matrix, annot=True, cmap='Blues', fmt='g', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'ResNet (baseline) for {dataset_name.upper()} - Confusion Matrix')

        plt.show()

        VGG11_pretrained_confusion_matrix = confusion_matrix(VGG11_pretrained_labels, VGG11_pretrained_predictions)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(VGG11_pretrained_confusion_matrix, annot=True, cmap='Blues', fmt='g', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'VGG11 (fine-tuned) for {dataset_name.upper()} - Confusion Matrix')

        plt.show()

        VGG11_NOT_pretrained_confusion_matrix = confusion_matrix(VGG11_NOT_pretrained_labels, VGG11_NOT_pretrained_predictions)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(VGG11_NOT_pretrained_confusion_matrix, annot=True, cmap='Blues', fmt='g', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'VGG11 (baseline) for {dataset_name.upper()} - Confusion Matrix')

        plt.show()

        

