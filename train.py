import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataset.mnist import MNISTDataLoader
from dataset.caltech101 import Caltech101DataLoader
from models.resnet import ResNet
from models.vgg import VGG11
from utils.trainer import Trainer
from utils.utlis import set_seed

def main():
    # Set random seed for reproducibility
    set_seed(seed=7)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--model_name', type=str, default='resnet', help='Name of the model to use')
    parser.add_argument('--dataset_name', type=str, default='mnist', help='Name of the dataset to use')
    parser.add_argument('--pretrained', type=str, default=False, help='Loading pretrained weights')
    args = parser.parse_args()

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define data loader
    if args.dataset_name=='mnist':
        data_loader = MNISTDataLoader(batch_size=args.batch_size)
        num_classes = 10
        in_channels = 1
    elif args.dataset_name=='caltech101':
        data_loader = Caltech101DataLoader(batch_size=args.batch_size)
        num_classes = 101
        in_channels = 3
    else:
        raise NotImplementedError(f'DataLoader for the {args.dataset_name} dataset is not implemented.')

    # Define model
    if args.model_name == 'resnet':
        if(args.pretrained): model = ResNet(num_classes=num_classes, in_channels=in_channels, pretrained=True)
        else: model = ResNet(num_classes=num_classes, in_channels=in_channels, pretrained=False)
    elif args.model_name == 'vgg':
        if(args.pretrained): model = model = VGG11(num_classes=num_classes, in_channels=in_channels, pretrained=True)
        else: model = VGG11(num_classes=num_classes, in_channels=in_channels)
    else:
        raise NotImplementedError('Model {} is not implemented.'.format(args.model_name))

    # Move model to device
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Define trainer
    trainer = Trainer(model, data_loader, optimizer, criterion, device, args.pretrained)

    # Train the model
    trainer.train(num_epochs=args.num_epochs)

if __name__ == '__main__':
    main()    