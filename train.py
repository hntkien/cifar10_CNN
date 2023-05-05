import os 
import torch 
import torch.nn as nn
import torchvision.transforms as transforms 
import numpy as np 
import matplotlib.pyplot as plt 

from torchvision import datasets 
from torch.utils.data import ConcatDataset, DataLoader 
from model import Classifier 
from helper import * 

##### Define Parameters ######

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 
           'ship', 'truck')
BATCH_SIZE = 128 
NUM_CLASSES = len(classes) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters 
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
MOMENTUM = 0.9 
WEIGHT_DECAY = 5E-4 

# Filename and checkpoint
filename = "cnn_cifar10.pth" 
checkpoint_dir = ".\checkpoints"
PATH = os.path.join(checkpoint_dir, filename)

# Specify path to checkpoints to resume training if available
checkpoint_path = None 

##### Gaussian Noise ###### 

def gauss_noise_tensor(img):
    
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32) 
    sigma = 1.0 
    out = img + sigma * torch.randn_like(img) 
    if out.dtype != dtype:
        out = out.to(dtype) 

    return out 

##### Main loop #####

def main():
    ### Define Transformations 

    # 1. Normalization 
    transform1 = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    # 2. Shifting 
    transform2 = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))]
    )
    # 3. Rotation
    transform3 = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.RandomRotation(degrees=(0,180))]
    )
    # 4. Horizontal Flip 
    transform4 = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.RandomHorizontalFlip(p=0.5)]
    )
    # 5. Gaussian Noise 
    transform5 = transforms.Compose(
        [transforms.PILToTensor(),
         gauss_noise_tensor,
         transforms.ConvertImageDtype(torch.float32),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    # Combine the transformation list 
    transform_list = [transform1, transform2, transform3, transform4,           transform5]

    ### Load the Data and Apply Data Augmentations

    # Create an empty data set 
    augmented_dataset = [] 
    # Download and apply transformation 
    for t in transform_list:
        augmented_dataset.append(datasets.CIFAR10(root = './Data/',
                                                 train = True,
                                                 download = True,
                                                 transform = t))
    cifar10_trainset = ConcatDataset(augmented_dataset) 

    ### DataLoader 
    train_loader = DataLoader(cifar10_trainset, 
                              batch_size = BATCH_SIZE,
                              shuffle = True)

    # Initialize untrained model if no checkpoint specified, otherwise, load model and resume training 
    if checkpoint_path is None: 
        start_epoch = 0 
        model = Classifier(num_classes=NUM_CLASSES) 
        criterion = nn.CrossEntropyLoss() 
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr = LEARNING_RATE, 
                                    momentum = MOMENTUM,
                                    weight_decay = WEIGHT_DECAY)
        epoch_loss = []  # Training from scratch 

    else: 
        checkpoint = torch.load(checkpoint_path) 
        start_epoch = checkpoint['epoch'] + 1
        print(f"Load checkpoint from epoch {start_epoch}. \n") 
        model = checkpoint['model']
        criterion = checkpoint['criterion'] 
        optimizer = checkpoint['optimizer'] 
        epoch_loss = checkpoint['epoch_loss']

    # Move model to default device 
    model = model.to(device) 
    criterion = criterion.to(device)
    for epoch in range(start_epoch, NUM_EPOCHS):
        running_loss = train_classifier(dataloader = train_loader,
                                        model = model,
                                        criterion = criterion,
                                        optimizer = optimizer,
                                        device = device)
        # Append the average loss for each epoch
        epoch_loss.append(running_loss)
        # Save checkpoint for each epoch 
        save_checkpoint(epoch = epoch,
                        model = model,
                        criterion = criterion,
                        optimizer = optimizer,
                        epoch_loss = epoch_loss,
                        path = PATH)
        print(f"Epoch: {epoch+1} \t"
              f"Training loss: {running_loss:.4f}")
    # Print out a message when the training finishes and plot the training loss
    print("Training Finishes!")
    plot_history(epoch_loss)

if __name__ == '__main__':
    main() 