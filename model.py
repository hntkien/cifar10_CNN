import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class Classifier(nn.Module):
  """
  Convolutional Neural Network with 6 convolution layers and 3 fully-connected layers. 
  
  """
  
  def __init__(self, num_classes):
    super(Classifier, self).__init__()

    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.conv_block_3 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.fc_layer = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=4*4*256, out_features=4096),
        nn.ReLU(),
        nn.Linear(in_features=4096, out_features=1024),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=num_classes)
    )

  def forward(self, x):
    # TODO: define your forward function
    x = self.conv_block_1(x)  # (b, 64, 16, 16)
    x = self.conv_block_2(x)  # (b, 128, 8, 8)
    x = self.conv_block_3(x)  # (b, 256, 4, 4)
    x = self.fc_layer(x)

    return x