import torch 
import torchvision
import torchvision.transforms as T

from helper import * 
from torch.utils.data import DataLoader 

##### Location of the Checkpoints #####
checkpoint_path = ".\checkpoints\cnn_cifar10.pth"

# Parameters 
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 
           'ship', 'truck')
BATCH_SIZE = 128 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##### Main Function 
def main():

    ### Load the test data 
    transforms = T.Compose(
        [T.ToTensor(),
         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cifar10_testset = torchvision.datasets.CIFAR10(root = './Data/',
                                                   train = False, 
                                                   download = True,
                                                   transform = transforms)
    test_loader = DataLoader(cifar10_testset,
                             batch_size = BATCH_SIZE,
                             shuffle = False)
    
    ### Load model from checkpoints
    checkpoint = torch.load(checkpoint_path) 
    start_epoch = checkpoint['epoch'] + 1
    print(f"Load checkpoint from epoch {start_epoch}. \n") 
    model = checkpoint['model'].to(device)
    criterion = checkpoint['criterion'].to(device)

    ### Evaluation
    print("----------- Evaluating ----------")
    test_classifier(dataloader = test_loader,
                    model = model,
                    classes = classes,
                    loss_fn = criterion,
                    device = device)
    print("Done!")

if __name__ == '__main__':
    main() 
