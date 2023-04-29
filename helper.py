# Import libraries 
import torch 
import numpy as np 
import matplotlib.pyplot as plt 

### Function to visualize the data
def data_viz(img):
    img = img / 2 + 0.5  # Unnormalize 
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0)) 

### Training Function 
def train_classifier(dataloader, model, loss_fn, optimizer, device):
    """ 
    Train the model in the current epoch. 

    :param dataloader : Training dataloader 
    :param model      : Defined network architecture 
    :param loss_fn    : Loss function 
    :param optimizer  : Optimzier 
    :param device     : Default device 
    -> return the average loss over batches in one epoch 

    """
    model.train() 
    running_loss = [] 

    for batch, (images, labels) in enumerate(dataloader):

        # Forward pass: compute prediction and loss 
        images, labels = images.to(device), labels.to(device) 
        preds = model(images) 
        loss = loss_fn(preds, labels) 
        running_loss.append(loss.item())

        # Back propagation 
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 

    return np.average(running_loss) 

### Testing Function 
def test_classifier(dataloader, classes,  model, loss_fn, device):
    """ 
    Function to test the trained model. Print out the average accuracy as well as accuracy for each class in the dataset.

    :param dataloader : Test dataloader 
    :param classes    : Classes for the dataset
    :param model      : Defined network architecture 
    :param loss_fn    : Loss function  
    :param device     : Default device 

    """
    size = len(dataloader.dataset) 
    test_loss = [] 
    accuracy = 0.0 
    correct_preds = {classname: 0 for classname in classes}
    total_preds = {classname: 0 for classname in classes} 

    model.eval() 
    with torch.no_grad():

        for images, labels in dataloader: 
            images, labels = images.to(device), labels.to(device) 
            preds = model(images) 
            test_loss.append(loss_fn(preds, labels).item()) 
            accuracy +=(preds.argmax(1)==labels).type(torch.float).sum().item()

            # Calcualtion for each class 
            for label, pred in zip(labels, preds.argmax(1)):
                if label == pred:
                    correct_preds[classes[label]] += 1 
                total_preds[classes[label]] += 1 

    test_loss = np.average(test_loss)
    accuracy /= size 
    print(f"Evaluation: \n Accuracy: {(100*accuracy):>0.1f}% \t Average Loss: {test_loss:>8f} \n")

    # Print the accuracy for each class 
    for classname, correct_count in correct_preds.items():
        class_acc = 100 * float(correct_count) / total_preds[classname] 
        print(f"Accuracy for class {classname:5s} is {class_acc:.1f}%") 

### Function to plot the training results
def plot_history(data):
    plt.plot(data) 
    plt.xlabel("Epoch") 
    plt.ylabel("Loss") 
    plt.title("Training Losses over each Epoch") 
    plt.show() 

### Function to save checkpoint for every epoch 
def save_checkpoint(epoch, model, loss_fn, optimizer, epoch_loss, path):
    """ 
    Save model checkpoint. 
    
    :param epoch     : Current epoch number
    :param model     : Model 
    :param optimizer : Optimizer 
    :param loss_fn   : Loss function used 
    :param epoch_loss: Loss for each epoch upto 'epoch'
    :param filename  : Name for the checkpoint file
    
    """
    state = {'epoch': epoch,
             'model': model,
             'criterion': loss_fn,
             'optimizer': optimizer,
             'epoch_loss': epoch_loss}
    torch.save(state, path)