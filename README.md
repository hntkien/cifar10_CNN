# CIFAR-10 Image Classification

Perform different types of data augmentation to the Cifar10 dataset and implement a CNN to classify images. Instead of randomly apply transformations to the dataset, we append the transformed data to the training set. We use 5 types of transformations, including:
- Normalizing 
- Shifting: Randomly shift the images up/down and left/right by within 10%
- Rotating: Randomly rotate the images by some angles
- Flipping: Horizontally flip the images
- Adding Noise: Randomly add some small Gaussian noise to the images
Thus, after the above data augmentation, our training set consists of $50,000 \times 5$ training images (rather than 50,000 images with random transformation). 

One can use any architecture for the classification network. Here, we use a simple LeNet with some modification. 
