import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
transform = transforms.ToTensor()

train_data = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
class_names = ['T-shirt','Trouser','Sweater','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot']

# Create data loaders
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=True)

# Examine a batch of images
# Use DataLoader, make_grid and matplotlib to display the first batch of 10 images.
# display the labels as well
first_batch = next(iter(train_loader))
g = make_grid(first_batch[0],nrow=5)
plt.figure(figsize=(1,1))
plt.imshow(g.permute(1, 2, 0))
plt.show()
# Downsampling
# If a 28x28 image is passed through a Convolutional layer using a 5x5 filter, a step size of 1, and no padding,
# create the conv layer and pass in one data sample as input, then printout the resulting matrix size
conv1 = nn.Conv2d(1, 1, 5, 1)
X_train = train_data[0][0]
x = X_train.view(1,1,28,28)
x = conv1(x)
print(x.shape)

# If the sample from question 3 is then passed through a 2x2 MaxPooling layer
# create the pooling layer and pass in one data sample as input, then printout the resulting matrix size
x = F.max_pool2d(x, 2, 2)
print(x.shape)


#test
# X_train = train_data[0][0]
# conv1 = nn.Conv2d(1, 6, 5, 1)
# conv2 = nn.Conv2d(6, 16, 3, 1)
# x = X_train.view(1,1,28,28)
# x = conv1(x)
# x = F.max_pool2d(x, 2, 2)
# x = conv2(x)
# x = F.max_pool2d(x, 2, 2)
# print(x.shape)

# Define a convolutional neural network
# Define a CNN model that can be trained on the Fashion-MNIST dataset.
# The model should contain two convolutional layers, two pooling layers, and two fully connected layers.
# You can use any number of neurons per layer so long as the model takes in a 28x28 image and returns an output of 10.
# and then printout the count of parameters of your model
class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)

        self.fc1 = nn.Linear(5*5*16, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 5*5*16)
        X = F.relu(self.fc1(X))
        X = F.log_softmax(self.fc2(X), dim=1)
        return X


# Define loss function & optimizer
# Define a loss function called "criterion" and an optimizer called "optimizer".
# You can use any loss functions and optimizer you want,
# although we used Cross Entropy Loss and Adam (learning rate of 0.001) respectively.
model = CNN1()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train and test the model
# try with any epochs you want
# and printout some interim results
epochs = 3

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    # Run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 3000 == 0:
            print(f'epoch:{i} batch:{b} loss: {loss.item()} accuracy: {trn_corr.item() * 100 / (10 * b)}%')

    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test)

            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()
    print(f'Test accuracy: {tst_corr.item() * 100 / (len(test_data))}%')

# Remember, always experiment with different architecture and different hyper-parameters, such as
# different activation function, different loss function, different optimizer with different learning rate
# different size of convolutional kernels, and different combination of convolutional layers/pooling layers/FC layers
# to make the best combination for solving your problem in real world

