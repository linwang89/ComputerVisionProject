# Before we start working with Convolutional Neural Networks (CNN), let's model the MNIST dataset using only linear layers.
# In this project we'll reshape the MNIST data from a 28x28 image to a flattened 1x784 vector to mimic a single row of 784 features.

# Perform standard imports
import torch
import torch.nn as nn
import torch.nn.functional as F          # adds some efficiency
from torch.utils.data import DataLoader  # lets us load data in batches
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Load the MNIST dataset
# Load the training set
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
# Load the test set
test_data = datasets.MNIST(root='./data',train=False,download=True,transform=transform)

print(train_data)
print(test_data)

# Examine a training record
print(train_data[0])
image, label=train_data[0]
print('Shape:',image.shape,'\nLabel',label)
plt.imshow(train_data[0][0].reshape((28,28)))
plt.show


# Batch loading with DataLoader
train_loader = DataLoader(train_data, batch_size=100,shuffle=True)
test_loader = DataLoader(test_data, batch_size=500,shuffle=False)
# Define the ANN model
class MultilayerPerception(nn.Module):
    def __init__(self,in_sz=784,out_sz=10,layers=[120,84]):
        super().__init__()
        self.fc1 = nn.Linear(in_sz,layers[0])
        self.fc2 = nn.Linear(layers[0],layers[1])
        self.fc3 = nn.Linear(layers[1], out_sz)

    def forward(self,X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.log_softmax(self.fc3(X),dim=1)
        return X

# Check our model and Count the model parameters
model =  MultilayerPerception()
print(model)

for param in model.parameters():
    print(param.numel())

# Define loss function & optimizer
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

# How to Flatten the training data? [1,28,28] --> [784]
for image,labels in train_loader:
    print(image.shape)
    break
image.view(100,-1)
# Train the model
# First thing that I want to do is I'm going to import time.
epochs = 3
import time
start_time = time.time()

for i in range(epochs):
    # let's define some variables for tracking purpose and for visualization the result later on
    train_corr = 0
    test_corr = 0

    # Run the training batches
    for b,(X,y) in enumerate(train_loader):
        b += 1
        # Apply the model
        y_pred = model(X.view(100,-1))
        loss = cost(y_pred, y)

        # Tally the number of correct predictions
        predicted = torch.max(y_pred.data,1)[1]
        batch_corr = (predicted == y).sum()
        train_corr += batch_corr

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print interim results
        # Update train loss & accuracy for the epoch
        if b%200 == 0:
            print(f'epoch:{i} batch {b} loss:{loss.item()} accuracy: {train_corr*100/(b*100)}%')


    # Run the testing batches
    with torch.no_grad():
        for b, (X, y) in enumerate(test_loader):
            y_val = model(X.view(500, -1))
            predicted = torch.max(y_val.data, 1)[1]
            test_corr += (predicted == y).sum()
    # Update test loss & accuracy for the epoch
    print(f'Test accuracy: {test_corr*100/len(test_data)}')

print(f'\nDuration:{time.time()-start_time: .0f} seconds')

# Evaluate the model
x=2019
plt.figure(figsize=(1,1))
plt.imshow(test_data[x][0].reshape((28,28)))
plt.show()

model.eval()
with torch.no_grad():
    new_pred = model(test_data[x][0].view(1,-1)).argmax()
print(new_pred.item())
