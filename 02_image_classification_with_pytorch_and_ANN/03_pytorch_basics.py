#This section covers:
# Converting NumPy arrays to PyTorch tensors
# Creating tensors from scratch

# import libs torch, numpy
import torch
import numpy as np


print('converting NumPy arrays to PyTorch tensors')
# A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.
# Calculations between tensors can only happen if the tensors share the same dtype.

print('lets simply create a numpy array with value 1,2,3,4,5')
arr=np.array([1,2,3,4,5])
x=torch.tensor(arr)
print(x)

print('now lets create a 4x3 2D array (matrix)')
arr2=np.arange(0,12)


print('lets convert this 2D array into a torch tensor')
y=torch.tensor(arr2)
print(y)

print('lets create a tensor from scratch')
#   Uninitialized tensors with .empty()
z=torch.empty(3,4)
print(z)

#   Initialized tensors with .zeros() and .ones()
arr1=torch.zeros(2,3)
arr2=torch.ones(4,5)
print(arr1)
print(arr2)

print('Tensors from ranges')
arr3=torch.arange(0,12)
print(arr3)

print('Tensors from data')
arr4=torch.tensor([1,2,3,4,5])
print(arr4)

print('Random number tensors that follow the input size')
arr5=torch.rand(3,4)
print(arr5)

print('Set random seed which allows us to share the same "random" results.')
torch.manual_seed(1)
arr6=torch.rand(3,4)
print(arr6)

print('Tensor attributes')
print(arr6.shape)
print(arr6.device)

# PyTorch supports use of multiple devices, harnessing the power of one or more GPUs in addition to the CPU.
# We won't explore that here, but you should know that operations between tensors can only happen for tensors installed on the same device.