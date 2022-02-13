#This section covers:
# Indexing and slicing
# Reshaping tensors (tensor views)
# Tensor basic operations
# Dot products
# Matrix multiplication
# Additional, more advanced operations

import torch
import numpy as np

print('Indexing and slicing')
x = torch.arange(6).reshape(3,2)
print(x)

print('Grabbing the right hand column values')
arr1=x[:,1]
print(arr1)

print('Grabbing the right hand column as a (3,1) slice')
arr2=x[:,1].reshape(3,1)
print(arr2)

# view() and reshape() do essentially the same thing by returning a reshaped tensor
# without changing the original tensor in place.


print('Views can infer the correct size')
arr3=x.view(1,-1)
print(arr3)

print('Tensor basic operations')
a=torch.tensor([1,2,3],dtype=torch.float)
b=torch.tensor([4,5,6],dtype=torch.float)
print(a+b)
print(a.add(b))
print(a)

print('Changing a tensor in-place')
a.add_(b)
print(a)

print('Dot products')
print(a.dot(b))

print('Matrix multiplication')
x = torch.tensor([[0,2,4],[1,3,5]])
y = torch.tensor([[6,7],[8,9],[10,11]])
print(x.mm(y))
print(torch.mm(x,y))

print('L2 or Euclidean Norm')
# The L2 norm calculates the distance of the vector coordinate from the origin of the vector space.
# As such, it is also known as the Euclidean norm as it is calculated as the Euclidean distance from the origin.
# The result is a positive distance value.
print(a.norm())



