# import numpy
import numpy as np

print('creating list [0,1,2,3,4] in python and print the type of your list')
my_list=[0,1,2,3,4]
print(type(my_list))

print('cast your list into a numpy arrary and print the type of your array')
my_array=np.array(my_list)
print(type(my_array))

print('create a 3x3 array filled with zeros')
a=np.zeros((3,3))
print(a)

print('create a 2x4 array filled with ones')
b=np.ones((2,4))
print(b)

print('create a 3x4 array filled with tens')
c=np.ones((3,4))*10
print(c)

print('create a 3x2 array filled with random numbers')
d=np.random.rand(3,2)
print(d)

print('create a 3x2 arrary filled with random integers')
e=np.random.randint(low=0,high=100,size=(3,2))
print(e)

print('create an 9 elements array [1,2,3,4,5,6,7,8,9]')
print('Values are generated within the half-open interval [start, stop) (in other words, the interval including start but excluding stop)')
f=np.arange(1,10)
print(f)

print('reshape your array (1x9) into a 3x3 matrix')
mat=f.reshape(3,3)
print(mat)

print('print the shape of your matrix')
print(mat.shape)

print('print the maximum number in your matrix')
print(mat.max())

print('print the minimum number in your matrix')
print(mat.min())

print('print the mean value in your matrix')
print(mat.mean())

print('retrieve the element on first row, second column')
print(mat[0,1])

print('retrieve all the elements on second column')
print(mat[:,1])

print('retrieve all the elements on first row')
print(mat[0,:])

print('retrieve all the elements on first two rows and last two columns')
print(mat[0:2,-2:])

