# import numpy
import numpy as np

# Create and print a 3 by 3 array where every number is a 15
a = np.ones((3,3))*15
print(a)

# print out what are the largest and smalled values in the array below
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr.max())
print(arr.min())

# import pyplot lib from matplotlib and Image lib from PIL
import matplotlib.pyplot as plt
from PIL import Image

# use PIL and matplotlib to read and display the ../data/zebra.jpg image
pic = Image.open('../data/zebra.jpg')
pic_arr=np.array(pic)
plt.imshow(pic_arr)
plt.show()
# convert the image to a numpy arrary and print the shape of the arrary
print(pic_arr.shape)
# use slicing to set the RED and GREEN channels of the picture to 0, then use imshow() to show the isolated blue channel
pic_blue=pic_arr.copy()
pic_blue[:,:,0]=0
pic_blue[:,:,1]=0
plt.imshow(pic_blue)
plt.show()