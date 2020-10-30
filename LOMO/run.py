import os
import numpy as np
import cv2
'''get image'''
data_path = 'data'
img_list = os.listdir(data_path)
n=len(img_list)
'''read image'''
if n == 0:
    print('Data directory is empty.')
    exit()
for i in range(n):
    img=cv2.imread(os.path.join(data_path, img_list[i]))
    info=img.shape
    images=np.zeros((info[0],info[1],3,n),dtype=np.int8)
    images[:,:,:,i]=img
    images[:,:,:,i]=cv2.resize(images[:,:,:,i],(128,48))

