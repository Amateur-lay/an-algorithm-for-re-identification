import os
import numpy as np
import cv2
import lomo
'''get image'''
data_path = 'data'
img_list = os.listdir(data_path)
n=len(img_list)
'''read image'''
if n == 0:
    print('Data directory is empty.')
    exit()
'''initiate images'''
img=cv2.imread(os.path.join(data_path, img_list[0]))
info=img.shape
images = np.zeros((info[0], info[1], 3, n), dtype=np.int8)
'''save images'''
for i in range(n):
    img=cv2.imread(os.path.join(data_path, img_list[i]))
    images[:,:,:,i]=img
descriptors = lomo.lomo(images)
des=descriptors.lomo()
