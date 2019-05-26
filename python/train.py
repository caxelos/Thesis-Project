from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression

import matplotlib
import scipy.io as sio
import pickle as pickle
#import tensorflow as tf
import numpy as np
import math
import cv2
#
# train_gaze = []
# train_img = []
# train_pose = []
#
# valid_gaze = []
# valid_img = []
# valid_pose =[]
#
# test_gaze = []
# test_img = []
# test_pose = []

### TODO:
#1) Dokimase ta regression forests me allo DB
#2) Dokimase tin ulopoihsh toy Caffe kai ftiakse ta data
#3) Dokimase to se Keras kai kanto import meta

with open('dataset.pickle','rb') as f:
    save = pickle.load(f,encoding='latin1')
    gaze = save['gaze']
    img = save['img']
    pose = save['pose']


        #gaze, img, pose=pickle.load(f)


tmp = np.zeros((len(gaze),2))
tmp2 = np.zeros((len(gaze),2))
rotation_matrix = np.zeros(shape=(3,3))
for i in range(len(gaze)):
    tmp[i,0]= math.asin(-gaze[i,1])
    tmp[i,1]=math.atan2(-gaze[i,0],-gaze[i,2])
    cv2.Rodrigues(gaze[i,:],rotation_matrix)
    tmp2[i,0]=math.asin(rotation_matrix[1][2])
    tmp2[i,1]=math.atan2(rotation_matrix[0][2],rotation_matrix[2][2])

   
gaze = tmp
pose=tmp2
train_gaze=gaze[0:200000,:]
train_img= img[0:200000,:,:]
train_pose=pose[0:200000,:]

#valid_gaze=gaze[200001:210000,:]
#valid_img=img[200001:210000,:,:]
#valid_pose=pose[200001:210000,:]

test_gaze=gaze[210001:213658,:]
test_img=img[210001:213658,:,:]
test_pose=pose[210001:213658,:]
n_sample = len(train_img)

#train_img=np.reshape(train_img,(len(train_img),2160))
#valid_img=np.reshape(valid_img,(len(valid_img),2160))
#test_img=np.reshape(test_img,(len(test_img),2160))

# # #todo
# numGrps=-1
# for i in range(len(pose[:,1])):
#     if can_be_center(groups, theta, phi, numGrps, curr_dist):
#         numGrps = numGrps+1
#         groups[numGrps]=[pose[i,0],pose[i,1]]

# #allocate mem
# for i in range(numGrps):
#     groups[i] = struct(gazes,poses,data)

# for i in range(len(pose[:,1])):
#     groupID = find_nearest_group(headpose,groups,numGrps)

# import h5py
# with h5py.File('train_dataset.h5','w') as hdf:
#     hdf.create_group()
#     hdf.create_dataset('gazes',data=train_gaze)
#     hdf.create_dataset('poses',data=train_pose)
#     hdf.create_dataset('label',data=train_img)
# with h5py.File('test_dataset.h5','w') as hdf:
#     hdf.create_dataset('gazes',data=test_gaze)
#     hdf.create_dataset('poses',data=test_pose)
#     hdf.create_dataset('label',data=test_img)




from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# # Initialising the CNN
classifier = Sequential()
classifier.add(Conv2D(20, (5, 5), input_shape = (36, 60, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(50, (5, 5), input_shape = (36, 60, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 500, activation = 'relu'))
classifier.add(Dense(units = 2, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])



from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

train_img = train_img.reshape(len(train_img[:,1,1]),36,60,1)
test_img = test_img.reshape(len(test_img[:,1,1]),36,60,1)

train_datagen.fit(train_img)
test_datagen = ImageDataGenerator(rescale = 1./255)
classifier.fit_generator(train_datagen.flow(train_img,train_gaze, batch_size=32),
                    steps_per_epoch=len(train_img[:,1,1]) / 32, epochs=1,
                    validation_data = test_img,
                    validation_steps = 2000)

