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
import keras.backend as K
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





def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


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


gaze=tmp
pose=tmp2
train_gaze=gaze[0:7000,:]
train_img= img[0:7000,:,:]
train_pose=pose[0:7000,:]
valid_gaze=gaze[7000:8000,:]#gaze[7000:8000,:]
valid_img=img[7000:8000,:,:]
valid_pose=pose[7000:8000,:]

test_gaze=gaze[8000:len(gaze[:,0]),:]
test_img=img[8000:len(gaze[:,0]),:,:]
test_pose=pose[8000:len(gaze[:,0]),:]
n_sample=len(train_img)
print(len(train_img[:,1,1]))
print(len(train_pose[:,1]))
print(len(train_gaze[:,1]))
del gaze,img,pose,tmp,tmp2
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



import keras
from keras.utils import plot_model
from keras.models import Sequential,Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Concatenate,Input


img_input = Input(shape=(36, 60, 1),name='img_input')
pose_input = Input(shape=(2,), name='pose_input')
x=Conv2D(20, (5, 5),activation = 'relu')(img_input)
x=MaxPooling2D(pool_size = (2, 2))(x)
x=Conv2D(20, (5, 5),activation = 'relu')(x)
x=MaxPooling2D(pool_size = (2, 2))(x)
x=Flatten()(x)
x=Dense(units = 500, activation = 'relu')(x)
mixed = keras.layers.concatenate([x, pose_input])
cnn_output = Dense(units = 2, activation = 'sigmoid',name='gaze_output')(mixed)

model = Model(inputs=[img_input,pose_input], outputs=cnn_output)
model.compile(optimizer = 'adam', loss = euclidean_distance_loss, metrics = ['accuracy'])
print(model.summary())
plot_model(model, to_file='multilayer_perceptron_graph.png')


# # Initialising the CNN
# classifier = Sequential()
# classifier.add(Conv2D(20, (5, 5), input_shape = (36, 60, 1), activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
# classifier.add(Conv2D(50, (5, 5), input_shape = (36, 60, 1), activation = 'relu'))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
# classifier.add(Flatten())
# classifier.add(Dense(units = 500, activation = 'relu'))
# classifier.add(Dense(units = 2, activation = 'sigmoid'))

# # Compiling the CNN
# classifier.compile(optimizer = 'adam', loss = euclidean_distance_loss, metrics = ['accuracy'])

train_img = train_img.reshape(len(train_img[:,1,1]),36,60,1)
test_img = test_img.reshape(len(test_img[:,1,1]),36,60,1)
valid_img = valid_img.reshape(len(valid_img[:,1,1]),36,60,1)


# train_pose=np.array(train_pose,np.float)
# train_gaze=np.array(train_gaze,np.float)
# test_gaze=np.array(test_gaze,np.float)
valid_pose=np.array(valid_pose)
# print("*** valid_img=",valid_img.shape)
# print("*** valid_pose=",valid_pose.shape)
# print("*** valid_gaze=",valid_gaze.shape)

import popa
popa.train_model(model=model,
            x_train=train_img,
            x_train_feat=train_pose,
            y_train=train_gaze,
            x_test=test_img,
            x_test_feat=test_pose,
            y_test=test_gaze,
            x_val_feat=valid_pose,
            train_batch_size=32,
            test_batch_size=32,
            epochs=10)





# from keras.preprocessing.image import ImageDataGenerator
# train_datagen = ImageDataGenerator(rescale = 1./255,
#                                    shear_range = 0.2,
#                                    zoom_range = 0.2,#,
#                                   horizontal_flip = True)

# test_datagen = ImageDataGenerator(rescale = 1./255)

# train_datagen.fit(train_img)

# model.fit([train_img,train_pose],[conv_output,cnn_output])

# model.fit_generator(train_datagen.flow({'img_input':train_img, 'pose_input':train_pose},train_gaze, batch_size=64),
#                     steps_per_epoch=len(train_img[:,1,1]) / 32, epochs=4,
#                     validation_data = (test_img,test_gaze),
#                     validation_steps = 2000)

