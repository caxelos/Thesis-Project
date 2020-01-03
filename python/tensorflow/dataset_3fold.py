from __future__ import print_function
import numpy as np
import os
import scipy.io as sio
import pickle as pickle

data_path = '../Normalized/'
image_width = 60  # Pixel width and height.
image_height = 36
pixel_depth = 255.0  # Number of levels per pixel.
    

def read_mat(root_mat):
    mat_contents=sio.loadmat(root_mat)
    data = mat_contents['data']
    right = data['right']#(40,3)
    right=right[0,0]    
    left = data['left']

    right_gaze = right['gaze'][0,0]
    right_img = right['image'][0,0]
    right_pose = right['pose'][0,0]
    
    #print(right_gaze.shape)
    return right_gaze, right_img, right_pose



def open_file(rootDir):
    list_dirs = os.walk(rootDir)
    dataset_img = []
    dataset_gaze = []
    dataset_pose = []
    for root, dirs, files in list_dirs: #gia kathe person
        print(root)
        for f in files: #gia kathe .mat arxeio
            print(f)
                
            gaze, img, pose=read_mat(os.path.join(root, f))
            if dataset_img==[]:
                dataset_img=img
                dataset_gaze=gaze
                dataset_pose=pose
            else:
                dataset_img=np.append(dataset_img,img,axis = 0)
                dataset_gaze=np.append(dataset_gaze,gaze,axis = 0)
                dataset_pose=np.append(dataset_pose,pose,axis = 0)
    return dataset_gaze,dataset_img,dataset_pose





def calcs():
    dir_path = os.getcwd()
    LIST_DIRS = os.walk(dir_path)
    for ROOT, DIRS, FILES in LIST_DIRS:#outter
        for i in range(len(DIRS)):
            print(i)
            if DIRS[i] == '..Normalized':
                print("deleting ",DIRS[i], " at position ",i)
                del DIRS[i]
                break 
            
        for D in DIRS:#now get in pij
            print(D)
            list_dirs = os.walk(os.path.join(ROOT,D))
            for root, dirs, files in list_dirs:#inner
                for f in files:#now get in .mat
                    print(os.path.join(root, f))
                    mat_contents=sio.loadmat(os.path.join(root, f))
                    data = mat_contents['data']
                    right = data['right']#(40,3)
  

def make_dataset():
    dir_path = os.getcwd()#+'#os.getcwd()+data_path
    print(dir_path)
    list_dirs = os.walk(dir_path)
    dataset_img = []
    dataset_gaze = []
    dataset_pose = []
    i=0
    # dirs=15 people
    #prepei na paw (1500 left + 1500 right apto kathe outer loop)
    for root, dirs,files in list_dirs:
        for d in dirs: # for each person
            
            gaze,img,pose=open_file(os.path.join(root,d))
            
            if dataset_img == []:
                dataset_img = img
                dataset_gaze = gaze
                dataset_pose = pose
            else:
                dataset_img = np.append(dataset_img, img, axis=0)
                dataset_gaze = np.append(dataset_gaze, gaze, axis=0)
                dataset_pose = np.append(dataset_pose, pose, axis=0)
            i=i+1
            if i > 5:
            	break
        if i > 5:
        	break
    #print(len(dataset_img))
    return dataset_gaze, dataset_img, dataset_pose

def randomize(dataset_gaze, dataset_img, dataset_pose):
  
  permutation = np.random.permutation(dataset_pose.shape[0])
  shuffled_gaze = dataset_gaze[permutation,:]
  shuffled_img = dataset_img[permutation,:,:]
  shuffled_pose=dataset_pose[permutation,:]
  return shuffled_gaze, shuffled_img,shuffled_pose


calcs()
#gaze,img,pose=make_dataset()
#gaze,img,pose=randomize(gaze,img,pose)

#print(gaze.shape,img.shape,pose.shape)

pickle_file = 'dataset.pickle'

#try:
  #f = open(pickle_file, 'wb')
  #save = {
  #  'gaze': gaze,
  #  'img': img,
  #  'pose': pose,
  #  }
  #pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  #f.close()
#except Exception as e:
#  print('Unable to save data to', pickle_file, ':', e)
#  raise

#statinfo = os.stat(pickle_file)
#print('Compressed pickle size:', statinfo.st_size)






