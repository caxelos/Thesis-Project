from __future__ import print_function
import numpy as np
import os
import scipy.io as sio
import pickle as pickle
import json

data_path = '../Normalized/'
image_width = 60  # Pixel width and height.
image_height = 36
pixel_depth = 255.0  # Number of levels per pixel.
totals = {}    

def calcs():
    global totals
    dir_path = os.getcwd()
    LIST_DIRS = os.walk(dir_path)    
    totals={}
    for ROOT, DIRS, FILES in LIST_DIRS:#outter
        for i in range(len(DIRS)):
            if DIRS[i] == '..Normalized':
                print("deleting ",DIRS[i], " at position ",i)
                del DIRS[i]
                break     
        for D in DIRS:#now get in pij
            print(D)
            totals[D]={}
            totals[D]['sum']=0
            list_dirs = os.walk(os.path.join(ROOT,D))
            for root, dirs, files in list_dirs:#inner
                for f in files:#now get in .mat
                    totals[D][f]={}
                    mat_contents=sio.loadmat(os.path.join(root, f))
                    data = mat_contents['data']
                    right = data['right']#(40,3)
                    right=right[0,0]
                    totals[D][f]=len(right['gaze'][0,0])
                    totals[D]['sum'] = totals[D]['sum'] + len(right['gaze'][0,0])                    
        
        for D in DIRS:            
            list_dirs = os.walk(os.path.join(ROOT,D))
            for root, dirs, files in list_dirs:#inner
                for f in files:#now get in .mat
                    mat_contents=sio.loadmat(os.path.join(root, f))
                    data = mat_contents['data']
                    right = data['right']
                    right=right[0,0]
                    totals[D][f]= int(round((totals[D][f]*2000)/totals[D]['sum']))#aristero melos: totals[D][f]['per_f']
    json.dump(totals, open("text.txt",'w'))




def read_mat(root_mat,pij, f):
    global totals
    mat_contents=sio.loadmat(root_mat)
    data = mat_contents['data']
    right = data['right']#(40,3)
    right=right[0,0]    
    left = data['left']
    left=left[0,0]
    
    if right['gaze'][0,0].shape[0]<totals[pij][f]:
        totals[pij][f]=right['gaze'][0,0].shape[0]
        #print("entered here!!!!!!!!!!!!!!!!!!!!!")
    
    indices=np.random.choice(right['gaze'][0,0].shape[0],totals[pij][f],replace=False)
    #right_gaze = right['gaze'][0,0][indices]
    #right_img = right['image'][0,0][indices]
    #right_pose = right['pose'][0,0][indices]
    #left_gaze = left['gaze'][0,0][indices]
    #left_img = left['image'][0,0][indices]
    left_pose=[]
    right_pose=[]
    left_gaze=[]
    right_gaze=[]
    right_img=[]
    left_img=[]
    for indice in indices:
        left_pose.append(convert_pose(left['pose'][0,0][indice]))
        right_pose.append(convert_pose(right['pose'][0,0][indice])* np.array([-1, 1]))
        left_gaze.append(convert_pose(left['gaze'][0,0][indice]))
        right_gaze.append(convert_pose(right['gaze'][0,0][indice])* np.array([-1, 1]))
        right_img.append(right['image'][0,0][indice])
        left_img.append(left['image'][0,0][indice])

    left_img = np.array(left_img).astype(np.float32) / 255
    left_pose = np.array(left_pose).astype(np.float32)
    left_gaze = np.array(left_gaze).astype(np.float32)
    right_img = np.array(right_img).astype(np.float32) / 255
    right_pose = np.array(right_pose).astype(np.float32)
    right_gaze = np.array(right_gaze).astype(np.float32)



    return np.concatenate((right_gaze,left_gaze),axis=0),np.concatenate((right_img,left_img),axis=0), np.concatenate((right_pose,left_pose),axis=0)
    


def open_file(rootDir, d):
    global totals
    list_dirs = os.walk(rootDir)
    dataset_img = []
    dataset_gaze = []
    dataset_pose = []
    for root, dirs, files in list_dirs: #gia kathe person
        for f in files: #gia kathe .mat arxeio
            if totals[d][f] != 0:                
                gaze, img, pose=read_mat(os.path.join(root, f), d, f)
                if dataset_img==[]:
                    dataset_img = img
                    dataset_gaze=gaze
                    dataset_pose=pose
                else:
                    #print("shape1:",dataset_img.shape,", shape2:",img.shape)
                    dataset_img=np.append(dataset_img,img,axis = 0)
                    dataset_gaze=np.append(dataset_gaze,gaze,axis = 0)
                    dataset_pose=np.append(dataset_pose,pose,axis = 0)
                
    return dataset_gaze,dataset_img,dataset_pose

import cv2

def convert_pose(vect):
    M, _ = cv2.Rodrigues(np.array(vect).astype(np.float32))
    vec = M[:, 2]
    yaw = np.arctan2(vec[0], vec[2])
    pitch = np.arcsin(vec[1])
    return np.array([yaw, pitch])

def convert_gaze(vect):
    x, y, z = vect
    yaw = np.arctan2(-x, -z)
    pitch = np.arcsin(-y)
    return np.array([yaw, pitch])


                      
def make_dataset():
    global totals

    dir_path = os.getcwd()#+'#os.getcwd()+data_path
    list_dirs = os.walk(dir_path)
    
   
    totals=json.load(open("text.txt"))
    # dirs=15 people
    #prepei na paw (1500 left + 1500 right apto kathe outer loop)
    print("ok?!")
    for root, dirs,files in list_dirs:

        for i in range(len(dirs)):
            if dirs[i] == '..Normalized':
                print("deleting ",dirs[i], " at position ",i)
                del dirs[i]
                break 
        for d in dirs: # for each person
            # dataset_img = []
            # dataset_gaze = []
            # dataset_pose = []
            print(d)
            gaze,img,pose=open_file(os.path.join(root,d),d)
            # if dataset_img == []:
            #     dataset_img = img
            #     dataset_gaze = gaze
            #     dataset_pose = pose
            # else:
            #     dataset_img = np.append(dataset_img, img, axis=0)
            #     dataset_gaze = np.append(dataset_gaze, gaze, axis=0)
            #     dataset_pose = np.append(dataset_pose, pose, axis=0)
            #outpath = os.path.join(outdir, subject_id)
            #np.savez('../../../newdata/'+d+'.npz', image=dataset_img, pose=dataset_pose, gaze=dataset_gaze)
            np.savez('../../../newdata/'+d+'.npz', image=img, pose=pose, gaze=gaze)
    return
    #return dataset_gaze, dataset_img, dataset_pose

def randomize(dataset_gaze, dataset_img, dataset_pose):
  
  permutation = np.random.permutation(dataset_pose.shape[0])
  shuffled_gaze = dataset_gaze[permutation,:]
  shuffled_img = dataset_img[permutation,:,:]
  shuffled_pose=dataset_pose[permutation,:]
  return shuffled_gaze, shuffled_img,shuffled_pose


#calcs()
make_dataset()#22497...xanw mono 2*3 paratiriseis(2*1500 to atomo!)

# gaze,img,pose=randomize(gaze,img,pose)

# print(gaze.shape,img.shape,pose.shape)

# pickle_file = 'dataset.pickle'

# try:
#   f = open(pickle_file, 'wb')
#   save = {
#    'gaze': gaze,
#    'img': img,
#    'pose': pose,
#    }
#   pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
#   f.close()
# except Exception as e:
#  print('Unable to save data to', pickle_file, ':', e)
#  raise

# statinfo = os.stat(pickle_file)
# print('Compressed pickle size:', statinfo.st_size)






