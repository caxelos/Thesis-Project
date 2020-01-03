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
            print(i)
            if DIRS[i] == '..Normalized':
                print("deleting ",DIRS[i], " at position ",i)
                del DIRS[i]
                break     
        for D in DIRS:#now get in pij
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
                    totals[D][f]['f']=len(right['gaze'][0,0])
                    totals[D]['sum'] = totals[D]['sum'] + len(right['gaze'][0,0])                    
        
        for D in DIRS:            
            list_dirs = os.walk(os.path.join(ROOT,D))
            for root, dirs, files in list_dirs:#inner
                for f in files:#now get in .mat
                    mat_contents=sio.loadmat(os.path.join(root, f))
                    data = mat_contents['data']
                    right = data['right']
                    right=right[0,0]
                    totals[D][f]['f']= int(round((totals[D][f]['f']*1500)/totals[D]['sum']))#aristero melos: totals[D][f]['per_f']
    json.dump(totals, open("text.txt",'w'))




def read_mat(root_mat,pij, f):
    global totals
    mat_contents=sio.loadmat(root_mat)
    data = mat_contents['data']
    right = data['right']#(40,3)
    right=right[0,0]    
    left = data['left']
    
    if right['gaze'][0,0].shape[0]<totals[pij][f]['f']:
        totals[pij][f]['f']=right['gaze'][0,0].shape[0]
        print("entered here!!!!!!!!!!!!!!!!!!!!!")
    
    indices=np.random.choice(right['gaze'][0,0].shape[0],totals[pij][f]['f'],replace=False)
    right_gaze = right['gaze'][0,0][indices]
    right_img = right['image'][0,0][indices]
    right_pose = right['pose'][0,0][indices]

    return right_gaze, right_img, right_pose



def open_file(rootDir, d):
    global totals
    list_dirs = os.walk(rootDir)
    dataset_img = []
    dataset_gaze = []
    dataset_pose = []
    for root, dirs, files in list_dirs: #gia kathe person
        for f in files: #gia kathe .mat arxeo                
            gaze, img, pose=read_mat(os.path.join(root, f), d, f)
            if dataset_img==[]:
                dataset_img=img
                dataset_gaze=gaze
                dataset_pose=pose
            else:
                dataset_img=np.append(dataset_img,img,axis = 0)
                dataset_gaze=np.append(dataset_gaze,gaze,axis = 0)
                dataset_pose=np.append(dataset_pose,pose,axis = 0)
    return dataset_gaze,dataset_img,dataset_pose


                      
def make_dataset():
    global totals

    dir_path = os.getcwd()#+'#os.getcwd()+data_path
    list_dirs = os.walk(dir_path)
    
    dataset_img = []
    dataset_gaze = []
    dataset_pose = []
    totals=json.load(open("text.txt"))
    # dirs=15 people
    #prepei na paw (1500 left + 1500 right apto kathe outer loop)
    for root, dirs,files in list_dirs:
        for i in range(len(dirs)):
            if dirs[i] == '..Normalized':
                print("deleting ",dirs[i], " at position ",i)
                del dirs[i]
                break 
        for d in dirs: # for each person
            gaze,img,pose=open_file(os.path.join(root,d),d)
            if dataset_img == []:
                dataset_img = img
                dataset_gaze = gaze
                dataset_pose = pose
            else:
                dataset_img = np.append(dataset_img, img, axis=0)
                dataset_gaze = np.append(dataset_gaze, gaze, axis=0)
                dataset_pose = np.append(dataset_pose, pose, axis=0)


    return dataset_gaze, dataset_img, dataset_pose

def randomize(dataset_gaze, dataset_img, dataset_pose):
  
  permutation = np.random.permutation(dataset_pose.shape[0])
  shuffled_gaze = dataset_gaze[permutation,:]
  shuffled_img = dataset_img[permutation,:,:]
  shuffled_pose=dataset_pose[permutation,:]
  return shuffled_gaze, shuffled_img,shuffled_pose


#calcs()
gaze,img,pose=make_dataset()
#gaze,img,pose=randomize(gaze,img,pose)

print(gaze.shape,img.shape,pose.shape)

#pickle_file = 'dataset.pickle'

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






