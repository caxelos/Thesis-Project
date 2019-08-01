#create dataset to regression forests
# coding: utf-8
import os
import numpy as np

import torch
import torch.utils.data

#groups=[theta,phi]
def can_be_center(groups,theta,phi,numGrps,curr_dist):
	for i in range(numGrps):
		if  np.sqrt( (groups[i][0]-theta)**2 + (groups[i][1]- phi)**2 ) < curr_dist:
		#if  np.sqrt( (groups[i][0]-theta)<<1 + (groups[i][1]- phi)<<1 ) < curr_dist:
			return False
	print("new center!!!!!")
	return True

def find_R_nearest_groups(centerTheta, centerPhi, groups, R, first, NUM_OF_GROUPS):
	pass

numGrps=-1;groups_centers=[]
subject_ids = ['p{:02}'.format(index) for index in range(15)]
dataset_dir='/home/olympia/MPIIGaze/python/pytorch/data/'
curr_dist = 0.06#[0.03 0.04 0.05 0.06 0.07];%evgala to 0.06

for subject_id in subject_ids[0:2]:
	path = os.path.join(dataset_dir, '{}.npz'.format(subject_id ))
	#/home/olympia/MPIIGaze/python/pytorch/data/p12.npz
	with np.load(path) as fin:
		images = fin['image']
		poses = fin['pose']
		gazes = fin['gaze']
		length = len(images)

	for i in range(len(poses[:,1])):
		#can_be_center(groups,theta,phi,numGrps,curr_dist):
		print(i)
		answer = True

		##### can_be_center_or_not ###
		for j in range(numGrps):
			if  np.sqrt( (groups_centers[j][0]-poses[i][0])**2 + (groups_centers[j][1]- poses[i][1])**2 ) < curr_dist:
				#if  np.sqrt( (groups[i][0]-theta)<<1 + (groups[i][1]- phi)<<1 ) < curr_dist:
				answer = False
		if answer==True:
			numGrps=numGrps+1
			groups_centers.append(poses[i,:])#([poses[i,0],poses[i,1]])
			print("new center!!!!!")

print("NumOfCenters:",numGrps)


#gia ta regression forests,prepei na kanoume reshape ta data.Opote,anti gia 60x36,
#to resolution einai 16x9

#initialize data
groups_poses={}
groups_gazes={}
groups_images={}
groups_nearests={}
for i in range(numGrps):
	groups_poses[i] = []#struct(gazes,poses,data)
	groups_gazes[i] = []
	groups_images[i] = []
	groups_nearests[i]=[]	

for subject_id in subject_ids[0:2]:
	path = os.path.join(dataset_dir, '{}.npz'.format(subject_id ))
	#/home/olympia/MPIIGaze/python/pytorch/data/p12.npz
	with np.load(path) as fin:
		images= np.empty((len(fin['image']), 1, 36, 60))
		images[:,0,:,:] = fin['image']*255
		#images[:,0,0:9,0:16]=images[:,0,13:22,22:38]#images[:,22:37,13:22]
		poses = fin['pose']
		gazes = fin['gaze']
		length = len(images)

	#img = temp.data.left.image(num_i, 14:22, 23:37);
    #img = reshape(img, HEIGHT ,WIDTH);

	### for every training sample ###
	for i in range(len(poses[:,1])):
		### find_nearest_group() ###
		minDst=1000
		maxDist=0.3;
		
		for j in range(numGrps):
			if abs(groups_centers[j][0]-poses[i,0]) < maxDist and abs(groups_centers[j][1]-poses[i,1]) < maxDist: 
				dist= abs(groups_centers[j][0]-poses[i,0])+abs(groups_centers[j][1]-poses[i,1]);
				if dist < minDst:
					minDst=dist
					nearestGrp=j


		print("Added to Group:",nearestGrp)
		groups_poses[nearestGrp].append(poses[i,:])
		groups_gazes[nearestGrp].append(gazes[i,:])
		groups_images[nearestGrp].append(images[i,0,13:22,22:37])#images[:,0,13:22,22:38]
		#groups_nearests[i].append(j)						
from PIL import Image
im = Image.fromarray(images[10,0,13:22,22:37])#np.flip(images[10]))
im.show()
#images=images[:,0,0:9,0:16]

import h5py
with h5py.File('train_dataset.h5','w') as hdf:
	for i in range(numGrps):
	    g=hdf.create_group('g'+str(i))
	    g.create_dataset('gaze',data=groups_gazes[i])
	    g.create_dataset('headpose',data=groups_poses[i])
	    images_final= np.empty((len(groups_images[i]), 1, 9, 15))
	    #(images_final.shape)=(53, 1, 9, 16)
	    groups_images[i]=np.array(groups_images[i])	    
	    images_final[:,0,:,:]=groups_images[i][:,:,:]

	    
	    g.create_dataset('data',data=images_final,dtype='u8')
	    g.create_dataset('center',data=groups_centers[i])
	    g.create_dataset('samples',data=len(groups_gazes[i]))

# with h5py.File('test_dataset.h5','w') as hdf:
#     hdf.create_dataset('gazes',data=test_gaze)
#     hdf.create_dataset('poses',data=test_pose)
#     hdf.create_dataset('label',data=test_img)


		#type=tensor
		#images = torch.unsqueeze(torch.from_numpy(images), 1)
		#poses = torch.from_numpy(poses)
		#gazes = torch.from_numpy(gazes)
		#print(type(gazes))

