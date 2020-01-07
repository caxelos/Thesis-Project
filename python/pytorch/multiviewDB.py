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
	#print("new center!!!!!")
	return True

####
def find_nearest_group(theta,phi,groups_centers):
	minDst = 100;
	maxDist=0.5
	nearestGrp = -1;   
	for i in range(len(groups_centers)):
		if abs(groups_centers[i,0]-theta) < maxDist and abs(groups_centers[i,1]-phi) < maxDist: 
			dist= abs(groups_centers[i,0]-theta)+abs(groups_centers[i,1]-phi)
			if dist < minDst:
				minDst=dist
				nearestGrp=i

	if nearestGrp==-1:
		print("probleeeeeeeeeeemmmmmmmmmmmmmmmmmmmmm")
		print("len:",)
		print("minDst:",minDst)
	return nearestGrp

def find_R_nearest_groups(centerTheta, centerPhi, groups_centers, R, first, NUM_OF_GROUPS):
	listOfGroupIds=[-1 for i in range(R+1)]
	listOfGroupIds[0] = first+1
	minDist=[]#zeros(R+1)
	for i in range(R+1):
		minDist.append(7+i)

	for i in range(NUM_OF_GROUPS):
		if i != first:#fi+1 not in  listOfGroupIds:
  			dist = abs(groups_centers[i][0]-centerTheta) + abs(groups_centers[i][1]-centerPhi)
  			if dist < minDist[R]:
  				for o in range(1,R+1):
  					if dist < minDist[o]:
  						if o == R:
  							listOfGroupIds[o]=i+1
  							minDist[o]=dist
  						else:
  							j=R-1
  							while j >= o:
  								listOfGroupIds[j+1]=listOfGroupIds[j]
  								minDist[j+1]=minDist[j]
  								j=j-1
  							listOfGroupIds[o]=i+1
  							minDist[o]=dist
  						break
	#print("final list:",listOfGroupIds)
	return np.array(listOfGroupIds)

import h5py
#initialize data
groups_poses={}
groups_gazes={}
groups_images={}
groups_nearests={}

numGrps=0;groups_centers=[]
subject_ids = ['p{:02}'.format(index) for index in range(15)]
dataset_dir='/home/olympia/MPIIGaze/python/pytorch/data/'
curr_dist = 0.05#[0.03 0.04 0.05 0.06 0.07];%evgala to 0.06

multiview_ids=['s{:02}'.format(index) for index in range(50)]
multiview_dir='/home/olympia/MPIIGaze/python/pytorch/data_UT_Multiview/'

#my_list = ['p00','p01','p02','p03','p04','p05','p06','p07','p08','p09','p10','p11','p12','p14']

for subject_id in multiview_ids:#my_list:#subject_ids[1:]:#(0,5,10,15)
	path = os.path.join(multiview_dir, '{}.npz'.format(subject_id ))
	#/home/olympia/MPIIGaze/python/pytorch/data/p12.npz
	
	with np.load(path) as fin:
		images = fin['image']#[0:2400]
		poses = fin['pose']#[0:2400]
		gazes = fin['gaze']#[0:2400]
		length = len(images)

	for i in range(len(poses[:,1])):
		#can_be_center(groups,theta,phi,numGrps,curr_dist):

		answer = True

		##### can_be_center_or_not ###
		for j in range(numGrps):
			if  np.sqrt( (groups_centers[j][0]-poses[i][0])**2 + (groups_centers[j][1]- poses[i][1])**2 ) < curr_dist:
				#if  np.sqrt( (groups[i][0]-theta)<<1 + (groups[i][1]- phi)<<1 ) < curr_dist:
				answer = False
		if answer==True:	
			groups_centers.append(poses[i,:])#([poses[i,0],poses[i,1]])


			# groups_poses[numGrps] = []#struct(gazes,poses,data)
			# groups_gazes[numGrps] = []
			# groups_images[numGrps] = []
			# groups_nearests[numGrps]=[]	
			# groups_poses[numGrps].append(poses[i,:])
			# groups_gazes[numGrps].append(gazes[i,:])
			# groups_images[numGrps].append(images[i,0,13:22,22:37])
			# groups_nearests[numGrps]=[numGrps]#isws exei thema edw.Vale "-1" an xreiastei
			# groups_poses[nearestGrp].append(poses[i,:])
			# groups_gazes[nearestGrp].append(gazes[i,:])
			# groups_images[nearestGrp].append(images[i,0,13:22,22:37])
			numGrps=numGrps+1
			#print("new center!!!!!")

#print("NumOfCenters:",numGrps)
print("numGrps:",numGrps)
#gia ta regression forests,prepei na kanoume reshape ta data.Opote,anti gia 60x36,
#to resolution einai 16x9
groups_centers=np.array(groups_centers)
w=0
for i in range(numGrps):
	groups_poses[i] = []#struct(gazes,poses,data)
	groups_gazes[i] = []
	groups_images[i] = []
	groups_nearests[i]=[]	

for subject_id in multiview_ids: #my_list:# subject_ids[1:]:#my_list
	path = os.path.join(multiview_dir, '{}.npz'.format(subject_id ))
	#/home/olympia/MPIIGaze/python/pytorch/data/p12.npz
	with np.load(path) as fin:
		images= np.empty((1600, 1, 36, 60))
		images[:,0,:,:] = fin['image']*255
		#images[:,0,0:9,0:16]=images[:,0,13:22,22:38]#images[:,22:37,13:22]
		poses = fin['pose']#[0:2400]
		gazes = fin['gaze']#[0:2400]
		#length = len(images)

	#img = temp.data.left.image(num_i, 14:22, 23:37);
    #img = reshape(img, HEIGHT ,WIDTH);

	### for every training sample ###
	for i in range(len(poses[:,1])):
		### find_nearest_group() ###
		minDst=1000
		maxDist=0.3;
		
		for j in range(numGrps):

			if abs(groups_centers[j,0]-poses[i,0]) < maxDist and abs(groups_centers[j,1]-poses[i,1]) < maxDist: 
				dist= abs(groups_centers[j,0]-poses[i,0])+abs(groups_centers[j,1]-poses[i,1]);
				if dist < minDst:
					minDst=dist
					nearestGrp=j
		if minDst==0:
			w=w+1
			#print("grps:",numGrps,"popa:",w)

		
		#print("Added to Group:",nearestGrp)
		groups_poses[nearestGrp].append(poses[i,:])
		groups_gazes[nearestGrp].append(gazes[i,:])
		groups_images[nearestGrp].append(images[i,0,13:22,22:37])#images[:,0,13:22,22:38]
		#groups_nearests[i].append(j)						
#from PIL import Image
#im = Image.fromarray(images[10,0,13:22,22:37])#np.flip(images[10]))
#im.show()

RADIUS=30

with h5py.File('big_train_multiview_80000.h5','w') as hdf:#mall_train_dataset_loocv1.h5
	for i in range(numGrps):
	    g=hdf.create_group('g'+str(i+1))
	    g.create_dataset('gaze',data=groups_gazes[i],dtype='f8')
	    g.create_dataset('headpose',data=groups_poses[i],dtype='f8')
	    images_final= np.empty((len(groups_images[i]), 1, 9, 15))#(images_final.shape)=(53, 1, 9, 16)
	    groups_images[i]=np.array(groups_images[i])#print('i=',i,groups_images[i].shape)#problima an den uparxoun arketa deigmata
	    images_final[:,0,:,:]=groups_images[i][:,:,:]

	    images_final.astype('uint8') 
	    g.create_dataset('data',data=images_final,dtype='uint8')
	    g.create_dataset('center',data=groups_centers[i].transpose(),dtype='f8')
	    g.create_dataset('samples',data=len(groups_gazes[i]),dtype='uint32')
	    listOfGroupIds=find_R_nearest_groups(centerTheta=groups_centers[i][0],
																 centerPhi=groups_centers[i][1],
																 groups_centers=groups_centers,
																 R=RADIUS,
																 first=i,
																 NUM_OF_GROUPS=len(groups_centers))
	    g.create_dataset('nearestIDs',data=listOfGroupIds,dtype='uint32')
	    groups_nearests[i].append(listOfGroupIds)


###### TEST DATA #####small_train_dataset_loocv1.h5
with h5py.File('big_test_mpiigaze_45000.h5','w') as hdf:
	gaze_dset=hdf.create_dataset('gaze',(0,2),maxshape=(None,2),dtype='f8')
	pose_dset=hdf.create_dataset('headpose',(0,2),maxshape=(None,2),dtype='f8')
	image_dset=hdf.create_dataset('data',(0,1,9,15),maxshape=(None,1,9,15),dtype='uint8')
	nearests_dset = hdf.create_dataset('nearestIDs',(0,RADIUS+1),maxshape=(None,RADIUS+1),dtype='uint32')
			
	for subject_id in subject_ids:#[13:14]:
		path = os.path.join(dataset_dir, '{}.npz'.format(subject_id ))
		#/home/olympia/MPIIGaze/python/pytorch/data/p12.npz
		with np.load(path) as fin:#dynamic datasets:https://stackoverflow.com/questions/25655588/incremental-writes-to-hdf5-with-h5py?rq=1
			images= np.empty((len(fin['image']), 1,9,15))
			images[:,0,:,:] = fin['image'][:,13:22,22:37]*255			
			poses = fin['pose']
			gazes = fin['gaze']


			#gaze_dset=hdf.create_dataset('gaze',data=gazes,dtype='f8')
			gaze_dset.resize(gaze_dset.shape[0]+len(gazes), axis=0)
			gaze_dset[-len(gazes):]=gazes 

			#pose_dset=hdf.create_dataset('headpose',data=poses,dtype='f8')
			pose_dset.resize(pose_dset.shape[0]+len(poses), axis=0)
			pose_dset[-len(poses):]=poses 
			
			#image_dset=hdf.create_dataset('data',data=images,dtype='uint8')
			images.astype('uint8')
			image_dset.resize(image_dset.shape[0]+len(images), axis=0)
			image_dset[-len(images):]=images#fin['image'] 

			nearests=np.zeros((len(poses[:,0]),RADIUS+1))
			for i in range(len(poses[:,0])):
				#print(i)
				grp=find_nearest_group(poses[i,0],poses[i,1],groups_centers)
				print(grp)
				print(np.array(groups_nearests[grp]))
				nearests[i,:]=np.array(groups_nearests[grp])
			#nearests=np.array(nearests)
			#nearests_dset = hdf.create_dataset('nearestIDs',data=nearests,dtype='uint32')
			nearests_dset.resize(nearests_dset.shape[0]+len(images), axis=0)
			nearests_dset[-len(images):]=nearests#fin['image'] 
	

# with h5py.File('test_dataset.h5','w') as hdf:
#     hdf.create_dataset('gazes',data=test_gaze)
#     hdf.create_dataset('poses',data=test_pose)
#     hdf.create_dataset('label',data=test_img)


		#type=tensor
		#images = torch.unsqueeze(torch.from_numpy(images), 1)
		#poses = torch.from_numpy(poses)
		#gazes = torch.from_numpy(gazes)
		#print(type(gazes))

