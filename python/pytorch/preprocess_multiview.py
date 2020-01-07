#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import pandas as pd
import scipy.io
import cv2
import json

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

import cv2


def main():
	samples=0
	root_dir = '/home/olympia/MPIIGaze/'
	folders = ['s00-09','s10-19','s20-29','s30-39','s40-49']
	

	# totals={}
	# totals['sum']=0
	# #calc how many from each file
	# for folder in folders:
	# 	subfolder = os.listdir(root_dir+folder)
	# 	for sij in subfolder:
	# 		totals[sij] = {}

	# 		curr_path=root_dir+folder+'/'+sij+'/synth/'
	# 		dirlist=os.listdir(curr_path)
	# 		for name in dirlist:
	# 			if os.path.isdir(os.path.join(curr_path,name)):
	# 				data =pd.read_csv(curr_path+name+'.csv',header=None,usecols=[0,1,2,3,4,5]).values#read only gaze and pose
	# 				totals[sij][name]=int(round(64000*len(data)/230400))
	# 				totals['sum'] = totals['sum']+totals[sij][name]

	# 		print(sij)
	# json.dump(totals, open("save_UT_Multiview.txt",'w'))
	
	#error("ok")
	#totals=json.load(open("save_UT_Multiview.txt"))
	num_img=0
	num_data=0
	for folder in folders:
		subfolder = os.listdir (root_dir+folder)
		for sij in subfolder:
			curr_path = root_dir+folder+'/'+sij+'/synth/'
			dirlist = os.listdir (curr_path)
			images = []
			poses = []
			gazes = []
			for name in dirlist:
				if os.path.isdir(os.path.join(curr_path,name)):
					data =pd.read_csv(curr_path+name+'.csv',header=None,usecols=[0,1,2,3,4,5]).values#read only gaze and pose
					img_list=os.listdir(curr_path+name)
					img_list.sort()#print(type(df.values))
					indices=np.random.choice(144,4,replace=False)#5 out of 144 eye imgs
					for i in indices:#144 images
						img = img_list[i]
						num_img=num_img+1

						image = cv2.imread(curr_path+name+"/"+img, 0)
						images.append(image)
					print(len(images))
					
					for i in indices:#range(len(data)):
						num_data=num_data+1
						if 'r' in name: #right eye
							gaze=convert_gaze(data[i,0:3])*np.array([-1,1])
							pose=convert_pose(data[i,3:6])*np.array([-1,1])

						else:#left eye
							gaze = convert_gaze(data[i,0:3])
							pose = convert_pose(data[i,3:6])
							
						poses.append(pose)
						gazes.append(gaze)
						print(len(gazes))	
								
						#pose = convert_pose(right_poses[day][index]) * np.array([-1, 1])
						#gaze = convert_gaze(right_gazes[day][index]) * np.array([-1, 1])
				


						#for i in range(img.shape[0]): #traverses through height of the image
						#	for j in range (img.shape[1]): #traverses through width of the image
						#		print(img[i][j])
			images = np.array(images).astype(np.float32) / 255
			poses = np.array(poses).astype(np.float32)
			gazes = np.array(gazes).astype(np.float32)

			outpath = os.path.join("data_UT_Multiview", sij)
			np.savez(outpath, image=images, pose=poses, gaze=gazes)
			samples=samples+len(gazes)#len(images)
			print("till now:", samples, " samples!")
	print("total samples:",samples)
	print("images:",num_img)
	print("data:",num_data)

				



if __name__ == '__main__':
    main()
