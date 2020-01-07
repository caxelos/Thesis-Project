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


def main():
	root_dir = '/home/olympia/MPIIGaze/'
	folders = ['s00-09','s10-19','s20-29','s30-39','s40-49']
	for folder in folders:
		subfolder = os.listdir (root_dir+folder)
		for sij in subfolder:
			curr_path = root_dir+folder+'/'+sij+'/synth/'
			dirlist = os.listdir (curr_path)
			for name in dirlist:
				if os.path.isdir(os.path.join(curr_path,name)):
					df =pd.read_csv(curr_path+name+'.csv',usecols=[0,1,2,3,4,5])#read only gaze and pose
					img_list=os.listdir(curr_path+name)
					img_list.sort()
					print(type(df.values))


				



if __name__ == '__main__':
    main()
