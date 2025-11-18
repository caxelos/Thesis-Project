### RUN MODEL WITH:
#python -u main.py --arch zhang --dataset data --test_id 0 --outdir results/resnet_preact/00 --batch_size 32 --base_lr 0.01 --momentum 0.9 --nesterov True --weight_decay 0.0005 --epochs 40 --milestones '[30, 35]' --lr_decay 0.1


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

#python -u main.py --arch zhang --dataset data --test_id 0 --outdir results/resnet_preact/00 --batch_size 32 --base_lr 0.1 --momentum 0.9 --nesterov True --weight_decay 1e-4 --epochs 40 --milestones '[30, 35]' --lr_decay 0.1


'''
STEP 1: CREATE MODEL CLASS
'''
class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
        
		# Convolution 1
		#O=(60-5+2*2)/1 + 1=60 kai 36:(1,60,36) 
		self.cnn1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0,bias=True, padding_mode='zeros')

		#edw prepei:(20,58,34)

		# Max Pooling 1
		#O=60/5=(12,7)
		self.maxpool1 = nn.MaxPool2d(kernel_size=2,stride=2)

		#edw prepei:(20,29,17)

		# Convolution 2
		self.cnn2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0,bias=True, padding_mode='zeros')
	
		#edw prepei:(50,27,15)

		# Max Pooling 2
		self.maxpool2 = nn.MaxPool2d(kernel_size=2,stride=2)      

		#edw prepei:(50,14,8)
		#enw einai:#(32,50,6,12)

		# compute conv feature size
		input_shape = (1, 1, 36, 60)#m1: [32 x 3600], m2: [115202 x 500]
										#[32 x 3600], m2: [3602 x 500]
		with torch.no_grad():#https://stackoverflow.com/questions/53784998/how-are-the-pytorch-dimensions-for-linear-layers-calculated
			self.feature_size = self._forward_conv(
           torch.zeros(*input_shape)).view(-1).size(0)
			#print(self.feature_size)

		# print("numOfFeatures:",self.feature_size)

		# Fully connected-500
		self.fc500 = nn.Linear(self.feature_size, 500)
		#self.fc500 = nn.Linear(50*14*8, 500)

		#Relu function
		self.relu = nn.ReLU()

		# Fully connected-2
		self.fc2 = nn.Linear(502, 2)

	def _forward_conv(self, x):#class Model
		#print("x0:(",x.size(0),",",x.size(1),",",x.size(2),",",x.size(3))

    	# Convolution 1
		x = self.cnn1(x)
		#print("x1:(",x.size(0),",",x.size(1),",",x.size(2),",",x.size(3))


		# Max pool 1
		x = self.maxpool1(x)
		#print("x2:(",x.size(0),",",x.size(1),",",x.size(2),",",x.size(3))

		# Convolution 2
		x = self.cnn2(x)
		#print("x3:(",x.size(0),",",x.size(1),",",x.size(2),",",x.size(3))

		# Max pool 2
		x = self.maxpool2(x)
		#print("x4:(",x.size(0),",",x.size(1),",",x.size(2),",",x.size(3))


		return x

	def forward(self, x, y):
		#kati paizei edw:
		#https://stackoverflow.com/questions/44357055/pytorch-linear-layer-input-dimension-mismatch
		#https://groups.google.com/forum/#!topic/torch7/6zU89K1YEOs


		# Convolution part
		out = self._forward_conv(x)#1
		#if (out != out).any():
		#	error('prwto!!!!!!')
		#print("feature_size:",self.feature_size)
		#print("popa:",x.size(0))#(32,50,6,12)3602 x 500

		#prin:out_prin:( 32 , 50 , 6 , 12)
		#out = out.view(out.size(0), -1)#why??! #(32,)
		out = out.view(-1, 3600)#why??! #(32,)
		#if (out != out).any():
		#	error('deftero!!!!!!')
		#if torch.isnan(out)==True:
		#	print("poulo")
		#	return
		#meta:out_meta:( 32 , 3600)

		#print("out_meta:(",out.size(0),",",out.size(1))

		######## Prosoxi #############


		######## Telos Prosoxis ######


		# Fully connected-500
		out = self.fc500(out)
		#if (out != out).any():
		#	error('trito!!!!!!')

		# Relu function
		out = self.relu(out)
		#if (out != out).any():
		#	error('tetarto!!!!!!')
		# Embedd the pose
		#out = out.view(out.size(0), -1)#why??

		out = torch.cat([out, y], dim=1)
		#if (out != out).any():
		#	error('pempto!!!!!!')
		# Fully connected-2
		out = self.fc2(out)
		#if (out != out).any():
		#	error('ekto!!!!!!')

		return out