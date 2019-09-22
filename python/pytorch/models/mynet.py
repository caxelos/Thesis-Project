import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

'''
STEP 1: CREATE MODEL CLASS
'''
class Model(nn.Module):
    #1)in_channels=1:1=gia grayscale
    #2)out_channels=16:16=arithmos twn kernels pou dialegoume.
    #Kathe 1 kernel pernaei to image apo 1 fora kai paragei 1 feature map.
    #Ara exoume 16 feature maps
    
    def __init__(self):
        super(Model, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # compute conv feature size
        input_shape = (1, 1, 36, 60)
        with torch.no_grad():
            self.feature_size = self._forward_conv(
                torch.zeros(*input_shape)).view(-1).size(0)

        self.fc1 = nn.Linear(self.feature_size + 2, 2)

        # Fully connected 1 (readout)
        #self.fc1 = nn.Linear(32 * 4 * 4, 10)#https://pytorch.org/docs/stable/nn.html#torch.nn.Linear 
        #self.fc1 = nn.Linear(32 , 2306)
        #[32 x 2306]
    def forward(self, x,y):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        
        # Max pool 1
        out = self.maxpool1(out)
        
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        
        # Max pool 2 
        out = self.maxpool2(out)
        
        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        print(out.size(3))
        out = out.view(out.size(0), -1)

        out = torch.cat([out, y], dim=1)
        # Linear function (readout)
        out = self.fc1(out)
        
        return out
print(())