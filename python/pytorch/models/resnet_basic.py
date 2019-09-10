### the article is here:https://github.com/FrancescoSaverioZuppichini/ResNet ###


#includes
import torch
import torch.nn as nn
from functools import partial

##### ACTIVATION FUNCTIONS #####
# We use ModuleDict to create a dictionary with different activation 
# functions, this will be handy later.If you are unfamiliar with ModuleDict 
# I suggest to read my previous article Pytorch: 
# "how and when to use Module, Sequential, ModuleList and ModuleDict"
def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]



##### Convolution with padding #####
# all we must have a convolution layer and since PyTorch does not 
#have the 'auto' padding in Conv2d, we will have to code ourself!
class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)#can change these parameters with "partial"
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
        
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)#https://pytorch.org/docs/stable/nn.html#conv2d
#conv = conv3x3(in_channels=32, out_channels=64)
#print(conv)
#del conv 


##### RESIDUAL BLOCK #####
# The residual block takes an input with in_channels, applies some blocks 
# of convolutional layers to reduce it to out_channels and sum it up to the 
# original input. If their sizes mismatch, then the input goes into an identity. 
# We can abstract this process and create an interface that can be extended.
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()#identity mapping is 
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)#identity
        x += residual#+identity
        x = self.activate(x)
        return x
   
    #function "should_apply_shortcut" replaced by special object "should_apply_shortcut"
    #that object has functions():https://stackoverflow.com/questions/17330160/how-does-the-property-decorator-work
    #@should_apply_shortcut.setter() metatrepei tin "setter()" se "should_apply_shortcut()"
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

#ResidualBlock(32, 64)

##### class ResNetResidualBlock extends class ResidualBlock #####
# In ResNet, each block has an expansion parameter in order to increase the out_channels 
# if needed. Also, the identity is defined as a Convolution followed by an BatchNorm layer, 
# this is referred to as shortcut. Then, we can just extend ResidualBlock and defined the 
# shortcut function.
# Downsampling has to do with the conv filter stride
class ResNetResidualBlock(ResidualBlock):
    
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        
        super().__init__(in_channels, out_channels, *args, **kwargs)
        
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):#to shortcut uparxei wste na einai iso to input kai output dimensions
        return self.in_channels != self.expanded_channels
#ResNetResidualBlock(32, 64)


##### BASIC BLOCK #####
# A basic ResNet block is composed by two layers of 3x3 conv/batchnorm/relu. 
# In the picture, the lines represent the residual operation. The dotted line means that the shortcut was applied to match the input and the output dimension.   
# Let's first create an handy function to stack one conv and batchnorm layer
def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))

class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


##### ResNetBottleNeckBlock #####
# To increase the network depth while keeping the parameters size as low as possible, the 
# authors defined a BottleNeck block that “The three layers are 1x1, 3x3, and 1x1 convolutions, 
# where the 1×1 layers are responsible for reducing and then increasing (restoring) dimensions, 
# leaving the 3×3 layer a bottleneck with smaller input/output dimensions.” 
# We can extend the ResNetResidualBlock and create these blocks. 
class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(#bottleneck block
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )


##### ResNet Layer #####
# A ResNet’s layer is composed of the same blocks stacked one after the other.
# We can easily define it by just stuck n blocks one after the other, just remember that 
# the first convolution block has a stride of two since "We perform downsampling directly 
# by convolutional layers that have a stride of 2".
class ResNetLayer(nn.Module):#layer = ResNetLayer(64, 128, block=ResNetBasicBlock, n=3)
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

##### ResNet Encoder #####
# Similarly, an Encoder is composed of multiple layers at increasing features size.
class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2], 
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 
                        block=block,*args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

##### Decoder #####
# The decoder is the last piece we need to create the full network. It is a fully 
# connected layer that maps the features learned by the network to their respective classes. 
class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


#### Resnet Network #####
# Final, we can put all the pieces together and create the final model.
class ResNet(nn.Module):
    
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


##### Defining several models #####
# We can now define the five models proposed by the authors, resnet18,34,50,101,152
def resnet18(in_channels, n_classes, block=ResNetBasicBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[2, 2, 2, 2], *args, **kwargs)

def resnet34(in_channels, n_classes, block=ResNetBasicBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 4, 6, 3], *args, **kwargs)

def resnet50(in_channels, n_classes, block=ResNetBottleNeckBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 4, 6, 3], *args, **kwargs)

def resnet101(in_channels, n_classes, block=ResNetBottleNeckBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 4, 23, 3], *args, **kwargs)

def resnet152(in_channels, n_classes, block=ResNetBottleNeckBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 8, 36, 3], *args, **kwargs)

from torchsummary import summary

### diko mas ###
model = resnet18(3, 1000)
summary(model,(3,224,224))#summary(model.cuda(), (3, 224, 224))

### etoimi ulopoihsh ###
#import torchvision.models as models
#summary(models.resnet18(False), (3, 224, 224))