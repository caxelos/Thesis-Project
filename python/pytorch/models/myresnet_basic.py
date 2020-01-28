### the article is here:https://github.com/FrancescoSaverioZuppichini/ResNet ###
### RUN MODEL WITH:
#python -u main_resnet18.py --arch myresnet_basic --dataset data_UT_Multiview --testset data  --outdir results/resnet_preact/00 --batch_size 32 --base_lr 0.0001 --momentum 0.9 --nesterov True --weight_decay 1e-4 --epochs 40 --milestones '[30, 35]' --lr_decay 0.1 --tensorboard --gate_kernel 3 --ff1_out 512 --ff2_out 512 --block_type bottle --block_sizes 32 64 128 256    --deepths 2 2 2 1  



#dokimase na to guriseis apo regression se klassification
#kane kai regression episis to sfalma provlepsis(kalman filter)

#includes
import torch
import torch.nn as nn
from functools import partial



#### Resnet Network #####
# Final, we can put all the pieces together and create the final model.
class ResNet6(nn.Module):
    # ResNet(in_channels, n_classes, block=block, deepths=[2, 2, 2, 2], *args, **kwargs)
    def __init__(self, in_channels=1, n_outputs=2, numOfFC=3,ff1_out=1024,ff2_out=1024, *args, **kwargs):#1


        #print("Resnet created!")
        #args: () 
        #kwargs:,'block': <class '__main__.ResNetBasicBlock'>,'deepths': [2, 2, 2, 2]
        #input_shape = (1, 1, 36, 60)
        # compute conv feature size
        #with torch.no_grad():
        #    self.feature_size = self._forward_conv(
        #        torch.zeros(*input_shape)).view(-1).size(0)


        super(ResNet6, self).__init__()

        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        #self.decoder = ResnetDecoder(in, n_outputs)


        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, numOfFC, ff1_out, ff2_out, n_outputs)
        #.expanded_channels=512.kai decoder=ResnetDecoder(features=512,classes=12)
        

        print(self.encoder,self.decoder)
    def forward(self, image,pose):
        x = self.encoder(image)
        x = self.decoder(x,pose)

        #print("x",x)
        return x


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
    #6#9
    def __init__(self, in_channels, out_channels, activation='relu', resnet_type='basic'):
        #print("                 ResidualBlock created!")
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()#identity mapping is 
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()
        

    def forward(self, x):
        ## Basic Resnet ###
        if self.resnet_type == 'basic':
            residual = x
            if self.should_apply_shortcut: residual = self.shortcut(x)
            x = self.blocks(x)#identity
            x += residual#+identity
            x = self.activate(x)
        elif self.resnet_type == 'preact':
            ## Preact Resnet ###
            residual = x
            if self.should_apply_shortcut: residual = self.shortcut(x)
            x = self.blocks(x)#identity
            x += residual#+identity
        elif self.resnet_type == 'relu_b_add':           
        # ### ReLU before addition ###
            residual = x
            if self.should_apply_shortcut: residual = self.shortcut(x)
            x = self.blocks(x)#identity
            x += residual#+identity

        #print("residual:",residual.shape):residual: torch.Size([32, 128, 9, 15])
        #print("after convs:",x.shape):after convs: torch.Size([32, 128, 5, 8])

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
    #5#8
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        #print("             ResNetResidualBlock created!")
        super().__init__(in_channels, out_channels, *args, **kwargs)
        
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        
        #expanded_channels={64,128,256,512}
        #activate): ReLU(inplace)
        #(shortcut): Sequential(
        #(0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        #(1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        #print("             expanded channels:",self.out_channels * self.expansion)
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
    #block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    #4#7
    expansion = 1
    def __init__(self,in_channels, out_channels, *args, **kwargs):
        #print("         ResNetBasicBlock created!")
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.resnet_type=kwargs['resnet_type']
        if self.resnet_type == 'basic':
            self.blocks = nn.Sequential(
                ### Basic Resnet ###
                conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
                activation_func(self.activation),
                conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
            )
        elif self.resnet_type == 'preact':
            ### Preact Resnet ###
            self.blocks = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                activation_func(self.activation),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.downsampling, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                activation_func(self.activation),            
                nn.Conv2d(out_channels,self.expanded_channels,kernel_size=3,stride=1,padding=1,bias=False)
            )
        elif self.resnet_type == 'relu_b_add':    
            ### ReLU before addition ###
            self.blocks = nn.Sequential(
                activation_func(self.activation),
                conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
                activation_func(self.activation),
                conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False)
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
        #print("         ResNetBottleNeckBlock created!")#print("ResNetBottleNeckBlock created!")
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
class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """
    #3ResNetBottleNeckBlock
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        #print("     ResNetLayer created!")
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )


    def forward(self, x):
        #print("     ResNetLayer runs!")
        x = self.blocks(x)
        #print("   after block:", x.shape)
        return x

##### ResNet Encoder #####
# Similarly, an Encoder is composed of multiple layers at increasing features size.
class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.

    """

    #blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2]

    #[32,64] kai [3,3] -> 258
    #
    def __init__(self,in_channels=1, blocks_sizes=[64,128], deepths=[2,2], 
                 activation='relu', block=ResNetBasicBlock, gate_kernel=3, *args, **kwargs):
        super().__init__()
        #print(" ResNetEncoder created!")
        if block=='basic':
            block=ResNetBasicBlock
        else:
            block=ResNetBottleNeckBlock


       
        self.gate_kernel=gate_kernel
        self.blocks_sizes = blocks_sizes # = [64, 128, 256, 512] = out_channels
        
        self.gate = nn.Sequential(#great info here:https://www.reddit.com/r/MachineLearning/comments/6fsqww/d_why_does_resnet_have_a_77_convolution_in_the/
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=self.gate_kernel, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        #sizes: [(64, 128), (128, 256), (256, 512)]

        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 
                        block=block,*args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
        #self.blocks = nn.ModuleList(
        #    gate:[ResNetLayer(64,64,n=2,relu,block),
        #    #list begins...
        #    ResNetLayer(64,128, n=2, activation=activation,block=block, *args, **kwargs), 
        #    ResNetLayer(128,256, n=2, activation=activation,block=block, *args, **kwargs), 
        #    ResNetLayer(256,512, n=2, activation=activation,block=block, *args, **kwargs)])
        
    def forward(self, x):#3->64->128->256->512
        #print("before gate:",x)
        #torch.Size([32, 1, 36, 60])
        #print("\ninit image:", x.shape) 
        x = self.gate(x)#torch.Size([32, 64, 9, 15])
        #print("after gate:", x.shape)
        #print("x before flatten:",x.size())
        for block in self.blocks:#iterate module list
            #print(" ResNetEncoder runs!")
            x = block(x)
            #print("after layer:", x.shape)
        return x

##### Decoder #####
# The decoder is the last piece we need to create the full network. It is a fully 
# connected layer that maps the features learned by the network to their respective classes. 
class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, numOfFC, ff1_out, ff2_out, n_outputs):
        #print("ResNetDecoder created!")
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        #in_features=512
        self.numOfFC=numOfFC#ff1_out
        if self.numOfFC == 3:
            self.ff1_out = ff1_out
            self.ff2_out = ff2_out
            self.decoder1 = nn.Linear(in_features, self.ff1_out)
            self.decoder2 = nn.Linear(self.ff1_out+2, self.ff2_out)
            self.decoderFinal = nn.Linear(self.ff2_out, n_outputs)
        elif self.numOfFC==2:
            self.ff1_out=ff1_out
            self.decoder1 = nn.Linear(in_features+2,self.ff1_out)
            self.decoderFinal = nn.Linear(self.ff1_out, n_outputs)
        else:
            print("in_features:",in_features)
            self.decoderFinal = nn.Linear(in_features+2, n_outputs)

    def forward(self, x,pose):
        
        x = self.avg(x)#512
        x = x.view(x.size(0), -1)#flatten
        if self.numOfFC == 3:
            x = self.decoder1(x)
            x = torch.cat([x, pose], dim=1)
            x = self.decoder2(x)
            x = self.decoderFinal(x)
        elif self.numOfFC == 2:
            x = torch.cat([x,pose],dim=1)
            x = self.decoder1(x)
            x = self.decoderFinal(x)
        else:
            x = torch.cat([x,pose],dim=1)
            x = self.decoderFinal(x)
        return x

#init image: torch.Size([32, 1, 36, 60])
#after gate: torch.Size([32, 64, 18, 30])
#after layer: torch.Size([32, 64, 18, 30])
#after layer: torch.Size([32, 128, 9, 15])
#before avg pool: torch.Size([32, 128, 9, 15])
#before concat(shape): torch.Size([32, 128, 1, 1])
#final gaze: torch.Size([32, 2])




##### Defining several models #####
# We can now define the five models proposed by the authors, resnet18,34,50,101,152
#def resnet18(in_channels, n_outputs, block=ResNetBasicBlock, *args, **kwargs):
#    #args=none,kwargs=none
#    return Model(in_channels, n_outputs, block=block, deepths=[2, 2, 2, 2], *args, **kwargs)

#def resnet34(in_channels, n_outputs, block=ResNetBasicBlock, *args, **kwargs):
#   return Model(in_channels, n_outputs, block=block, deepths=[3, 4, 6, 3], *args, **kwargs)

#def resnet50(in_channels, n_outputs, block=ResNetBottleNeckBlock, *args, **kwargs):
#    return Model(in_channels, n_outputs, block=block, deepths=[3, 4, 6, 3], *args, **kwargs)

#def resnet101(in_channels, n_outputs, block=ResNetBottleNeckBlock, *args, **kwargs):
#    return Model(in_channels, n_outputs, block=block, deepths=[3, 4, 23, 3], *args, **kwargs)

#def resnet152(in_channels, n_outputs, block=ResNetBottleNeckBlock, *args, **kwargs):
#    return Model(in_channels, n_outputs, block=block, deepths=[3, 8, 36, 3], *args, **kwargs)

#from torchsummary import summary

### diko mas ###
#model = resnet18(in_channels=1, n_outputs=2,block=ResNetBasicBlock)
#summary(model,(3,224,224))#summary(model.cuda(), (3, 224, 224))
#print(model)
### etoimi ulopoihsh ###
#import torchvision.models as models
#summary(models.resnet18(False), (3, 224, 224))