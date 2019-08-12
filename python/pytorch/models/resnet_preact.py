### RUN MODEL WITH:
#python -u main.py --arch resnet_preact --dataset data --test_id 0 --outdir results/resnet_preact/00 --batch_size 32 --base_lr 0.1 --momentum 0.9 --nesterov True --weight_decay 1e-4 --epochs 40 --milestones '[30, 35]' --lr_decay 0.1

# coding: utf-8
#resnet official:https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F#https://pytorch.org/docs/master/nn.functional.html


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.constant_(module.bias, 0)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)#(bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)#(conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)#(bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)#(conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        self.shortcut = nn.Sequential()#(shortcut): Sequential()
        if in_channels != out_channels:#
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False))

    def forward(self, x):#3 fores ana train
        #print("kai meta edw")
        #print("x1:(",x.size(0),",",x.size(1),",",x.size(2),",",x.size(3))
        x = F.relu(self.bn1(x), inplace=True)
        y = self.conv1(x)
        #print("y1:(",y.size(0),",",y.size(1),",",y.size(2),",",y.size(3))
        y = F.relu(self.bn2(y), inplace=True)
        #print("y2:(",y.size(0),",",y.size(1),",",y.size(2),",",y.size(3))
        y = self.conv2(y)
        #print("y3:(",y.size(0),",",y.size(1),",",y.size(2),",",y.size(3))
        y += self.shortcut(x)
        #print("y4:(",y.size(0),",",y.size(1),",",y.size(2),",",y.size(3))
        return y


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()#https://www.artima.com/weblogs/viewpost.jsp?thread=236275

        depth = 8
        base_channels = 16
        input_shape = (1, 1, 36, 60)

        n_blocks_per_stage = (depth - 2) // 6
        assert n_blocks_per_stage * 6 + 2 == depth

        n_channels = [base_channels, base_channels * 2, base_channels * 4]

        #in_channels=1, out_channels=16
        self.conv = nn.Conv2d(
            input_shape[1],#1
            n_channels[0],#16
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)

        # print("basicBlock:",BasicBlock)
        # print("n_blocks_per_stage:",n_blocks_per_stage)
        # print("n_channels[1]:",n_channels[1])
        # print("n_channels[2]:",n_channels[2])
        self.stage1 = self._make_stage(
            n_channels[0],#16
            n_channels[0],#16
            n_blocks_per_stage,
            BasicBlock,
            stride=1)
        self.stage2 = self._make_stage(
            n_channels[0],#16
            n_channels[1],#32
            n_blocks_per_stage,
            BasicBlock,
            stride=2)
        self.stage3 = self._make_stage(
            n_channels[1],#32
            n_channels[2],#64
            n_blocks_per_stage,
            BasicBlock,
            stride=2)
        self.bn = nn.BatchNorm2d(n_channels[2])

        # compute conv feature size
        with torch.no_grad():
            self.feature_size = self._forward_conv(
                torch.zeros(*input_shape)).view(-1).size(0)
            #print('code:')#,self._forward_conv(torch.zeros(*input_shape)).view(-1))


        #(fc): Linear(in_features=66, out_features=2, bias=True)
        self.fc = nn.Linear(self.feature_size + 2, 2)

        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):

        stage = nn.Sequential()#https://pytorch.org/docs/stable/nn.html#sequential
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                stage.add_module(
                    block_name, block(
                        in_channels, out_channels, stride=stride))
            else:
                stage.add_module(block_name,
                                 block(out_channels, out_channels, stride=1))
        
        return stage

    def _forward_conv(self, x):#class Model
        x = self.conv(x)#3
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x), inplace=True)#inplace=True means that it will modify the input directly, without allocating any additional output
        x = F.adaptive_avg_pool2d(x, output_size=1)#https://discuss.pytorch.org/t/what-is-adaptiveavgpool2d/26897/2
        return x

    def forward(self, x, y):#class Model
        print("prwta edw")

        x = self._forward_conv(x)#1
        # prin:( 32 , 64 , 1 , 1 )
        x = x.view(x.size(0), -1)#why??!
        
        # meta:( 32 , 64 )
        #print("meta:(",x.size(0),",",x.size(1),")")

        x = torch.cat([x, y], dim=1)
        x = self.fc(x)
        return x
