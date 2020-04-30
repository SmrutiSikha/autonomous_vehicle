# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 19:56:21 2020

@author: Santanu
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet

class BuildingBlock(nn.Module):
    
    def __init__(self, in_feature, out_feature, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(BuildingBlock, self).__init__()
        
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.groups = groups
        
        #framing the layers
        self.conv1 = nn.Conv2d(in_feature, out_feature, kernel_size, stride, padding, groups = groups,
                               bias = False)
        self.bn1 = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_feature, out_feature, kernel_size, 1, padding, groups = groups, 
                               bias = False)
        self.bn2 = nn.BatchNorm2d(out_feature)
        self.downsample = None
        
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_feature, out_feature, kernel_size, groups = groups,
                                                      bias = False), nn.BatchNorm2d(out_feature))
        
    
    def forward(self, x):
        residuary = x
        
        #forward propagation
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        
        if self.downsample is not None:
            residuary = self.downsample(x)
            
        out += residuary
        out = self.relu(out)
        
        return out
    
# Encoder block starting.............................................

class EncoderBlock(nn.Module):
    
    def __init__(self, in_feature, out_feature, kernel_size, stride =1, groups = 1, padding = 0, bias = False):
        super(EncoderBlock, self).__init__.()
        
        self.in_feature = in_feature
        self.out_feature = out_feature
        
        #inner layers of encoder block
        self.block1 = BuildingBlock(in_feature, out_feature, kernel_size, stride, ,padding, groups, bias)
        self.block2 = BuildingBlock(out_feature, out_feature, kernel_size, 1, padding, groups, bias)
        
    def forward(self, x):
        x = self.block2(self.block1(x))
        
        return x
    
# Decoder block starting..............................................
        
class DecoderBlock(nn.Module):
    
    def __init__(self, in_feature, out_feature, kernel_size, stride =1, groups = 1, padding = 0, 
                 output_padding = 0, bias = False):
        super(DecoderBlock, self).__init()
        
        self.in_feature = in_feature
        self.out_feature = out_feature
        
        #inner layers of decoder block
        self.conv1 = nn.Sequential(nn.Conv2d(in_feature, in_feature//4, 1, 1, 0, bias = False),
                                   nn.BatchNorm2d(in_feature//4), 
                                   nn.ReLU())
        self.conv_tp = nn.Sequential(nn.ConvTranspose2d(in_feature//4, in_feature//4, kernel_size, stride, 
                                                        padding, output_padding, bias = False), 
                                     nn.BatchNorm2d(in_feature//4),
                                     nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_feature//4, out_feature, 1, 1, 0, bias = False),
                                   nn.BatchNorm2d(out_feature),
                                   nn.ReLU())
        
    def forward(self, x):
        x = self.conv2(self.conv_tp(self.conv1(x)))
        
        return x
    
# linknet formed from the encoder and decoder blocks defined above

class LinkNetBase(nn.Module):
    
    def __init__(self, n_classes):
        super(LinkNetBase, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.encoder1 = EncoderBlock(64, 64, 3, 1, 1)
        self.encoder2 = EncoderBlock(64, 128, 3, 2, 1)
        self.encoder3 = EncoderBlock(128, 256, 3, 2, 1)
        self.encoder4 = EncoderBlock(256, 512, 3, 2, 1)

        self.decoder1 = DecoderBlock(64, 64, 3, 1, 1, 0)
        self.decoder2 = DecoderBlock(128, 64, 3, 2, 1, 1)
        self.decoder3 = DecoderBlock(256, 128, 3, 2, 1, 1)
        self.decoder4 = DecoderBlock(512, 256, 3, 2, 1, 1)

        # Classifier
        self.conv1_tp = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.conv2_tp = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Initial block
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        # Propagation through Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Propagation through Decoder blocks
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)

        # Classifier
        y = self.lsm(self.conv2_tp(self.conv2(self.conv1_tp(d1))))

        return y

# linknet formed using the encoders of resnet-18
        
class LinkNet(nn.Module):
    
    def __init__(self, n_classes=21):
        super(LinkNet, self).__init__()

        base = resnet.resnet18(pretrained=True)

        self.in_block = nn.Sequential(base.conv1,
                                      base.bn1,
                                      base.relu,
                                      base.maxpool
                                      )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.decoder1 = DecoderBlock(64, 64, 3, 1, 1, 0)
        self.decoder2 = DecoderBlock(128, 64, 3, 2, 1, 1)
        self.decoder3 = DecoderBlock(256, 128, 3, 2, 1, 1)
        self.decoder4 = DecoderBlock(512, 256, 3, 2, 1, 1)

        # Classifier
        self.conv1_tp = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.conv2_tp = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        self.lsm = nn.LogSoftmax(dim=1)


    def forward(self, x):
        # Initial block
        x = self.in_block(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)

        # Classifier
        y = self.lsm(self.conv2_tp(self.conv2(self.conv1_tp(d1))))

        return y
