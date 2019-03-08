import torch
import torch.nn as nn

from collections import namedtuple
import functools

Conv = namedtuple('Conv', ['stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['stride', 'depth'])
InvertedResidual = namedtuple('InvertedResidual', ['stride', 'depth', 'num', 't']) # t is the expension factor
DilatedResidual = namedtuple('DilatedResidual', ['stride', 'depth', 'num', 't'])

Pool2d = namedtuple('Pool2d',['kernel_size','stride'])
V1_CONV_DEFS = [
    Conv(stride=2, depth=32),
    DepthSepConv(stride=1, depth=64),
    DepthSepConv(stride=2, depth=128),
    DepthSepConv(stride=1, depth=128),
    DepthSepConv(stride=2, depth=256),
    DepthSepConv(stride=1, depth=256),
    DepthSepConv(stride=2, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=2, depth=1024),
    DepthSepConv(stride=1, depth=1024)
]
V2_CONV_DEFS = [
    Conv(stride=2, depth=32),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=24, num=2, t=6),
    InvertedResidual(stride=2, depth=32, num=3, t=6),
    InvertedResidual(stride=1, depth=64, num=4, t=6),
    InvertedResidual(stride=1, depth=96, num=3, t=6),
    InvertedResidual(stride=2, depth=160, num=3, t=6),
    InvertedResidual(stride=1, depth=320, num=1, t=6),
]

#3-29
V2_CONV_DEFS = [
    #1
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=20, num=3, t=4),
    InvertedResidual(stride=2, depth=24, num=5, t=4),
    InvertedResidual(stride=1, depth=32, num=3, t=4),
    InvertedResidual(stride=2, depth=40, num=3, t=4),
    InvertedResidual(stride=1, depth=52, num=1, t=4),
]
#3-30
V2_CONV_DEFS = [
    #1InvertedResidual = namedtuple('InvertedResidual', ['stride', 'depth', 'num', 't'])
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=20, num=3, t=4),
    InvertedResidual(stride=2, depth=24, num=3, t=4),
    InvertedResidual(stride=1, depth=30, num=5, t=4),
    InvertedResidual(stride=2, depth=36, num=3, t=4),
]
#4-2    def forward(self, x):

V2_CONV_DEFS = [
    #1
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=20, num=3, t=4),
    InvertedResidual(stride=2, depth=24, num=5, t=4),
    InvertedResidual(stride=1, depth=30, num=3, t=4),
    InvertedResidual(stride=2, depth=36, num=3, t=4),
    InvertedResidual(stride=1, depth=42, num=1, t=4),
]
#4-3Conv
V2_CONV_DEFS = [
    #1
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=20, num=3, t=4),
    InvertedResidual(stride=2, depth=24, num=5, t=4),
    InvertedResidual(stride=1, depth=28, num=3, t=4),
    InvertedResidual(stride=2, depth=32, num=3, t=4),
    InvertedResidual(stride=1, depth=36, num=1, t=4),
]

#4-7
V2_CONV_DEFS = [
    #1
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    InvertedResidual(stride=2, depth=24, num=5, t=4),
    InvertedResidual(stride=1, depth=28, num=3, t=4),
    InvertedResidual(stride=2, depth=32, num=3, t=4),
    InvertedResidual(stride=1, depth=36, num=1, t=4),
]

#4-7-2
V2_CONV_DEFS = [  

    #1
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=16, num=1, t=4),
    InvertedResidual(stride=2, depth=20, num=5, t=4),
    InvertedResidual(stride=1, depth=24, num=3, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=1, t=4),
]

'''

#4-8 quick stride=2-fail
V2_CONV_DEFS = [
    #1Conv
    Conv(stride=2, depth=12),
    InvertedResidual(stride=2, depth=12, num=1, t=1),
    InvertedResidual(stride=1, depth=16, num=1, t=4),
    InvertedResidual(stride=2, depth=20, num=5, t=4),
    InvertedResidual(stride=1, depth=24, num=3, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=1, t=4),
]
#4-8-2
V2_CONV_DEFS = [
    #1
    Conv(stride=2, depth=16),
    InvertedResidual(stride=2, depth=16, num=1, t=1),
    InvertedResidual(stride=1, depth=16, num=1, t=4),
    InvertedResidual(stride=2, depth=20, num=5, t=4),
    InvertedResidual(stride=1, depth=24, num=3, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=1, t=4),
]
'''
#4-11-3
V2_CONV_DEFS = [
    #1
    #nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    InvertedResidual(stride=1, depth=24, num=5, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=1, t=4),
]
#4-12-1
V2_CONV_DEFS = [
    #1
    #nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    InvertedResidual(stride=1, depth=24, num=3, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=1, t=2),
]
#4-12-2-2
V2_CONV_DEFS = [
    #1
    #nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=20, num=3, t=4),
    InvertedResidual(stride=1, depth=24, num=3, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=1, t=2),
]
#4-13-1-2
V2_CONV_DEFS = [
    #1
    #nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=20, num=1, t=1),
    InvertedResidual(stride=2, depth=24, num=3, t=4),
    InvertedResidual(stride=1, depth=24, num=3, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=1, t=2),
]
#4-14-1-2
V2_CONV_DEFS = [
    #1
    #nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=20, num=1, t=4),
    InvertedResidual(stride=2, depth=24, num=3, t=4),
    InvertedResidual(stride=1, depth=30, num=3, t=4),
    InvertedResidual(stride=2, depth=36, num=3, t=4),
    InvertedResidual(stride=1, depth=40, num=1, t=2),
]
#4-16-1-2
V2_CONV_DEFS = [
    #1
    #nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=16, num=2, t=4),
    InvertedResidual(stride=2, depth=24, num=3, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=1, t=2),
]



#4-16-2-2
V2_CONV_DEFS = [
    #1
    #nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=4),
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    InvertedResidual(stride=1, depth=24, num=4, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=3, t=2),

]
#4-17-1
V2_CONV_DEFS = [
    #1
    #nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=4),
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    InvertedResidual(stride=1, depth=24, num=4, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=1, t=4),
]
#4-18-2
V2_CONV_DEFS = [
    #1
    #nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=10, num=1, t=4),
    InvertedResidual(stride=2, depth=14, num=2, t=4),
    InvertedResidual(stride=1, depth=18, num=4, t=4),
    InvertedResidual(stride=2, depth=22, num=4, t=4),
    InvertedResidual(stride=1, depth=26, num=4, t=4),
]
#4-19-1
V2_CONV_DEFS = [
    #1
    #nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=10, num=1, t=4),
    InvertedResidual(stride=2, depth=12, num=2, t=4),
    InvertedResidual(stride=1, depth=16, num=4, t=4),
    InvertedResidual(stride=2, depth=20, num=4, t=4),
    InvertedResidual(stride=1, depth=24, num=4, t=4),
]
#5-4-1
V2_CONV_DEFS = [
    #1
    #nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    DilatedResidual(stride=1, depth=10, num=1, t=4),
    DilatedResidual(stride=2, depth=12, num=2, t=4),
    DilatedResidual(stride=1, depth=16, num=4, t=4),
    DilatedResidual(stride=2, depth=20, num=3, t=4),
    DilatedResidual(stride=1, depth=24, num=1, t=4),
]
#5-7-1
V2_CONV_DEFS = [
    #1
    #nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    DilatedResidual(stride=1, depth=10, num=1, t=4),
    DilatedResidual(stride=2, depth=12, num=2, t=4),
    DilatedResidual(stride=1, depth=16, num=4, t=4),
    DilatedResidual(stride=2, depth=20, num=4, t=4),
    DilatedResidual(stride=1, depth=24, num=4, t=4),
]

#5-8-1
V2_CONV_DEFS = [
    #1
    #nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=10, num=1, t=4),
    InvertedResidual(stride=2, depth=12, num=2, t=4),
    InvertedResidual(stride=1, depth=16, num=3, t=4),
    DilatedResidual(stride=1, depth=16, num=1, t=4),
    InvertedResidual(stride=2, depth=20, num=4, t=4),
    InvertedResidual(stride=1, depth=24, num=3, t=4),    
    DilatedResidual(stride=1, depth=24, num=1, t=4),
]

#5-10-1
V2_CONV_DEFS = [
    #1
    #nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=10, num=1, t=4),
    InvertedResidual(stride=2, depth=12, num=2, t=4),
    InvertedResidual(stride=1, depth=16, num=2, t=4),
    DilatedResidual(stride=1, depth=16, num=2, t=4),
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    DilatedResidual(stride=1, depth=20, num=1, t=4),
    DilatedResidual(stride=1, depth=24, num=1, t=4),
]
#5-14-1

V2_CONV_DEFS = [
    #1
    #nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=10, num=1, t=4),
    
    InvertedResidual(stride=2, depth=12, num=2, t=4),
    
    InvertedResidual(stride=1, depth=16, num=2, t=4),
    DilatedResidual(stride=1, depth=16, num=2, t=4),
    
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    DilatedResidual(stride=1, depth=20, num=2, t=4),
    
    
    InvertedResidual(stride=1, depth=24, num=2, t=4),
    DilatedResidual(stride=1, depth=24, num=2, t=4),
]

#6-15-2
V2_CONV_DEFS = [
    #1
    #nn.MaxPool2d(2)
    Conv(stride=2, depth=16),
    InvertedResidual(stride=2, depth=10, num=2, t=4),
    
    InvertedResidual(stride=2, depth=12, num=2, t=4),

    InvertedResidual(stride=2, depth=16, num=2, t=4),
    DilatedResidual(stride=1, depth=16, num=2, t=4),
    
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    DilatedResidual(stride=1, depth=20, num=2, t=4),
    
    
    InvertedResidual(stride=1, depth=24, num=2, t=4),
    DilatedResidual(stride=1, depth=24, num=2, t=4),
]

#6-16-3
V2_CONV_DEFS = [
    #1
    #nn.MaxPool2d(2)
    Conv(stride=2, depth=16),
    InvertedResidual(stride=2, depth=12, num=2, t=4),
    
    InvertedResidual(stride=2, depth=16, num=2, t=4),

    InvertedResidual(stride=2, depth=20, num=2, t=4),
    DilatedResidual(stride=1, depth=20, num=2, t=4),
    
    InvertedResidual(stride=2, depth=24, num=2, t=4),
    DilatedResidual(stride=1, depth=24, num=2, t=4),
    
    
    InvertedResidual(stride=1, depth=32, num=2, t=4),
    DilatedResidual(stride=1, depth=32, num=2, t=4),
]

#6-18-3
V2_CONV_DEFS = [
    #1
    #nn.MaxPool2d(2)
    Conv(stride=2, depth=16),
    InvertedResidual(stride=2, depth=12, num=2, t=4),
    
    InvertedResidual(stride=2, depth=16, num=2, t=4),

    InvertedResidual(stride=2, depth=24, num=2, t=4),
    DilatedResidual(stride=1, depth=24, num=2, t=4),
    
    InvertedResidual(stride=2, depth=36, num=2, t=4),
    DilatedResidual(stride=1, depth=36, num=2, t=4),
    
    
    InvertedResidual(stride=1, depth=48, num=2, t=4),
    DilatedResidual(stride=1, depth=48, num=2, t=4),
]

V2_CONV_DEFS = [
    Conv(stride=2, depth=32),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=24, num=2, t=6),
    InvertedResidual(stride=2, depth=32, num=3, t=6),
    InvertedResidual(stride=1, depth=64, num=4, t=6),
    InvertedResidual(stride=1, depth=96, num=3, t=6),
    InvertedResidual(stride=2, depth=160, num=3, t=6),
    InvertedResidual(stride=1, depth=320, num=1, t=6),
]


#6-18-3
V2_CONV_DEFS = [
    #1
    #nn.MaxPool2d(2)
    Conv(stride=2, depth=16),
    InvertedResidual(stride=2, depth=12, num=2, t=4),
    
    InvertedResidual(stride=2, depth=16, num=2, t=4),

    InvertedResidual(stride=2, depth=24, num=2, t=4),
    DilatedResidual(stride=1, depth=24, num=2, t=4),
    
    InvertedResidual(stride=2, depth=36, num=2, t=4),
    DilatedResidual(stride=1, depth=36, num=2, t=4),
    
    
    InvertedResidual(stride=1, depth=48, num=2, t=4),
    DilatedResidual(stride=1, depth=48, num=2, t=4),
]
#6-19-3
V2_CONV_DEFS = [
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=12, num=1, t=4),
    InvertedResidual(stride=2, depth=16, num=2, t=4),
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    DilatedResidual(stride=1, depth=20, num=1, t=4),
    InvertedResidual(stride=2, depth=24, num=3, t=4),
    DilatedResidual(stride=1, depth=24, num=1, t=4),
    InvertedResidual(stride=1, depth=30, num=2, t=4),
    DilatedResidual(stride=1, depth=30, num=1, t=4),
    InvertedResidual(stride=2, depth=36, num=3, t=4),
    DilatedResidual(stride=1, depth=48, num=1, t=4),
]

#6-21
V2_CONV_DEFS = [
    Conv(stride=2, depth=32),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=24, num=2, t=6),
    InvertedResidual(stride=2, depth=32, num=3, t=6),
    InvertedResidual(stride=2, depth=64, num=4, t=6),
    InvertedResidual(stride=1, depth=96, num=3, t=6),
    InvertedResidual(stride=2, depth=160, num=3, t=6),
    InvertedResidual(stride=1, depth=320, num=1, t=6),
]

#6-21
V2_CONV_DEFS = [
    Conv(stride=2, depth=32),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=24, num=2, t=6),
    InvertedResidual(stride=2, depth=32, num=3, t=6),
    InvertedResidual(stride=2, depth=64, num=4, t=6),
    InvertedResidual(stride=1, depth=96, num=3, t=6),
    InvertedResidual(stride=2, depth=160, num=3, t=6),
    InvertedResidual(stride=1, depth=320, num=1, t=6),
]

#6-20
V2_CONV_DEFS = [
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=12, num=1, t=4),
    InvertedResidual(stride=2, depth=16, num=2, t=4),
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    InvertedResidual(stride=1, depth=20, num=1, t=4),
    InvertedResidual(stride=2, depth=24, num=3, t=4),
    InvertedResidual(stride=1, depth=24, num=1, t=4),
    InvertedResidual(stride=1, depth=30, num=2, t=4),
    InvertedResidual(stride=1, depth=30, num=1, t=4),
    InvertedResidual(stride=2, depth=36, num=3, t=4),
    InvertedResidual(stride=1, depth=48, num=1, t=4),
]
#6-24-0
V2_CONV_DEFS = [
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=12, num=1, t=4),
    InvertedResidual(stride=2, depth=16, num=2, t=4),
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    InvertedResidual(stride=1, depth=20, num=1, t=4),
    InvertedResidual(stride=2, depth=24, num=3, t=4),
    InvertedResidual(stride=1, depth=24, num=1, t=4),
    InvertedResidual(stride=1, depth=30, num=2, t=4),
    InvertedResidual(stride=1, depth=30, num=1, t=4),
    InvertedResidual(stride=2, depth=36, num=3, t=4),
    InvertedResidual(stride=1, depth=48, num=1, t=4),
]
#6-24-0-2
V2_CONV_DEFS = [
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=12, num=1, t=4),
    InvertedResidual(stride=2, depth=16, num=2, t=4),
    InvertedResidual(stride=2, depth=20, num=3, t=4),
    InvertedResidual(stride=1, depth=24, num=4, t=4),
    InvertedResidual(stride=1, depth=30, num=3, t=4),
    InvertedResidual(stride=2, depth=36, num=3, t=4),
    InvertedResidual(stride=1, depth=48, num=1, t=4),
]
#6-25-0
V2_CONV_DEFS = [
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=12, num=1, t=4),
    InvertedResidual(stride=2, depth=16, num=2, t=4),
    InvertedResidual(stride=2, depth=20, num=3, t=4),
    InvertedResidual(stride=2, depth=24, num=4, t=4),
    InvertedResidual(stride=1, depth=30, num=3, t=4),
    InvertedResidual(stride=2, depth=36, num=3, t=4),
    DilatedResidual(stride=1, depth=48, num=1, t=4),
]

class _conv_bn(nn.Module):
    def __init__(self, inp, oup, stride):
        super(_conv_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class _pool2d(nn.Module):
    def __init__(self):
        super(_pool2d, self).__init__()
        self.pool2d = nn.Sequential(
                nn.MaxPool2d(2,2),
        )
    def forward(self, x):
        return self.pool2d(x)
    
    
class _conv_dw(nn.Module):
    def __init__(self, inp, oup, stride):
        super(_conv_dw, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class _inverted_residual_bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(_inverted_residual_bottleneck, self).__init__()
        self.use_res_connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.depth = oup
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class _dilated_residual_bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(_dilated_residual_bottleneck, self).__init__()
        self.use_res_connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride,  groups=inp * expand_ratio, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.depth = oup
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

def mobilenet(conv_defs, depth_multiplier=1.0, min_depth=8):
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    layers = []
    in_channels = 3
    for conv_def in conv_defs:
        if isinstance(conv_def, Conv):
            layers += [_conv_bn(in_channels, depth(conv_def.depth), conv_def.stride)]
            in_channels = depth(conv_def.depth)
        elif isinstance(conv_def, DepthSepConv):
            layers += [_conv_dw(in_channels, depth(conv_def.depth), conv_def.stride)]
            in_channels = depth(conv_def.depth)
        elif isinstance(conv_def, InvertedResidual):
          for n in range(conv_def.num):
            stride = conv_def.stride if n == 0 else 1
            layers += [_inverted_residual_bottleneck(in_channels, depth(conv_def.depth), stride, conv_def.t)]
            in_channels = depth(conv_def.depth)
        elif isinstance(conv_def,DilatedResidual ):
          for n in range(conv_def.num):
            stride = conv_def.stride if n == 0 else 1
            layers += [_dilated_residual_bottleneck(in_channels, depth(conv_def.depth), stride, conv_def.t)]
            in_channels = depth(conv_def.depth)
    return layers

def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func

mobilenet_v1 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=1.0)
mobilenet_v1_075 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=0.75)
mobilenet_v1_050 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=0.50)
mobilenet_v1_025 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=0.25)

mobilenet_v2 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=1.0)
mobilenet_v2_075 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=0.75)
mobilenet_v2_050 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=0.50)
mobilenet_v2_025 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=0.25)
