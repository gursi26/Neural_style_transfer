import torch 

'''

'''
class Convlayer(torch.nn.Module):

    def __init__(self, in_c, out_c, ksize, stride, pad=0):
        super(Convlayer,self).__init__()
        reflection_padding = ksize // 2

        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv = torch.nn.Conv2d(
            in_channels=in_c, 
            out_channels=out_c, 
            kernel_size=ksize, 
            stride=stride,
            padding=pad
        )

    def forward(self,x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        return out

'''

'''
class Upsample(torch.nn.Module):

    def __init__(self, in_c, out_c, ksize, stride, scale=2, pad=0):
        super(Upsample,self).__init__()
        reflection_padding = ksize // 2
        self.scale = scale

        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv = torch.nn.Conv2d(
            in_channels=in_c, 
            out_channels=out_c, 
            kernel_size=ksize, 
            stride=stride,
            padding=pad
        )

    def forward(self,x):
        out = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
        out = self.reflection_pad(out)
        out = self.conv(out)
        return out

'''

'''

class ResidualBlock(torch.nn.Module):

    def __init__(self, in_c, out_c):
        super(ResidualBlock,self).__init__()

        self.conv1 = Convlayer(in_c, out_c, ksize=3, stride=1)
        self.norm1 = torch.nn.InstanceNorm2d(out_c, affine=True)
        self.conv2 = Convlayer(out_c, out_c, ksize=3, stride=1)
        self.norm2 = torch.nn.InstanceNorm2d(out_c, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        identity = x.clone()
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + identity 
        return out


'''

'''

class ResNet(torch.nn.Module):

    def __init__(self):
        super(ResNet,self).__init__()

        self.downsample1 = Convlayer(in_c=3, out_c=32, ksize=9, stride=1)
        self.norm1 = torch.nn.InstanceNorm2d(32, affine=True)

        self.downsample2 = Convlayer(in_c=32, out_c=64, ksize=3, stride=2)
        self.norm2 = torch.nn.InstanceNorm2d(64, affine=True)

        self.downsample3 = Convlayer(in_c=64, out_c=128, ksize=3, stride=2)
        self.norm3 = torch.nn.InstanceNorm2d(128, affine=True)

        self.res1 = ResidualBlock(128,128)
        self.res2 = ResidualBlock(128,128)
        self.res3 = ResidualBlock(128,128)
        self.res4 = ResidualBlock(128,128)
        self.res5 = ResidualBlock(128,128)

        self.upsample1 = Upsample(128, 64, ksize=3, stride=1)
        self.norm4 = torch.nn.InstanceNorm2d(64, affine=True)

        self.upsample2 = Upsample(64, 32, ksize=3, stride=1)
        self.norm5 = torch.nn.InstanceNorm2d(32, affine=True)

        self.final = Convlayer(32, 3, ksize=9, stride=1)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        out = self.relu(self.norm1(self.downsample1(x)))
        out = self.relu(self.norm2(self.downsample2(out)))
        out = self.relu(self.norm3(self.downsample3(out)))

        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)

        out = self.relu(self.norm4(self.upsample1(out)))
        out = self.relu(self.norm5(self.upsample2(out)))
        out = self.final(out)
        return out




def test():
    net = ResNet()
    noise = torch.randn((5,3,256,256)) # (Batch_size, channels, height, width)
    output = net(noise)

    print('Input shape : ', noise.shape)
    print('Output shape : ', output.shape)