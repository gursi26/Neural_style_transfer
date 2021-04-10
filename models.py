import torch 
from torchvision.models import vgg16 

## Loss network 
class vgg(torch.nn.Module):

    def __init__(self):
        super(vgg,self).__init__()

        self.activation_layers = ['3','8','15','22']
        self.model = vgg16(pretrained=True).features[:23]

    def forward(self,x):
        activations = []
        for layer_number, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_number) in self.activation_layers : 
                activations.append(x)

        return activations


## ResNet model
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



class UpsampleBlock(torch.nn.Module):

    def __init__(self, in_c, out_c, ksize, stride, scale=2, pad=0):
        super(UpsampleBlock,self).__init__()
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

        self.upsample1 = UpsampleBlock(128, 64, ksize=3, stride=1)
        self.norm4 = torch.nn.InstanceNorm2d(64, affine=True)

        self.upsample2 = UpsampleBlock(64, 32, ksize=3, stride=1)
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


# Unet model
class convblock(torch.nn.Module):

  def __init__(self, in_c, out_c, k=4, s=2, p=0):
    super(convblock,self).__init__()
    self.conv = torch.nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
    self.norm = torch.nn.InstanceNorm2d(out_c, affine=True)
    self.relu = torch.nn.ReLU()

  def forward(self,x):
    out = self.conv(x)
    out = self.norm(out)
    return self.relu(out)


class convtransposeblock(torch.nn.Module):

  def __init__(self, in_c, out_c, k=4, s=2, p=0):
    super(convtransposeblock,self).__init__()
    self.upsample = torch.nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
    self.norm = torch.nn.InstanceNorm2d(out_c, affine=True)
    self.relu = torch.nn.ReLU()

  def forward(self,x):
    out = self.upsample(x)
    out = self.norm(out)
    return self.relu(out)



class UNET_Gen(torch.nn.Module):

  def __init__(self):
    super(UNET_Gen,self).__init__()

    self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=0)

    self.downsample1 = convblock(64,128)
    self.downsample2 = convblock(128,256)
    self.downsample3 = convblock(256,512)

    self.midlayer1 = convblock(in_c=512, out_c=1024)
    self.midlayer2 = convtransposeblock(in_c=1024, out_c=512)

    self.upsample1 = convtransposeblock(1024,256)
    self.upsample2 = convtransposeblock(512,128)
    self.upsample3 = convtransposeblock(256,64,k=5)

    self.final = torch.nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2)
    

  def forward(self,x):
    out = self.conv1(x)
    skips = []
    
    out = self.downsample1(out)
    skips.append(out.clone())
    out = self.downsample2(out)
    skips.append(out.clone())
    out = self.downsample3(out)
    skips.append(out.clone())

    skips.reverse()

    out = self.midlayer1(out)
    out = self.midlayer2(out)
    
    upsample_layers = [self.upsample1, self.upsample2, self.upsample3]
    for i in range(3):
      out = torch.cat([out,skips[i]], dim=1)
      out = upsample_layers[i](out)

    out = self.final(out)

    return out


