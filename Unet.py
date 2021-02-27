import torch 

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
    
    out = torch.cat([skips[0],out], dim=1)
    out = self.upsample1(out)
    out = torch.cat([skips[1],out], dim=1)
    out = self.upsample2(out)
    out = torch.cat([skips[2],out], dim=1)
    out = self.upsample3(out)

    out = self.final(out)

    return out




def test():
    model = UNET_Gen()
    noise = torch.randn((10,3,256,256))
    output = model.forward(noise)
    print('Input shape : ', noise.shape)
    print('Output shape : ', output.shape)

test()