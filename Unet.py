import torch 

class UNET_Gen(torch.nn.Module):

  def __init__(self):
    super(UNET_Gen,self).__init__()

    self.start = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=0),
        torch.nn.ReLU()
    )

    self.downsample = [
                       self.convblock(64,128),
                       self.convblock(128,256),
                       self.convblock(256,512)
    ]

    self.midlayer1 = self.convblock(in_c=512, out_c=1024)
    self.midlayer2 = self.convtransposeblock(in_c=1024, out_c=512)

    self.upsample = [
                     self.convtransposeblock(1024,256),
                     self.convtransposeblock(512,128),
                     self.convtransposeblock(256,64, k=5)
    ]

    self.final = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2),
        torch.nn.Tanh()
    )

  def convblock(self, in_c, out_c, k=4, s=2, p=0):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
        torch.nn.InstanceNorm2d(out_c, affine=True),
        torch.nn.ReLU()
    )
    

  def convtransposeblock(self, in_c, out_c, k=4, s=2, p=0):
    return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
        torch.nn.InstanceNorm2d(out_c, affine=True),
        torch.nn.ReLU()
    )


  def forward(self,x):
    x = self.start(x)
    skips = []
    for layer in self.downsample : 
      x = layer(x)
      skips.append(x)

    skips.reverse()
    x = self.midlayer1(x)
    x = self.midlayer2(x)
    
    for skipped,layer in zip(skips,self.upsample):
      x = torch.cat([skipped,x], dim=1)
      x = layer(x)
    
    x = self.final(x)

    return x


def test():
    model = UNET_Gen()
    noise = torch.randn((10,3,256,256))
    output = model.forward(noise)
    print('Input shape : ', noise.shape)
    print('Output shape : ', output.shape)