import torch 
from torchvision.models import vgg16 
from collections import namedtuple

'''

'''

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


def test():
    model = vgg()
    noise = torch.randn((5,3,256,256))
    print('Input shape : ', noise.shape)

    output = model.forward(noise)
    for activation in output : 
        print('Activation shape : ', activation.shape)