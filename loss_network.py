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

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(activations[0], activations[1], activations[2], activations[3])
        return out


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


def test():
    model = vgg()
    noise = torch.randn((5,3,256,256))
    print('Input shape : ', noise.shape)

    output = model.forward(noise)
    for activation in output : 
        print('Activation shape : ', activation.shape)