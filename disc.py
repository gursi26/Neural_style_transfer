import torch 


class ConvLayer(torch.nn.Module):

    def __init__(self, in_c, out_c, ksize=3, s=2, p=0):
        super(ConvLayer,self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_c, 
            out_channels=out_c, 
            kernel_size=ksize,
            stride=s,
            padding=p
        )
        self.bn = torch.nn.BatchNorm2d(out_c)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))



class Discriminator(torch.nn.Module):

    def __init__(self, input_size):
        super(Discriminator,self).__init__()
        self.conv1 = ConvLayer(3,64)
        self.conv2 = ConvLayer(64,128)
        self.conv3 = ConvLayer(128,256)
        self.conv4 = ConvLayer(256,64)

        self.output_size = self.calculate_output_size(input_size)

        self.fc1 = torch.nn.Linear(self.output_size,500)
        self.fc2 = torch.nn.Linear(500,100)
        self.fc3 = torch.nn.Linear(100,1)

        self.dropout = torch.nn.Dropout(p=0.3)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def calculate_output_size(self, input_size):
        ksize = 3
        stride = 2
        padding = 0 
        num_layers = 4
        final_channels = 64
        size = input_size

        for i in range(num_layers):
            size = int(((size - ksize + (2*padding))/stride) + 1)

        return int(size * size * final_channels)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.dropout(out)
        
        out = torch.flatten(out, start_dim=1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


def test():
    input_shape = 512
    net = Discriminator(input_shape)
    noise = torch.zeros((5,3,input_shape,input_shape))
    output = net.forward(noise)
    print('Input shape : ', noise.shape)
    print('Output shape : ', output.shape)