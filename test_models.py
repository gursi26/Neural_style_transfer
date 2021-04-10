import torch, os, math
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from models import ResNet

#############################################################################
class Display_samples(object):

    def __init__(self, dataset_path, model_path, dev='cpu'):

        self.dataset_path = dataset_path 
        self.image_names = os.listdir(self.dataset_path)
        self.image_names.remove('.DS_Store')
        self.image_paths = [os.path.join(self.dataset_path, x) for x in self.image_names]
        self.model_path = model_path 
        self.dev = dev

        self.load_model()

    def load_model(self):
        self.model = ResNet().to(self.dev)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.dev))
        self.model.eval()

    def post_process_image(self, data):
        img = data.clone().clamp(0,255).detach().numpy()
        img = img.transpose(1,2,0).astype('uint8')
        img = Image.fromarray(img)
        return img


    def display(self, size_threshold=600, show_original=True, save=False):
        to_tensor = transforms.ToTensor()
        to_image = transforms.ToPILImage()

        ncols = 4
        nrows = 2 if show_original else 1
        nfigs = math.ceil(len(self.image_paths)/ncols)
        img_counter = 0

        for i in range(nfigs):
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(23,12))

            if show_original : 
                ax1, ax2 = ax 
            else : 
                ax2 = ax
                ax1 = torch.ones((ax.shape))

            for ax_1, ax_2 in zip(ax1, ax2): 

                try : 

                    img = Image.open(self.image_paths[img_counter]).convert('RGB')
                    img_counter += 1
                    h, w = img.size 
                    
                    while h > size_threshold or w > size_threshold : 
                        h = int(h/1.2)
                        w = int(w/1.2)
                        img = img.resize((h,w), Image.ANTIALIAS)

                    img_tensor = to_tensor(img).unsqueeze(0)

                    out_tensor = self.model.forward(img_tensor)
                    out_img = self.post_process_image(out_tensor[0])

                    if show_original : 
                        ax_1.imshow(img)
                        ax_1.axis(False)

                    ax_2.imshow(out_img)
                    ax_2.axis(False)
                    
                except IndexError : 
                    pass

            self.fig = fig
            if save : 
                self.savefig()

    def savefig(self):
        rootdir = os.path.dirname(os.path.abspath(self.model_path))
        modelname = os.path.basename(self.model_path)
        savename = modelname[:-3] + '.png'
        save_path = os.path.join(rootdir, savename)
        self.fig.savefig(save_path, bbox_inches='tight', transparent=True, pad_inches=0)


#############################################################################
class SingleImage(object):

    def __init__(self):
        self.models_list = [] 


    def add_model(self, model_path):
        model = ResNet()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.models_list.append(model)


    def add_multiple(self, models_path):
        names_list = os.listdir(models_path)
        paths_list = [os.path.join(models_path, x) for x in names_list]

        for i in paths_list : 
            if i.endswith('.pt'):
                model = ResNet()
                model.load_state_dict(torch.load(i, map_location='cpu'))
                self.models_list.append(model)

    def add_all(self):
        path = '/Users/gursi/Desktop/ML/Neural_style_transfer/models_outputs'
        items = os.listdir(path)
        items.remove('original.png')
        items.remove('.DS_Store')
        for item in items : 
            self.add_multiple(os.path.join(path,item))


    def deprocess(self,img):
        img = img.clamp(0,255).detach().numpy()
        img = img.transpose(1,2,0).astype('uint8')
        img = Image.fromarray(img)
        return img


    def convert(self, input_img_path, size_threshold=600, individual=False, ncols = 4): 
        import math 

        input_img = Image.open(input_img_path).convert('RGB')
        h, w = input_img.size 

        while h > size_threshold or w > size_threshold : 
            h = int(h // 1.2)
            w = int(w // 1.2)

        input_img = input_img.resize((h,w)) 
        num_images = len(self.models_list)

        nrows = math.ceil(num_images/ncols)

        if not individual : 
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows * 4,ncols * 4))
            ax = ax.flatten()
            ax = ax[:num_images + 1]
        else : 
            ax = torch.zeros((len(self.models_list)))
        
        model_input = transforms.ToTensor()(input_img).unsqueeze(0)

        if not individual : 
            ax[0].imshow(input_img)
            ax[0].axis(False)
        else : 
            plt.figure(figsize = (12,9))
            plt.axis(False)
            plt.imshow(input_img)

        if not individual : 
            to_loop = ax[1:]
        else :
            to_loop = ax

        for model, axes in zip(self.models_list,to_loop) : 
            out = model.forward(model_input)
            img = self.deprocess(out[0])
            if not individual : 
                axes.axis(False)
                axes.imshow(img)
            else : 
                plt.figure(figsize=(12,9))
                plt.axis(False)
                plt.imshow(img)


#############################################################################
class Random_sample(object):

    def __init__(self, dataset_path, model_path, dev='cpu'):
        self.dataset_path = dataset_path 
        self.model_path = model_path 
        self.dev = dev

        self.load_model()

    def load_model(self):
        self.model = ResNet().to(self.dev)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.dev))
        self.model.eval()

    def post_process_image(self, data):
        img = data.clone().clamp(0,255).detach().numpy()
        img = img.transpose(1,2,0).astype('uint8')
        img = Image.fromarray(img)
        return img


    def display(self, num, size_threshold=600):
        to_tensor = transforms.ToTensor()
        to_image = transforms.ToPILImage()

        image_names = os.listdir(self.dataset_path)
        random_sample_idx = torch.randint(low=0, high=len(image_names), size=(num,))


        cols = 2
        rows = int(num // cols)
        fig, axes = plt.subplots(ncols=rows*2, nrows=cols, figsize=(23,(num+1)*2))
        col1ax = axes[0,:]
        col2ax = axes[1,:]

        counter = 0
        for ax1, ax2, idx in zip(col1ax, col2ax, random_sample_idx) : 
            img = Image.open(os.path.join(self.dataset_path, image_names[idx]))
            height, width = img.size

            while height > size_threshold or width > size_threshold : 
                height = int(height //1.5)
                width = int(width // 1.5)


            img = img.resize((height,width))
            img_tensor = to_tensor(img)

            out = self.model.forward(img_tensor.unsqueeze(0).to(self.dev))
            out = self.post_process_image(out[0])

            ax1.imshow(img)
            print('Image name : ', image_names[idx])
            ax1.axis(False)

            ax2.imshow(out)
            ax2.axis(False)

            counter += 1