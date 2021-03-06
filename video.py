import cv2 
import torch 
import ResNet
from torchvision import transforms
from PIL import Image
import numpy as np 

model_path = '/Users/gursi/Desktop/ML/Neural_style_transfer/models_outputs/geometric/7e8geometric.pt'
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
size_threshold = 600

model = ResNet.ResNet()
model.load_state_dict(torch.load(model_path, map_location=dev))

t = transforms.ToTensor()

cap = cv2.VideoCapture(0)

def post_process_image(data):
        img = data.clone().clamp(0,255).detach().numpy()
        img = img.transpose(1,2,0).astype('uint8')
        img = Image.fromarray(img)
        return img

while True : 

    _, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)

    h, w = img.size 
    while h > size_threshold or w > size_threshold : 
        h = int(h/1.2)
        w = int(w/1.2)
        img = img.resize((h,w), Image.ANTIALIAS)

    model_input = t(img).unsqueeze(dim=0)
    model_output = model.forward(model_input)[0]

    output_img = np.array(post_process_image(model_output))
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

    cv2.imshow('Output', output_img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
            break