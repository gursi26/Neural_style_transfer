import cv2 
import torch 
from ResNet import ResNet
from torchvision import transforms
from PIL import Image
import numpy as np 

model_path = '/Users/gursi/Desktop/ML/Neural_style_transfer/models_outputs/ooo/8e8ooo.pt'
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
size_threshold = 300

model = ResNet()
model.load_state_dict(torch.load(model_path, map_location=dev))

t = transforms.ToTensor()

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height)

writer = cv2.VideoWriter('/users/gursi/desktop/output2.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, size)

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

    if _ == True : 
        output_img = cv2.resize(output_img, size)
        writer.write(output_img)

    cv2.imshow('Output', output_img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
            break