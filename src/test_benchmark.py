import shutil
import json
from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from facenet_pytorch import fixed_image_standardization, MTCNN
from model_efficientnet.model import EfficientNet

class MyResNeXt(models.resnet.ResNet): 
    def __init__(self, checkpoint, num_classes=1):
        super(MyResNeXt, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 6, 3],
                                        groups=32,
                                        width_per_group=4)
        # Override the existing FC layer with a new one.
        self.fc = nn.Linear(2048, num_classes)
        if checkpoint:
            self.load_state_dict(checkpoint)


# load model - ResNext
checkpoint_path = "runs/May12_02-47-20_4497d0b803cd/epc_12.pth"  #"models/Apr07_17-35-04_bb90ec5a54cc/model.pt"
net = MyResNeXt(torch.load(checkpoint_path))
net.eval()
margin = 16
mtcnn = MTCNN(device='cuda', margin=margin)
mtcnn.eval()

# get data
test_imgs_folder = "../dataset/faceforensics_benchmark_images"
test_imgs = glob(test_imgs_folder + "/*.png")
result_dict = dict()

transform = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
])

with torch.no_grad():
    for test_img in tqdm(test_imgs):
        img = Image.open(test_img)
        # get face
        boxes, probs = mtcnn.detect([img])
        box = boxes[0][0]
        box = [box[0]-margin, box[1]-margin, box[2]+margin, box[3]+margin]
        face = img.crop(box)
        # # test
        # face.save(f"test.jpg", "jpeg") 
        # break
        face = transform(face).unsqueeze(0)
        output = net(face)  #torch.sigmoid(net(face))
        # save result
        img_name = test_img.split("/")[-1]
        classify = "real" if output > 0. else "fake"
        result_dict[img_name] = classify

# print(result_dict)
with open('../submit/pre_benchmark.json', 'w') as outfile:
    json.dump(result_dict, outfile)
shutil.make_archive("../final_result", "zip", "../submit")


