'''
在 test_benchmark_eff.py 的基础上加了tta，这个tta是手写的没有做成函数。

@date: 2020.7.1
@author: zz
'''

import shutil
import json
from glob import glob
from tqdm import tqdm
from PIL import Image
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import torchvision.transforms.functional as F
from facenet_pytorch import fixed_image_standardization, MTCNN
from model_efficientnet.model import EfficientNet
from detect_face import *



# load model - EfficientNet-B7
# May21_05-24-35_4497d0b803cd的模型用了Attention，这里模型要加上
checkpoint_path = "runs/Jul02_04-16-43_4497d0b803cd/epc_24.pth"  #"models/Apr07_17-35-04_bb90ec5a54cc/model.pt"
net = EfficientNet.from_pretrained("efficientnet-b7", checkpoint=torch.load(checkpoint_path), num_classes=1, use_att=True)
net.eval().to('cuda')
# margin = 16
detector = face_detector()
print("Use ckp: ", checkpoint_path)
# print(net)

# get data
test_imgs_folder = "../dataset/faceforensics_benchmark_images"
test_imgs = glob(test_imgs_folder + "/*.png")
result_dict = dict()

def tta_aug(img):  # 懒得写参数了，直接写要做什么
    '''
    输入一张PIL图片，输出一组PIL图片
    '''
    imgs = [img]
    # Affine
    rotates = [15,30,45,0,0,0,0]
    shears = [0,0,0,10,20,0,0]
    scales = [1,1,1,1,1,0.8,1.2]
    imgs.extend([F.affine(img, angle=r,translate=(0,0),scale=sc,shear=sh) for r,sc,sh in zip(rotates,scales,shears)])
    # Hflip
    flips = []
    flips.extend([F.hflip(img=i) for i in imgs])
    imgs.extend(flips)
    return imgs

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
])

with torch.no_grad():
    for test_img in tqdm(test_imgs):
        img = cv.imread(test_img)
        # get face
        face = detector.detect(img)
        face = Image.fromarray(face[...,::-1])  # BGR->RGB

        faces = tta_aug(face)

        faces = [transform(f) for f in faces]
        faces = torch.stack(faces).to('cuda')
        output = net(faces)  #torch.sigmoid(net(face))
        # print('output: ', output)
        output = output.mean()
        # print('mean: ', output)
        # save result
        img_name = test_img.split("/")[-1]
        classify = "real" if output > 0. else "fake"
        result_dict[img_name] = classify

# print(result_dict)
with open('../submit/pre_benchmark.json', 'w') as outfile:
    json.dump(result_dict, outfile)
shutil.make_archive("../final_result", "zip", "../submit")


