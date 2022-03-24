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
from facenet_pytorch import fixed_image_standardization, MTCNN
from model_efficientnet.model import EfficientNet
from detect_face import *



# load model - EfficientNet-B7
# May21_05-24-35_4497d0b803cd的模型用了Attention，这里模型要加上
checkpoint_path = "runs/Jun29_03-07-18_4497d0b803cd/epc_36.pth"  #"models/Apr07_17-35-04_bb90ec5a54cc/model.pt"
net = EfficientNet.from_pretrained("efficientnet-b7", checkpoint=torch.load(checkpoint_path), num_classes=1, use_att=True)
net.eval()
# margin = 16
detector = face_detector()
print("Use ckp: ", checkpoint_path)
# print(net)

# get data
test_imgs_folder = "../../faceforencisc_detection/dataset/faceforensics_benchmark_images"  #"../dataset/faceforensics_benchmark_images"
test_imgs = glob(test_imgs_folder + "/*.png")
result_dict = dict()

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


