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
from facenet_pytorch import fixed_image_standardization


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


# load model
checkpoint_path = "../best_model.pt"  #"models/Apr07_17-35-04_bb90ec5a54cc/model.pt"
net = MyResNeXt(torch.load(checkpoint_path))
net.eval()

# get data
test_imgs_folder = "../dataset/faceforensics_benchmark_images"
test_imgs = glob(test_imgs_folder + "/*.png")
result_dict = dict()

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

with torch.no_grad():
    for test_img in tqdm(test_imgs):
        img = Image.open(test_img)
        img = transform(img).unsqueeze(0)
        output = torch.sigmoid(net(img))
        # save result
        img_name = test_img.split("/")[-1]
        classify = "fake" if output > 0.5 else "real"
        result_dict[img_name] = classify

# print(result_dict)
with open('../submit/pre_benchmark.json', 'w') as outfile:
    json.dump(result_dict, outfile)
shutil.make_archive("../final_result", "zip", "../submit")


