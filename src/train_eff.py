import datetime
import argparse
import pickle
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
from grid import *
from facenet_pytorch import InceptionResnetV1
from torch.utils.tensorboard import SummaryWriter
from my_utils import set_loger, freeze_until, model_validation, set_lr, randomJPEGcompression
from model_efficientnet.model import EfficientNet


class FaceRecognitionCNN(nn.Module):
    def __init__(self):
        super(FaceRecognitionCNN, self).__init__()
        self.resnet = InceptionResnetV1(pretrained='vggface2')
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 1)

    def forward(self, images):
        out = self.resnet(images)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out.squeeze()


class MyResNeXt(models.resnet.ResNet):
    def __init__(self):
        super(MyResNeXt, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 6, 3],
                                        groups=32,
                                        width_per_group=4)

        self.load_state_dict(torch.load("pretrained_model/resnext50_32x4d-7cdf4587.pth"))
        # Override the existing FC layer with a new one.
        self.fc = nn.Linear(2048, 1)


class ImageDataset(Dataset):
    def __init__(self, datas, labels, transform=None):
        self.datas = datas
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        this_data = self.datas[index]
        this_label = self.labels[index]
        img = Image.open(this_data).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(this_label, dtype=torch.float32)  # bce loss
        return img, label

    def __len__(self):
        return len(self.datas)


## Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", "-lr", type=float, default=0.00001)
parser.add_argument("--regularization", type=float, default=0.05)
parser.add_argument("--freeze_until_layer", type=str, default="")  # layer2.0.conv1.weight  _blocks.38._expand_conv.weight
parser.add_argument("--batch_size", "-b", type=int, default=16, help="每张显卡的数量")
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--att", action="store_true", default=False)
# set grid attribute
parser.add_argument("--grid", action="store_true", default=False, help="是否用grid增强")
parser.add_argument("--d1", type=float, default=10, help="遮挡块的大小下限")
parser.add_argument("--d2", type=float, default=50, help="遮挡块的大小上限")
parser.add_argument("--rotate", type=float, default=45)
parser.add_argument("--ratio", type=float, default=0.6)
parser.add_argument("--mode", type=int, default=1, help="GridMask mode, 1 or 0")
parser.add_argument("--prob", type=float, default=0.5)
args = parser.parse_args()

## Get logger
writer = SummaryWriter()
logger = set_loger(f"{writer.get_logdir()}/train.log")
logger.info(args)

## Get model
# ResNext50
# net = MyResNeXt()
# net.load_state_dict(torch.load("runs/Apr11_13-14-08_bb90ec5a54cc/epc_12.pth"))

# Inception
# net = FaceRecognitionCNN()

# EfficinetNet
# ckp = torch.load("pretrained_model/efficientnet-b7-dcc49843.pth")
ckp = torch.load("pretrained_model/epc_17.pth")  # epc_12.pth是最开始用原始efficientnet的模型，epc_17.pth是之后用了attention模块的efficientnet
net = EfficientNet.from_pretrained("efficientnet-b7", checkpoint=ckp, num_classes=1, use_att=args.att)
# net._fc = nn.Linear(2560, 1)


device = torch.cuda.device_count()
batch_size = args.batch_size
if args.freeze_until_layer:
    freeze_until(net, args.freeze_until_layer)
if device:
    net = net.cuda()
    net = nn.DataParallel(net, device_ids=[i for i in range(device)])
    batch_size *= device

## Get data
data_pickle_path = '../dataset/retina_all.pkl'  #"../dataset/c40_all.pkl"  #"../dataset/data_all.pkl"
data_train, label_train, data_val, label_val = pickle.load(open(data_pickle_path, "rb"))

train_transformer = transforms.Compose([
    transforms.Lambda(randomJPEGcompression),  # 对图片进行随机压缩
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=45, scale=(0.6,1.5), shear=30),  # degree旋转(在二维平面旋转)，scale缩放，shear旋转(绕z轴旋转的角度，也就是在第3个维度旋转)
    transforms.ColorJitter(brightness=(0.2,1.5), contrast=(0.2,2), saturation=(0.2,2), hue=0.1),
    transforms.Resize((224, 224)),  # 256
    # transforms.RandomCrop(224),  # 160
    transforms.ToTensor(),
    transforms.Normalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
])

val_transformer = transforms.Compose([
    transforms.Resize((224,224)),  # 160
    transforms.ToTensor(),
    transforms.Normalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
])

dataset_train = ImageDataset(data_train, label_train, train_transformer)
dataset_val = ImageDataset(data_val, label_val, val_transformer)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

# Get gridmask
if args.grid:
    grid = GridMask(args.d1, args.d2, args.rotate, args.ratio, args.mode, args.prob)

# Get loss and optimizer
criterion = nn.BCEWithLogitsLoss()
lr = args.learning_rate
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.regularization)
# WARM_UP_RATIO = 1e3
# LR_RAMPUP_EPOCHS = 5
# warmup_rate = WARM_UP_RATIO**(1/LR_RAMPUP_EPOCHS)
# lr /= WARM_UP_RATIO
# optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

# lr scheduler
def lrfn(optimizer, epoch):
    LR_START = args.learning_rate  #1e-5
    LR_MAX = args.learning_rate*10  #5e-5
    LR_MIN = args.learning_rate*1e-3  #1e-8
    LR_RAMPUP_EPOCHS = 5
    LR_SUSTAIN_EPOCHS = 0
    LR_EXP_DECAY = .85

    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    set_lr(optimizer, lr)


## Train model
len_loader = len(dataloader_train)
step = 1
best_loss = 100
best_acc = 0
best_epc = 0
for epoch in range(args.epochs):
    running_loss = 0.0
    total_bz = 0
    net.zero_grad()
    net.train()
    if args.grid:
        grid.set_prob(epoch, args.epochs)

    for itr, (data, label) in enumerate(dataloader_train):
        bz = data.shape[0]
        total_bz += bz
        optimizer.zero_grad()
        data = data.cuda()
        # use gridmask
        if args.grid:
            data = grid(data)
        output = net(data).squeeze()
        label = label.cuda()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*bz

        # get acc
        # output[output>0] = 1
        # output[output<=0] = 0
        output = output > 0.0
        real_id = label == 1
        fake_id = label == 0
        real_predictions = output[real_id]
        fake_predictions = output[fake_id]
        if len(real_predictions)==0:
            real_acc = 0 if (real_predictions==1).sum().float().item() else 1
        if len(fake_predictions)==0:
            fake_acc = 0 if (fake_predictions==0).sum().float().item() else 1
        try:
            real_acc = (real_predictions==1).sum().float().item() / len(real_predictions)
            fake_acc = (fake_predictions==0).sum().float().item() / len(fake_predictions)
        except:
            pass
        train_acc = (output==label).sum().float().item()/bz
        writer.add_scalar("Loss/train", loss.item(), step)
        writer.add_scalar("Acc/train", train_acc, step)
        step += 1
        if not itr%100:
            logger.info('[EPOCH %d ITER %3d/%d] train acc: %.8f (real: %.8f, fake: %.8f), loss: %.8f (%.4f)' % (epoch, itr, len_loader, train_acc, real_acc, fake_acc, loss.item(), running_loss/total_bz))
            # val_loss, val_acc, real_acc, fake_acc = model_validation(net, dataloader_val, criterion)
            # writer.add_scalar("Loss/val", val_loss, step-1)
            # writer.add_scalar("Acc/val", val_acc, step-1)
            # writer.add_scalar("DetailAcc/real", real_acc, step-1)
            # writer.add_scalar("DetailAcc/fake", fake_acc, step-1)
            # if val_loss > best_loss:
            #     torch.save(net.module.state_dict(), f"{writer.get_logdir()}/best_loss.pth")
            #     best_loss = val_loss
    # lr schedule
    lrfn(optimizer, epoch)
    # # Wramup
    # if epoch < LR_RAMPUP_EPOCHS:
    #     lr *= warmup_rate
    #     set_lr(optimizer, lr) 
    #     logger.info(f'[EPOCH %d] Set LR to {lr}')
    # elif epoch in [15, 30]:
    #     lr *= 0.9
    #     set_lr(optimizer, lr)
    #     logger.info(f'[EPOCH %d] Set LR to {lr}')

    # Do validation
    val_loss, val_acc, real_acc, fake_acc = model_validation(net, dataloader_val, criterion)
    writer.add_scalar("Loss/val", val_loss, step-1)
    writer.add_scalar("Acc/val", val_acc, step-1)
    writer.add_scalar("DetailAcc/real", real_acc, step-1)
    writer.add_scalar("DetailAcc/fake", fake_acc, step-1)
    if val_loss < best_loss:
        torch.save(net.module.state_dict(), f"{writer.get_logdir()}/best_loss.pth")
        best_loss = val_loss
        best_acc = val_acc
        best_epc = epoch

    # Save model
    torch.save(net.module.state_dict(), f"{writer.get_logdir()}/epc_{epoch}.pth")

    logger.info('[EPOCH %d] train loss:%.8f, val loss:%.8f, val acc:%.8f, real acc:%.8f, fake acc:%.8f (Best epc:%d, val loss:%.8f, val acc:%.8f)' % (epoch, running_loss/total_bz, val_loss, val_acc, real_acc, fake_acc, best_epc, best_loss, best_acc))

logger.info("Train end.")
