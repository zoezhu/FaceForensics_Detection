import datetime
import argparse
import pickle
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
from grid import *
from facenet_pytorch import InceptionResnetV1
from torch.utils.tensorboard import SummaryWriter
from my_utils import set_loger, freeze_until, model_validation, set_lr
# from efficientnet.model import EfficientNet


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
parser.add_argument("--learning_rate", type=float, default=0.00001)
parser.add_argument("--regularization", type=float, default=0.05)
parser.add_argument("--freeze_until_layer", type=str, default="layer2.0.conv1.weight")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--epochs", type=int, default=30)
# set grid attribute
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
net = MyResNeXt()
# net.load_state_dict(torch.load("runs/Apr11_13-14-08_bb90ec5a54cc/epc_12.pth"))

# Inception
# net = FaceRecognitionCNN()

# EfficinetNet
# model_name = "efficientnet-b7"
# load_path = {"efficientnet-b0": "pretrained_model/efficientnet-b0-355c32eb.pth",
#                 "efficientnet-b5": "pretrained_model/efficientnet-b5-b6417697.pth",
#                 "efficientnet-b7": "src/pretrained_model/efficientnet-b7-dcc49843.pth"}
# checkpoint = torch.load(load_path[model_name])
# net = EfficientNet.from_pretrained(model_name, checkpoint, num_classes=2)

device = torch.cuda.device_count()
batch_size = args.batch_size
if args.freeze_until_layer:
    freeze_until(net, args.freeze_until_layer)
if device:
    net = net.cuda()
    net = nn.DataParallel(net, device_ids=[i for i in range(device)])
    batch_size *= device

## Get data
data_pickle_path = "../dataset/c40_all.pkl"  #"../dataset/neural_original_all.pkl"
data_train, label_train, data_val, label_val = pickle.load(open(data_pickle_path, "rb"))

train_transformer = transforms.Compose([
    transforms.Resize((184, 184)),
    transforms.RandomCrop(160),
    transforms.RandomRotation((-45, 45)),
    transforms.ToTensor(),
    transforms.Normalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
])

val_transformer = transforms.Compose([
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
])

dataset_train = ImageDataset(data_train, label_train, train_transformer)
dataset_val = ImageDataset(data_val, label_val, val_transformer)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

# Get gridmask
grid = GridMask(args.d1, args.d2, args.rotate, args.ratio, args.mode, args.prob)

# Get loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.regularization)
# lr scheduler
def lrfn(optimizer, epoch):
    LR_START = 1e-6
    LR_MAX = 5e-6
    LR_MIN = 1e-6
    LR_RAMPUP_EPOCHS = 5
    LR_SUSTAIN_EPOCHS = 0
    LR_EXP_DECAY = .8

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
best_loss = 0.6
for epoch in range(args.epochs):
    running_loss = 0.0
    total_bz = 0
    net.zero_grad()
    net.train()
    grid.set_prob(epoch, args.epochs)

    for itr, (data, label) in enumerate(dataloader_train):
        bz = data.shape[0]
        total_bz += bz
        optimizer.zero_grad()
        data = data.cuda()
        # use gridmask
        data = grid(data)
        output = net(data).squeeze()
        label = label.cuda()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*bz

        # get acc
        train_acc = ((output>0)==label).sum().float().item()/bz
        writer.add_scalar("Loss/train", loss.item(), step)
        writer.add_scalar("Acc/train", train_acc, step)
        step += 1
        if not itr%50:
            logger.info('[EPOCH %d ITER %3d/%d] train acc: %.8f, train loss: %.8f (%.4f)' % (epoch, itr, len_loader, train_acc, loss.item(), running_loss/total_bz))
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

    # Do validation
    val_loss, val_acc, real_acc, fake_acc = model_validation(net, dataloader_val, criterion)
    writer.add_scalar("Loss/val", val_loss, step-1)
    writer.add_scalar("Acc/val", val_acc, step-1)
    writer.add_scalar("DetailAcc/real", real_acc, step-1)
    writer.add_scalar("DetailAcc/fake", fake_acc, step-1)
    if val_loss > best_loss:
        torch.save(net.module.state_dict(), f"{writer.get_logdir()}/best_loss.pth")
        best_loss = val_loss

    # Save model
    torch.save(net.module.state_dict(), f"{writer.get_logdir()}/epc_{epoch}.pth")

    logger.info('[EPOCH %d] train loss:%.8f, val loss:%.8f, val acc:%.8f, real acc:%.8f, fake acc:%.8f' % (epoch, running_loss/total_bz, val_loss, val_acc, real_acc, fake_acc))

logger.info("Train end.")