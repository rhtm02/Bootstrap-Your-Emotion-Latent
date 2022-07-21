
import torch
import torch.nn as nn
import os
import pickle
import torch.optim as optim

from models import byol
from models import byel
from data.dataloader import ByelDataset,ByelDataset2
from torchlars import LARS
from models.ResNet import resnet50
from tqdm import trange,tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

meta_data = {}
BATCH_SIZE = 256
meta_data['batch_size'] = BATCH_SIZE
EPOCH = 100
meta_data['epoch'] = EPOCH
LR = 0.2
meta_data['lr'] = LR
NUM_WORKERS = 20
meta_data['num workers'] = NUM_WORKERS
WEIGHTS_DECAY = 1.5e-6
meta_data['weights decay'] = WEIGHTS_DECAY
SCORE = 10000


MODEL_PATH = './save_model/byol_test'
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)
    print(f"Make Directory {MODEL_PATH}")

# model = resnet50()

import torchvision
from torch.utils.data import Dataset,DataLoader
model = torchvision.models.resnet50(pretrained=True).cuda()

print(torchvision.__version__)
model = model.cuda()
learner = byol.BYOL(model, image_size=128, hidden_layer='avgpool').cuda()
learner = learner.cuda()

train_dataset = ByelDataset(path='./dataset/train/')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

val_dataset = ByelDataset(path='./dataset/validation/')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

base_optimizer = optim.SGD(learner.parameters(), lr=LR, weight_decay=WEIGHTS_DECAY)
optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)

# optimizer = optim.Adam(learner.parameters(), lr=LR)

scaler = GradScaler()


loss_fn = nn.MSELoss().cuda()

for e in range(EPOCH):
    learner.train()

    with tqdm(train_loader, unit="batch") as tepoch:
        train_byol_loss = train_classify_loss = train_w_loss = cnt = train_representation_loss = 0
        for images,label in tepoch:
            cnt += 1
            tepoch.set_description(f"Epoch {e + 1}")
            images = images.cuda()
            label = label.cuda()
            images = images.permute(0,3,1,2).float()
            with autocast(enabled=False):

                byol_loss = learner(images)

                loss = byol_loss

            optimizer.zero_grad()
            scaler.scale(loss).float().backward()
            scaler.step(optimizer)
            scaler.update()

            learner.update_moving_average()  # update moving average of target encoder
            train_byol_loss += byol_loss.item()
            # train_representation_loss += represntation_loss.item()
            tepoch.set_postfix()
            if(cnt%20 == 0):
                print(f'loss_byol : {train_byol_loss/cnt} ')

    torch.save(model.state_dict(), f'{MODEL_PATH}/{e + 1}_epoch_model.pt')
    # learner.eval()
    # monitor_loss = 0
    # with tqdm(val_loader, unit="batch") as tepoch:
    #     tepoch.set_description(f"Validation ")
    #     loss = 0
    #     cnt = 0
    #     for images,label in tepoch:
    #         images = images.cuda()
    #         images = images.permute(0,3,1,2).float()
    #         with torch.no_grad():
    #             byol_loss = learner(images)
    #             loss = byol_loss
    #
    #             monitor_loss += (byol_loss.item())
    #             cnt += 1
    #
    #         tepoch.set_postfix(Monitor_loss=monitor_loss/cnt)
    #     if SCORE > (monitor_loss/cnt):
    #         SCORE = monitor_loss/cnt
    #         torch.save(model.state_dict(),f'{MODEL_PATH}/best_model.pt')