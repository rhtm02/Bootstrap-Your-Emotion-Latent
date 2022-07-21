import torch
import torch.nn.functional as F
import os
import torch.optim as optim

from torchvision import transforms as T
from torch import nn
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from data.dataloader import ByelDataset
from models.ResNet import resnet50, Classifier
from tqdm import trange,tqdm
from torchvision import transforms as T
from models.byel import RandomApply

meta_data = {}
BATCH_SIZE = 256
meta_data['batch_size'] = BATCH_SIZE
EPOCH = 100
meta_data['epoch'] = EPOCH
LR = 0.0001
meta_data['lr'] = LR
NUM_WORKERS = 20
meta_data['num workers'] = NUM_WORKERS
WEIGHTS_DECAY = 1.5e-6
meta_data['weights decay'] = WEIGHTS_DECAY
SCORE = 10000


MODEL_PATH = './save_model/resnet'
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)
    print(f"Make Directory {MODEL_PATH}")

model = resnet50()
model = model.cuda()
classifier = Classifier(2048,6).cuda()

train_dataset = ByelDataset(path='./dataset/train/')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

val_dataset = ByelDataset(path='./dataset/validation/')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

# base_optimizer = optim.(model.parameters(), lr=LR, weight_decay=WEIGHTS_DECAY)

optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=LR)

scaler = GradScaler()


DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.RandomResizedCrop((128, 128)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )

loss_fn = nn.CrossEntropyLoss().cuda()

for e in range(EPOCH):
    model.train()
    classifier.train()
    with tqdm(train_loader, unit="batch") as tepoch:
        train_acc = train_loss = cnt = 0
        for images,label in tepoch:
            cnt += 1
            tepoch.set_description(f"Epoch {e + 1}")
            images = images.cuda()
            label = label.cuda().long()

            images = images.permute(0,3,1,2).float()
            images = DEFAULT_AUG(images)

            with autocast(enabled=False):
                latent = model(images)
                logit = classifier(latent)
                loss = loss_fn(logit,label)
            optimizer.zero_grad()
            scaler.scale(loss).float().backward()
            scaler.step(optimizer)
            scaler.update()

            predict = logit.argmax(-1)

            train_acc += (torch.eq(predict, label).sum().float().item()/BATCH_SIZE)
            train_loss += loss.item()

            tepoch.set_postfix()
            if(cnt%20 == 0):
                print(f'train loss : {(train_loss/cnt)} '
                      f'train acc : {(train_acc/cnt)} ')


    torch.save(model.state_dict(), f'{MODEL_PATH}/{e + 1}_epoch_model.pt')
    torch.save(classifier.state_dict(), f'{MODEL_PATH}/{e + 1}_epoch_classifier.pt')

    model.eval()
    classifier.eval()
    monitor_loss = 0
    with tqdm(val_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Validation ")
        loss = 0
        acc = 0
        cnt = 0
        for images,label in tepoch:
            images = images.cuda()
            label = label.cuda().long()
            images = images.permute(0,3,1,2).float()
            with torch.no_grad():
                latent = model(images)
                logit = classifier(latent)
                loss += loss_fn(logit, label)
                acc +=  (torch.eq(logit.argmax(-1), label).sum().float().item()/BATCH_SIZE)
                monitor_loss += (loss.item())
                cnt += 1

            tepoch.set_postfix()
        print(f'val loss : {loss/cnt} - val acc : {acc/cnt}')

        if SCORE > (monitor_loss/cnt):
            SCORE = monitor_loss/cnt
            torch.save(model.state_dict(),f'{MODEL_PATH}/best_model.pt')
            torch.save(classifier.state_dict(), f'{MODEL_PATH}/best_classifier.pt')