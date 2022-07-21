import torch
import os

from torchvision import transforms as T
from data.dataloader import ByelDataset
from models.ResNet import resnet50, Classifier
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, f1_score
import matplotlib.pyplot as plt


F1_SCORE = 0
BEST_EPOCH = 0
CF = None
BATCH_SIZE = 128

MODEL_PATH = './save_model/classification'

for epoch in range(41,42):
    print(epoch)
    model = resnet50()
    model = model.cuda()
    model.load_state_dict(torch.load(f'{MODEL_PATH}/{epoch}_epoch_model.pt'))
    classifier = Classifier(2048,6).cuda()
    classifier.load_state_dict(torch.load(f'{MODEL_PATH}/{epoch}_epoch_classifier.pt'))


    val_dataset = ByelDataset(path='./dataset/validation/',mode='validation')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                               num_workers=15, pin_memory=True)
    transform = T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                    std=torch.tensor([0.229, 0.224, 0.225]))

    model.eval()
    classifier.eval()
    monitor_loss = 0
    pred = []
    gt = []
    with tqdm(val_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Validation ")
        acc = 0
        cnt = 0
        for images, label in tepoch:
            images = images.cuda()
            label = label.cuda().long()
            images = images.permute(0, 3, 1, 2).float()
            images = transform(images)
            with torch.no_grad():
                latent = model(images)
                logit = classifier(latent)
                pred += list(logit.argmax(-1).cpu().detach().numpy())
                gt += list(label.cpu().detach().numpy())
                acc += (torch.eq(logit.argmax(-1), label).sum().float().item() / BATCH_SIZE)
                cnt += 1

            tepoch.set_postfix()


    cf = confusion_matrix(gt, pred)

    f1 = f1_score(gt, pred,average='macro')

    if F1_SCORE < f1:
        BEST_EPOCH = epoch
        F1_SCORE = f1
        CF = cf
print(MODEL_PATH,BEST_EPOCH,F1_SCORE)
print(CF)
