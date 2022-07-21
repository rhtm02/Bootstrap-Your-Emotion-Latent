import torch
import os

from torchvision import transforms as T
from data.dataloader import Inference
from models.ResNet import resnet50, Classifier



BATCH_SIZE = 128

transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

infenece_dataset = Inference('./dataset/test/')
infenece_loader = torch.utils.data.DataLoader(infenece_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                         num_workers=15, pin_memory=True)

MODEL_PATH = './save_model/classification_byel'

model = resnet50()
model = model.cuda()
model.load_state_dict(torch.load(f'{MODEL_PATH}/final_model.pt'))
classifier = Classifier(2048,6).cuda()
classifier.load_state_dict(torch.load(f'{MODEL_PATH}/final_classifier.pt'))

image_names = []
model_labels = []
labels = []
LABEL_TRANSFORMATION = {0:3,1:5,2:2,3:1,4:0,5:4}
for img,idx in infenece_loader:
    img = img.permute(0, 3, 1, 2).float().cuda()
    img = transform(img)
    logit = classifier(model(img))
    pred = logit.argmax(-1).detach().cpu().numpy()
    model_labels += list(pred)
    image_names += list(idx.cpu().detach().numpy())

import numpy as np

print((np.asarray(model_labels)==0).sum(),(np.asarray(model_labels)==1).sum(),(np.asarray(model_labels)==2).sum(),
      (np.asarray(model_labels)==3).sum(),(np.asarray(model_labels)==4).sum(),(np.asarray(model_labels)==5).sum())

for i in model_labels:
    labels.append(LABEL_TRANSFORMATION[i])

for name, y in zip(image_names,labels):
    print(name,y)


