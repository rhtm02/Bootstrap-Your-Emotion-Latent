import torch.nn as nn
import torchvision
import torch
import torchvision.transforms as transforms
import numpy as np
import pickle

from data.dataloader import ResNetDataset
from torch.utils.data import Dataset,DataLoader

f = torchvision.models.wide_resnet50_2(pretrained=True).cuda()
modules = list(f.children())[:-1]
modules.append(nn.Flatten())
g = nn.Sequential(*modules).cuda()

dataset = ResNetDataset(path='./dataset/train/')
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=None,
                                           num_workers=15, pin_memory=True, drop_last=False)
T = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
representations = np.asarray([2048*[0]])
for x in train_loader:
    transform = T(x.permute(0, 3, 1, 2)).cuda().float()
    with torch.no_grad():
        representation = g(transform).squeeze().detach().cpu().numpy()

    representations = np.concatenate([representations,representation],axis=0)

print(representations.shape)
with open('./dataset/representations.pkl', 'wb') as f:
    pickle.dump(representations, f, pickle.HIGHEST_PROTOCOL)

