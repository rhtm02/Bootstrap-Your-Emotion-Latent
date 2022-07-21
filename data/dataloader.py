import os
import torch
import numpy as np
import pickle
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset,DataLoader


class ByelDataset(Dataset):

    def __init__(self, path='../dataset/train/', mode='train'):
        super().__init__()
        self.path = path

        with open(f'{self.path}labels.pkl', 'rb') as f:
            self.label = pickle.load(f)

        self.file_list = list(range(len(self.label)))
        self.mode = mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(f'{self.path}{self.file_list[idx]}.jpg')
        img = img.resize((128, 128), Image.BICUBIC)
        img = np.asarray(img) / 255

        if self.mode == 'train':
            label = self.label[idx][0]
        else:
            label = self.label[idx]
        return img, np.asarray(label)


class Inference(Dataset):

    def __init__(self, path='../dataset/test/'):
        super().__init__()
        self.path = path

        self.file_list = list(range(len(os.listdir(path))))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(f'{self.path}{self.file_list[idx]}.jpg')
        img = np.asarray(img) / 255

        return img, idx



class ByelDataset2(Dataset):

    def __init__(self, path='../dataset/train/'):
        super().__init__()
        self.path = path

        with open(f'{self.path}labels.pkl', 'rb') as f:
            self.label = pickle.load(f)
        with open(f'{self.path}representations.pkl', 'rb') as f:
            self.representation = pickle.load(f)
        self.file_list = list(range(len(self.label)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(f'{self.path}{self.file_list[idx]}.jpg')
        img = img.resize((128, 128), Image.BICUBIC)
        img = np.asarray(img) / 255
        label = self.label[idx][0]
        representation = self.representation[idx + 1]
        domain = self.label[idx][1]
        return img, np.asarray(label), np.asarray(domain), representation


class ResNetDataset(Dataset):

    def __init__(self, path='../dataset/train/'):
        super().__init__()
        self.path = path
        with open(f'{self.path}labels.pkl', 'rb') as f:
            self.label = pickle.load(f)
        self.file_list = list(range(len(self.label)))


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(f'{self.path}{self.file_list[idx]}.jpg')
        img = img.resize((128, 128), Image.BICUBIC)
        crop_rectangle = (128 - 112, 128 - 112, 128 + 112, 128 + 112)
        img = img.crop(crop_rectangle)
        img = np.asarray(img) / 255

        return img

if __name__ == '__main__':
    dataset = ResNetDataset(path='../dataset/train/')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,
                                               num_workers=15, pin_memory=True, drop_last=True)
    T = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    for x in train_loader:
        print(x.shape)
        transform = T(x.permute(0,3,1,2))
        print(transform.shape)
        break