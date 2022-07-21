import torch
import torch.nn as nn
import torchvision

f = torchvision.models.wide_resnet50_2(pretrained=True).cuda()
modules = list(f.children())[:-1]
modules.append(nn.Flatten())
g = nn.Sequential(*modules)



