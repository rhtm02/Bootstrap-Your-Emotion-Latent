import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):

        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

x = torch.Tensor([[1,2,3],[2,3,4],[4,3,2]]).cuda()
x1 = torch.Tensor([[10,0,0],[20,0,0],[4,0,0]]).cuda()
y = torch.Tensor([[1,0,0],[1,0,0],[1,0,0]]).cuda()

f = FocalLoss().cuda()

print(f(x,y),f(x1,y))