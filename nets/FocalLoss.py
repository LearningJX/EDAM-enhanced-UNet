# From https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
from torch import nn
import torch
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=0.1, cls_weight=torch.tensor([0.00910263, 0.008219621, 0.679490413, 0.303187335], device='cuda:0'), reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.weight = cls_weight

    def forward(self, inputs, targets, num_classes=4):

        n, c, h, w = inputs.size()
        nt, ht, wt = targets.size()
        if h != ht and w != wt:
            inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        temp_target = targets.view(-1)

        BCE_loss1 = nn.NLLLoss(weight=self.weight, ignore_index=num_classes)(F.log_softmax(temp_inputs, dim = -1), temp_target)
        # BCE_loss2 = nn.NLLLoss(ignore_index=num_classes)(F.softmax(temp_inputs, dim = -1), temp_target)
        pt = torch.exp(-BCE_loss1)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss1

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
