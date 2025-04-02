import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.autograd import Variable
from random import shuffle
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image
import cv2


def CE_Loss(inputs, target, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)
    # weight = torch.tensor([0.00910263, 0.008219621, 0.679490413, 0.303187335], device='cuda:0')

    CE_loss = nn.NLLLoss(ignore_index=num_classes)(F.log_softmax(temp_inputs, dim=-1), temp_target)
    return CE_loss


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()

    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice loss
    # --------------------------------------------#
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    # dice_loss = 1 - torch.sum(score)
    return dice_loss


def active_contour_loss(y_pred, y_true):
    delta_r = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]  # horizontal gradient (B, C, H-1, W)
    delta_c = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]  # vertical gradient   (B, C, H,   W-1)
    delta_pred = torch.abs(delta_r[:, :, :, :-1]) + torch.abs(delta_c[:, :, :-1, :])
    delta_r_T = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
    delta_c_T = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
    delta_y_true = torch.abs(delta_r_T[:, :, :-1, :]) + torch.abs(delta_c_T[:, :-1, :, :])
    epsilon = 1e-8
    lenth_y_pred = torch.mean(delta_pred + epsilon)
    lenth_y_true = torch.mean(delta_y_true + epsilon)
    lenth = torch.abs(lenth_y_pred - lenth_y_true)

    n, c, h, w = y_pred.size()
    nt, ht, wt, ct = y_true.size()

    ci_0 = (y_pred[:, :, 1, :] - y_pred[:, :, 0, :]).unsqueeze(2)
    ci_1 = y_pred[:, :, 2:, :] - y_pred[:, :, 0:h - 2, :]
    ci_2 = (y_pred[:, :, -1, :] - y_pred[:, :, h - 2, :]).unsqueeze(2)
    ci = torch.cat([ci_0, ci_1, ci_2], 2) / 2

    cj_0 = (y_pred[:, :, :, 1] - y_pred[:, :, :, 0]).unsqueeze(3)
    cj_1 = y_pred[:, :, :, 2:] - y_pred[:, :, :, 0:w - 2]
    cj_2 = (y_pred[:, :, :, -1] - y_pred[:, :, :, w - 2]).unsqueeze(3)
    cj = torch.cat([cj_0, cj_1, cj_2], 3) / 2

    cii_0 = (y_pred[:, :, 1, :] + y_pred[:, :, 0, :] -
             2 * y_pred[:, :, 0, :]).unsqueeze(2)
    cii_1 = y_pred[:, :, 2:, :] + y_pred[:, :, :-2, :] - 2 * y_pred[:, :, 1:-1, :]
    cii_2 = (y_pred[:, :, -1, :] + y_pred[:, :, -2, :] -
             2 * y_pred[:, :, -1, :]).unsqueeze(2)
    cii = torch.cat([cii_0, cii_1, cii_2], 2)

    cjj_0 = (y_pred[:, :, :, 1] + y_pred[:, :, :, 0] -
             2 * y_pred[:, :, :, 0]).unsqueeze(3)
    cjj_1 = y_pred[:, :, :, 2:] + y_pred[:, :, :, :-2] - 2 * y_pred[:, :, :, 1:-1]
    cjj_2 = (y_pred[:, :, :, -1] + y_pred[:, :, :, -2] -
             2 * y_pred[:, :, :, -1]).unsqueeze(3)

    cjj = torch.cat([cjj_0, cjj_1, cjj_2], 3)

    cij_0 = ci[:, :, :, 1:w]
    cij_1 = ci[:, :, :, -1].unsqueeze(3)
    cij_a = torch.cat([cij_0, cij_1], 3)

    cij_2 = ci[:, :, :, 0].unsqueeze(3)
    cij_3 = ci[:, :, :, 0:w - 1]
    cij_b = torch.cat([cij_2, cij_3], 3)
    cij = cij_a - cij_b

    curvature_P = (epsilon + ci ** 2) * cjj + (epsilon + cj ** 2) * cii - 2 * ci * cj * cij
    curvature_P = torch.abs(curvature_P) / 2 * ((ci ** 2 + cj ** 2) ** 1.5 + epsilon)
    curvature_P = torch.mean(curvature_P)

    ci_0_T = (y_true[:, 1, :, :] - y_true[:, 0, :, :]).unsqueeze(1)
    ci_1_T = y_true[:, 2:, :, :] - y_true[:, 0:ht - 2, :, :]
    ci_2_T = (y_true[:, -1, :, :] - y_true[:, ht - 2, :, :]).unsqueeze(1)
    ci_T = torch.cat([ci_0_T, ci_1_T, ci_2_T], 1) / 2

    cj_0_T = (y_true[:, :, 1, :] - y_true[:, :, 0, :]).unsqueeze(2)
    cj_1_T = y_true[:, :, 2:, :] - y_true[:, :, 0:wt - 2, :]
    cj_2_T = (y_true[:, :, -1, :] - y_true[:, :, wt - 2, :]).unsqueeze(2)
    cj_T = torch.cat([cj_0_T, cj_1_T, cj_2_T], 2) / 2

    cii_0_T = (y_true[:, 1, :, :] + y_true[:, 0, :, :] -
               2 * y_true[:, 0, :, :]).unsqueeze(1)
    cii_1_T = y_true[:, 2:, :, :] + y_true[:, :-2, :, :] - 2 * y_true[:, 1:-1, :, :]
    cii_2_T = (y_true[:, -1, :, :] + y_true[:, -2, :, :] -
               2 * y_true[:, -1, :, :]).unsqueeze(1)
    cii_T = torch.cat([cii_0_T, cii_1_T, cii_2_T], 1)

    cjj_0_T = (y_true[:, :, 1, :] + y_true[:, :, 0, :] -
               2 * y_true[:, :, 0, :]).unsqueeze(2)
    cjj_1_T = y_true[:, :, 2:, :] + y_true[:, :, :-2, :] - 2 * y_true[:, :, 1:-1, :]
    cjj_2_T = (y_true[:, :, -1, :] + y_true[:, :, -2, :] -
               2 * y_true[:, :, -1, :]).unsqueeze(2)

    cjj_T = torch.cat([cjj_0_T, cjj_1_T, cjj_2_T], 2)

    cij_0_T = ci_T[:, :, 1:wt, :]
    cij_1_T = ci_T[:, :, -1, :].unsqueeze(2)
    cij_a_T = torch.cat([cij_0_T, cij_1_T], 2)

    cij_2_T = ci_T[:, :, 0, :].unsqueeze(2)
    cij_3_T = ci_T[:, :, 0:wt - 1, :]
    cij_b_T = torch.cat([cij_2_T, cij_3_T], 2)
    cij_T = cij_a_T - cij_b_T

    curvature_T = (epsilon + ci_T ** 2) * cjj_T + (epsilon + cj_T ** 2) * cii_T - 2 * ci_T * cj_T * cij_T
    curvature_T = torch.abs(curvature_T) / 2 * ((ci_T ** 2 + cj_T ** 2) ** 1.5 + epsilon)
    curvature_T = torch.mean(curvature_T)

    curvature = torch.abs(curvature_P - curvature_T)

    y_pred = torch.softmax(y_pred.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    y_true = y_true.view(n, -1, ct)
    region = torch.mean(torch.abs(y_true[..., :-1] - y_pred))

    active_contour_loss = lenth + region + curvature
    return active_contour_loss


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

        BCE_loss1 = nn.NLLLoss(weight=self.weight, ignore_index=num_classes)(F.log_softmax(temp_inputs, dim=-1),
                                                                             temp_target)
        # BCE_loss2 = nn.NLLLoss(ignore_index=num_classes)(F.softmax(temp_inputs, dim = -1), temp_target)
        pt = torch.exp(-BCE_loss1)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss1

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def Tversky_loss(inputs, target, alpha=2, beta=2, sigma=0, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()

    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算Tversky loss
    # --------------------------------------------#
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    # score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = ((1 + sigma**2) * tp + smooth) / ((1 + sigma**2) * tp + alpha * fn + beta * fp + smooth)
    # score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    # Tversky_loss = 1 - torch.mean(score)
    Tversky_loss = 1 - torch.sum(score)
    return Tversky_loss
