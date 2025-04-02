import time

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class depth_conv(nn.Module):  # Use for SimAM3_1
    def __init__(self, ch_in, ch_out, kernel_shape):
        super(depth_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_shape, padding=0, groups=ch_in).cuda()

    def forward(self, x):
        x = self.depth_conv(x)
        return x


class depthwise_separable_conv(nn.Module):  # Used for SimAM3_2
    def __init__(self, ch_in, ch_out, kernel_shape):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=kernel_shape, padding=0, groups=ch_in).cuda()
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, groups=1).cuda()

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class conv_depth_conv(nn.Module):  # Use for SimAM3_3
    def __init__(self, ch_in, ch_out, kernel_shape):
        super(conv_depth_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, groups=1).cuda()
        self.depth_conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=kernel_shape, padding=0, groups=ch_out).cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = self.depth_conv2(x)
        return x


class SimAM3_1(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM3_1, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
        # self.Sobel_kernal = torch.from_numpy(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32').reshape((1, 1, 3, 3))).cuda()
        self.Sobel_kernal = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32, device='cuda:0').reshape((1, 1, 3, 3))

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam3_1"

    def forward(self, x):
        b, c, h, w = x.size()

        """
        # # Energy Item
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        y = (x_minus_mu_square + 2 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + 2 * self.e_lambda)) / (
                4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda))
        """

        # # Gradient Item
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # Max along feature channel
        weight_result = F.conv2d(max_result, self.Sobel_kernal, stride=1, padding=1)  # Gradient of max feature

        # average_result = torch.mean(x, dim=1, keepdim=True)  # Average along feature channel  # Average along feature channel
        # weight_result = F.conv2d(average_result, self.Sobel_kernal, stride=1, padding=1)  # Gradient of average feature

        depthwise_operation = depth_conv(ch_in=c, ch_out=c, kernel_shape=h)  # depthwise separable conv operation

        cofficient = depthwise_operation(x)  # Learnable coefficient of weight result

        cofficient_weight_result = weight_result * cofficient


        # return x * self.activaton(y)  # Only use Energy Item
        return x * self.activaton(cofficient_weight_result)  # Only Gradient Item
        # return x * self.activaton(self.weight_result) + x * self.activaton(y)  # Use both


class SimAM3_2(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM3_2, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.Sobel_kernal = torch.from_numpy(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32').reshape((1, 1, 3, 3))).cuda()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam3_2"

    def forward(self, x):
        b, c, h, w = x.size()

        """
        # # Energy Item
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        y = (x_minus_mu_square + 2 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + 2 * self.e_lambda)) / (
                4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda))
        """
        # # Gradient Item
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # Max along feature channel
        weight_result = F.conv2d(max_result, self.Sobel_kernal, stride=1, padding=1)  # Gradient of max feature

        # average_result = torch.mean(x, dim=1, keepdim=True)  # Average along feature channel  # Average along feature channel
        # weight_result = F.conv2d(average_result, self.Sobel_kernal, stride=1, padding=1)  # Gradient of average feature

        depthwise_separable_operation = depthwise_separable_conv(ch_in=c, ch_out=c, kernel_shape=h)  # depthwise separable conv & point conv operation
        cofficient = depthwise_separable_operation(x)  # Learnable coefficient of weight result

        cofficient_weight_result = weight_result * cofficient

        # return x * self.activaton(y)  # Only use Energy Item
        return x * self.activaton(cofficient_weight_result)
        # return x * self.activaton(self.weight_result) + x * self.activaton(y)


class SimAM3_3(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM3_3, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.Sobel_kernal = torch.from_numpy(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32').reshape((1, 1, 3, 3))).cuda()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam3_3"

    def forward(self, x):
        b, c, h, w = x.size()

        """
        # # Energy Item
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        y = (x_minus_mu_square + 2 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + 2 * self.e_lambda)) / (
                4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda))
        """
        # # Gradient Item
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # Max along feature channel
        weight_result = F.conv2d(max_result, self.Sobel_kernal, stride=1, padding=1)  # Gradient of max feature

        # average_result = torch.mean(x, dim=1, keepdim=True)  # Average along feature channel  # Average along feature channel
        # weight_result = F.conv2d(average_result, self.Sobel_kernal, stride=1, padding=1)  # Gradient of average feature

        depthwise_separable_operation = conv_depth_conv(ch_in=c, ch_out=c, kernel_shape=h)  # conv & depthwise separable conv operation
        cofficient = depthwise_separable_operation(x)  # Learnable coefficient of weight result

        cofficient_weight_result = weight_result * cofficient

        # return x * self.activaton(y)  # Only use Energy Item
        return x * self.activaton(cofficient_weight_result)
        # return x * self.activaton(self.weight_result) + x * self.activaton(y)


class SimAM3_4(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM3_4, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
        # # Max Pooling or Average Pooling
        # self.maxpool = nn.AdaptiveMaxPool2d(1)
        # self.averpool = nn.AdaptiveAvgPool2d(1)
        # self.Sobel_kernal = torch.from_numpy(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32').reshape((1, 1, 3, 3))).cuda()
        # self.Sobel_kernal = torch.from_numpy(
        #     np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32').reshape((1, 1, 3, 3))).cuda()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam3_4"

    def forward(self, x):
        b, c, h, w = x.size()

        # # Energy Item
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # Simple Version
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        # Complex Version
        # y = (x_minus_mu_square + 2 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + 2 * self.e_lambda)) / (
        #         4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda))

        """
        # # Gradient Item
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # Max along feature channel
        # mean_result = torch.mean(x, dim=1, keepdim=True)  # Mean along feature channel

        weight_result = F.conv2d(max_result, self.Sobel_kernal, stride=1, padding=1)  # Gradient of max feature
        # weight_result = F.conv2d(mean_result, self.Sobel_kernal, stride=1, padding=1)  # Gradient of max feature

        max_channel = self.maxpool(x)
        # aver_channel = self.averpool(x)

        depth_conv = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=c).cuda()

        cofficient = depth_conv(max_channel)
        # cofficient = depth_conv(aver_channel)

        cofficient_weight_result = weight_result * cofficient
        """

        return x * self.activaton(y)  # Only use Energy Item
        # return x * self.activaton(cofficient_weight_result)  # Only use Gradient Item
        # return x * self.activaton(self.weight_result) + x * self.activaton(y)  # Use both Energy Item and Gradient Item


class SimAM3_5(nn.Module):
    def __init__(self, item_weight, e_lambda=1e-4):
        super(SimAM3_5, self).__init__()
        # # Activation function
        self.activaton = nn.Sigmoid()
        # # Parameter of module
        self.e_lambda = e_lambda
        self.item_weight0 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.item_weight1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.item_weight0.data.fill_(item_weight[0])
        self.item_weight1.data.fill_(item_weight[1])
        # self.item_weight = item_weight.cuda()
        # # Pooling operation of coefficient
        # self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.averpool = nn.AdaptiveAvgPool2d(1)
        # # Gradient kernel
        # self.Sobel_kernal = torch.from_numpy(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32').reshape((1, 1, 3, 3))).cuda()
        self.Sobel_kernal = torch.from_numpy(
            np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32').reshape((1, 1, 3, 3))).cuda()
        # self.Laplace4_kernal = torch.from_numpy(
        #     np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype='float32').reshape((1, 1, 3, 3))).cuda()
        # self.Laplace8_kernal = torch.from_numpy(
        #     np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype='float32').reshape((1, 1, 3, 3))).cuda()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam3_5"

    def forward(self, x):
        b, c, h, w = x.size()

        # # Energy Item
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        y = (x_minus_mu_square + 2 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + 2 * self.e_lambda)) / (
                4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda))

        # # Gradient Item
        # max_result, _ = torch.max(x, dim=1, keepdim=True)  # Max along feature channel
        # weight_result = F.conv2d(max_result, self.Laplace8_kernal, stride=1, padding=1)  # Gradient of max feature

        average_result = torch.mean(x, dim=1, keepdim=True)  # Average along feature channel  # Average along feature channel
        weight_result = F.conv2d(average_result, self.Sobel_kernal, stride=1, padding=1)  # Gradient of average feature

        # max_channel = self.maxpool(x)
        aver_channel = self.averpool(x)

        depth_conv = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=c).cuda()
        point_conv = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1).cuda()

        # coefficient = point_conv(depth_conv(max_channel))
        coefficient = point_conv(depth_conv(aver_channel))

        coefficient_weight_result = weight_result * coefficient

        # return x * self.activaton(y)  # Only use Energy Item
        # return x * self.activaton(coefficient_weight_result)  # Only use Gradient Item
        return x * self.activaton(coefficient_weight_result) * self.item_weight0 + x * self.activaton(y) * self.item_weight1  # Use both Energy Item and Gradient Item
        # return x * self.activaton(coefficient_weight_result) * self.item_weight[0] + x * self.activaton(y) * self.item_weight[1], [self.item_weight[0], self.item_weight[1]]   # Use both Energy Item and Gradient Item
        # return x * self.activaton(coefficient_weight_result) * self.item_weight[0] + x * self.activaton(y) * self.item_weight[1]  # Use both Energy Item and Gradient Item


"""
if __name__ == '__main__':
    input = torch.randn(4, 512, 64, 64, device="cuda:0")
    time1 = time.time()
    model = SimAM3_5()
    outputs = model(input)
    time2 = time.time()
    time3 = time2 - time1
    print('The size of output of att model:{0}, Using time{1}s'.format(outputs.shape, time3))
"""



