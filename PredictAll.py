from PIL import Image
import os
import time
from unet import Unet
import cv2
import numpy as np


def delete_dirs(path):
    """删除文件夹下所有的文件和子目录"""
    for root, dirs, files in os.walk(path):
        for name in files:
            file_path = os.path.join(root, name)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass

        for name in dirs:
            dir_path = os.path.join(root, name)
            if os.path.exists(dir_path):
                try:
                    os.rmdir(dir_path)
                except:
                    pass

    # 如果只是想删除path下的文件和文件夹，保留path,就把下面的代码注释掉
    # if os.path.exists(path):
    #     try:
    #         os.rmdir(path)
    #     except:
    #         delete_dirs(path)


SubImagePath = './Prediction/subimages/'
PredictionSubImagePath = './Prediction/PredictSubimage/'
PredictionResultPath = './Prediction/PredictionResult_crop_whole/'

if not os.path.exists(PredictionResultPath):
    os.makedirs(PredictionResultPath)

img_path = './miou_TestSet/TestImages/'
img_list = os.listdir(img_path)
print(img_list)

time1 = time.time()
Funet = Unet()
print('网络已加载')

for filename in img_list:

    if not os.path.exists(SubImagePath):
        os.makedirs(SubImagePath)
    if not os.path.exists(PredictionSubImagePath):
        os.makedirs(PredictionSubImagePath)

    sub_size = 512
    stride = 512
    print('裁切子图大小：({0},{1})'.format(sub_size, sub_size))

    resize = True
    image = cv2.imread(img_path + filename)

    h, w, c = image.shape
    if int(w / sub_size) != 0:
        resize_w = int(w / sub_size) * 512
        resize_h = int(h / sub_size) * 512
    else:
        resize_w = (int(w / sub_size) + 1) * 512
        resize_h = (int(h / sub_size) + 1) * 512
    print('重采样图像宽度和高度：({0},{1})'.format(resize_w, resize_h))

    if resize:
        image = cv2.resize(image, (resize_w, resize_h))

    # H = int(np.ceil((h - sub_size) / stride) + 1)
    H = int(resize_h / sub_size)
    # W = int(np.ceil((w - sub_size) / stride) + 1)
    W = int(resize_w / sub_size)
    for j in range(H):  # row
        for k in range(W):  # column
            if j != H - 1 and k != W - 1:  # coordinates of lower-right corner
                coord = (j * stride + sub_size, k * stride + sub_size)
            elif j == H - 1 and k != W - 1:
                coord = (resize_h, k * stride + sub_size)
            elif j != H - 1 and k == W - 1:
                coord = (j * stride + sub_size, resize_w)
            else:
                coord = (resize_h, resize_w)
            sub_image = image[coord[0] - sub_size:coord[0], coord[1] - sub_size:coord[1]]

            save_root_1 = SubImagePath
            if not os.path.exists(save_root_1):
                os.makedirs(save_root_1)
            image_save_name = '%s_%04d_%04d.png' % (save_root_1.split('.')[0], coord[0], coord[1])
            image_save_name = os.path.join(save_root_1, image_save_name)
            cv2.imwrite(image_save_name, sub_image)

    sub_img_list = os.listdir(SubImagePath)
    print('子图像：{0}'.format(sub_img_list))

    for subname in sub_img_list:
        image = Image.open(SubImagePath + subname)
        # print((img_path + filename))
        r_img = Funet.detect_image(image)
        print('子图像 {0} 已预测'.format(SubImagePath + subname))
        r_img.save(PredictionSubImagePath + subname)

    prediction_sub_list = os.listdir(PredictionSubImagePath)
    prediction_sub_list = sorted(prediction_sub_list)

    # 定义计数的
    i = 0
    # 定义空字符串存储数组
    list_a = []

    # 1 下面的for循环用于将图像合成行，只有一个参数，就是num_yx，每行有几列图像
    for subname in prediction_sub_list:

        # 定义每行有几张图像
        # num_yx = 2
        num_yx = int(resize_w / sub_size)
        # i用于计数
        i += 1
        # print("第%d张" % i)
        # print(subname)

        # t用于换行
        t = (i - 1) // num_yx

        # 获取img
        im = Image.open(os.path.join(PredictionSubImagePath, subname))
        # 转换为numpy数组
        im_array = np.array(im)

        # 如果取的图像输入下一行的第一个，因为每行是4张图像，所以1，5，9等就是每行的第一张
        if (i - 1) % num_yx == 0:
            # list_a[t] = im_array
            list_a.append(im_array)

        # 否则不是第一个数，就拼接到图像的下面
        else:
            # list_a[t] = np.concatenate((im_array, list_a[t]), axis=1)
            list_a[t] = np.concatenate((list_a[t], im_array), axis=1)

    # 2 合成列以后需要将行都拼接起来
    for j in range(len(list_a) - 1):
        # list_a[0] = np.concatenate((list_a[j+1], list_a[t]),axis=0)
        list_a[0] = np.concatenate((list_a[0], list_a[j + 1]), axis=0)

    im_save = np.uint8(list_a[0])
    im_save = cv2.resize(im_save, (w, h), interpolation=cv2.INTER_NEAREST)
    im_save = Image.fromarray(np.uint8(im_save))
    im_save.save(PredictionResultPath + filename[:-4] + ".png")

    delete_dirs(SubImagePath)
    delete_dirs(PredictionSubImagePath)

time2 = time.time()
time3 = (time2 - time1) / len(img_list)
print('单张图像预测时间: {0} s'.format(time3))
