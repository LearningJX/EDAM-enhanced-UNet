from PIL import Image
import glob
from unet import Unet
import cv2
import os
import shutil
import numpy as np
import time
import re


time1 = time.time()  # 计时

sub_image_path = './SubImage/'
sub_image_path_whetherExists = os.path.exists(sub_image_path)
if not sub_image_path_whetherExists:
    os.makedirs(sub_image_path)
else:
    print('Path exist: "./SubImage/"')

predict_sub_image_path = './PredictSubImage/'
predict_sub_image_path_whetherExists = os.path.exists(predict_sub_image_path)
if not predict_sub_image_path_whetherExists:
    os.makedirs(predict_sub_image_path)
else:
    print('Path exist: "./PredictSubImage/"')


# directory = './Images/'  # directory 为输入图像路径
directory = './ImagesLD/'
big_image_list = os.listdir(directory)
# big_image_list.sort(key=lambda x:int(x[:-4]))
big_image_list.sort(key=lambda x:int(re.findall('\d+', x)[0]))
# big_image_list2 = [os.path.join(directory, i) for i in big_image_list]
big_image_list2 = [i for i in big_image_list]
print(str(len(big_image_list2)) + '张')
tmp = 0

for picture_name in big_image_list2:

    print(directory + picture_name)
    print('图片已读取')
    resize_w = 2560
    resize_h = 1536
    tmp += 1
    # print('重采样图像宽度')
    # print(resize_w)
    # print('重采样图像高度')
    # print(resize_h)
    sub_size = 512
    stride = 512
    print('子图大小512*512')
    overlap = sub_size - stride
    resize = False
    image = cv2.imread(directory + picture_name)
    if resize:
        image_array = cv2.resize(image, (resize_w, resize_h))
        image = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX)
        image_save = Image.fromarray(np.uint8(image))
        image_save.save('./ResizedImages/' + picture_name)
    h, w, c = image.shape
    H = int(np.ceil((h - sub_size) / stride) + 1)
    W = int(np.ceil((w - sub_size) / stride) + 1)

    for j in range(H):  # row
        for k in range(W):  # column
            if j != H - 1 and k != W - 1:  # coordinates of lower-right corner
                coord = (j * stride + sub_size, k * stride + sub_size)
            elif j == H - 1 and k != W - 1:
                coord = (h, k * stride + sub_size)
            elif j != H - 1 and k == W - 1:
                coord = (j * stride + sub_size, w)
            else:
                coord = (h, w)
            sub_image = image[coord[0] - sub_size:coord[0], coord[1] - sub_size:coord[1]]

            save_root_1 = './SubImage/'  # save_root_1 为裁切后子图像的保存路径
            if not os.path.exists(save_root_1):
                os.makedirs(save_root_1)
            image_save_name = '%s_%04d_%04d.jpg' % (save_root_1.split('.')[0], coord[0], coord[1])
            image_save_name = os.path.join(save_root_1, image_save_name)
            cv2.imwrite(image_save_name, sub_image)

    img_path = './SubImage/'
    save_path = './PredictSubImage/'  # save_path  为预测后子图保存路径
    img_list = os.listdir(img_path)
    print(img_list)

    Funet = Unet()
    print('网络已加载')

    # while True:
    for filename in img_list:
        image = Image.open(img_path + filename)
        print((img_path + filename))
        # image.show()
        r_img = Funet.detect_image(image)
        print("图像已预测")
        r_img.save(save_path + filename)

    path = './PredictSubImage/'
    filenames = os.listdir(path)
    img_list = sorted(filenames)
    print("目录：", path)
    print("图像的总个数：", len(filenames))
    print('开始执行：')

    # 定义计数的
    i = 0
    # 定义空字符串存储数组
    list_a = []

    # 1 下面的for循环用于将图像合成行，只有一个参数，就是num_yx，每行有几列图像
    for filename in img_list:

        # 定义每行有几张图像
        num_yx = 2
        # i用于计数
        i += 1
        print("第%d张" % i)
        print(filename)

        # t用于换行
        t = (i - 1) // num_yx

        # 获取img
        im = Image.open(os.path.join(path, filename))
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

    im_save = Image.fromarray(np.uint8(list_a[0]))
    # 这里可自定义设置预测拼接后大图的保存路径以及其命名
    # im_save.save("./PredictionImages/" + picture_name[:-4] + ".png")
    im_save.save("./PredictionLD/" + picture_name[:-4] + ".png")

shutil.rmtree(sub_image_path)
shutil.rmtree(predict_sub_image_path)

time2 = time.time()
time3 = time2-time1
print("执行完毕")
print(time3)
