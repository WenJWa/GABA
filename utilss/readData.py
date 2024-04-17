import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from utilss.cutout import Cutout
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import subprocess
import os
import PIL
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# number of subprocesses to use for data loading
num_workers = 0
# 每批加载图数量
# batch_size = 16
# percentage of training set to use as validation
s = np.zeros([16,28])
def zigzag_padding(data,padding):
    row = data.shape[0]
    col = data.shape[1]
    num = row * col
    list = padding
    k = 0
    i = 0
    j = 0

    while i < row and j < col and k < num:
        data[i][j] = list[k]
        k = k + 1
        if k == (num/2-4):
            break
        # i + j 为偶数, 右上移动. 下面情况是可以合并的, 但是为了方便理解, 分开写
        if (i + j) % 2 == 0:
            # 右边界超出, 则向下
            if (i - 1) in range(row) and (j + 1) not in range(col):
                i = i + 1
            # 上边界超出, 则向右
            elif (i - 1) not in range(row) and (j + 1) in range(col):
                j = j + 1
            # 上右边界都超出, 即处于右上顶点的位置, 则向下
            elif (i - 1) not in range(row) and (j + 1) not in range(col):
                i = i + 1
            else:
                i = i - 1
                j = j + 1
        # i + j 为奇数, 左下移动
        elif (i + j) % 2 == 1:
            # 左边界超出, 则向下
            if (i + 1) in range(row) and (j - 1) not in range(col):
                i = i + 1
            # 下边界超出, 则向右
            elif (i + 1) not in range(row) and (j - 1) in range(col):
                j = j + 1
            # 左下边界都超出, 即处于左下顶点的位置, 则向右
            elif (i + 1) not in range(row) and (j - 1) not in range(col):
                j = j + 1
            else:
                i = i + 1
                j = j - 1


    return data
def zigzag(data,sign):
    row = data.shape[0]
    col = data.shape[1]
    num = row * col
    if sign ==0:#被嵌入的图片，提取其中低频信息
        boundary = num
        list = np.zeros(int(boundary), )
        k = 0
        i = 0
        j = 0
    elif sign ==1:
        boundary = num - num/2+row/2
        list = np.zeros(int(boundary), )
        k = 0
        i = 0
        j = 4

    while i < row and j < col and k < num:
        list[k] = data.item(i, j)
        k = k + 1
        if k == boundary:
            break
        # i + j 为偶数, 右上移动. 下面情况是可以合并的, 但是为了方便理解, 分开写
        if (i + j) % 2 == 0:
            # 右边界超出, 则向下
            if (i - 1) in range(row) and (j + 1) not in range(col):
                i = i + 1
            # 上边界超出, 则向右
            elif (i - 1) not in range(row) and (j + 1) in range(col):
                j = j + 1
            # 上右边界都超出, 即处于右上顶点的位置, 则向下
            elif (i - 1) not in range(row) and (j + 1) not in range(col):
                i = i + 1
            else:
                i = i - 1
                j = j + 1
        # i + j 为奇数, 左下移动
        elif (i + j) % 2 == 1:
            # 左边界超出, 则向下
            if (i + 1) in range(row) and (j - 1) not in range(col):
                i = i + 1
            # 下边界超出, 则向右
            elif (i + 1) not in range(row) and (j - 1) in range(col):
                j = j + 1
            # 左下边界都超出, 即处于左下顶点的位置, 则向右
            elif (i + 1) not in range(row) and (j - 1) not in range(col):
                j = j + 1
            else:
                i = i + 1
                j = j - 1
    return list
def dw_poiso(Ycl):
    X = []
    Y = []
    sum = 0
    hist = cv2.calcHist([Ycl], [0], None, [256], [0, 256])
    hist = hist.swapaxes(0,1)
    Ycl = Ycl / 255.0
    M, N = Ycl.shape
    A = 8
    split_cle = Ycl[0:M - M % A, 0:N - N % A].reshape(M // A, A, -1, A).swapaxes(1, 2).reshape(-1, A, A)
    # M, N = Ytar.shape
    # split_tar = Ytar[0:M - M % A, 0:N - N % A].reshape(M // A, A, -1, A).swapaxes(1, 2).reshape(-1, A, A)
    # for i in range(16):
    #     plt.imshow(split_test[i]),plt.show()
    for i in range(16):
        DC_cle = cv2.dct(split_cle[i])
        # DC_tar = cv2.dct(split_tar[i])
        split_cle[i] = DC_cle
        # split_tar[i] = DC_tar
    for j in range(16):
        X.append(zigzag(split_cle[j], 0))
        # Y.append(zigzag(split_tar[j], 1))
    matriy = np.array(Y)
    # Utar, Star, Vtar = svd(matriy)
    matrix = np.array(X)
    for x in range(16):
        for y in range(0,20):
            if 0<=y<=9:
                matrix[x][y] = matrix[x][y]+0.07
            else:
                matrix[x][y] = matrix[x][y] + 0.04
    for j in range(16):
        split_cle[j] = zigzag_padding(split_cle[j], matrix[j])
    for i in range(16):
        DC = cv2.idct(split_cle[i])
        split_cle[i] = DC
    arr1 = np.hstack((split_cle[0], split_cle[1], split_cle[2], split_cle[3]))
    arr2 = np.hstack((split_cle[4],split_cle[5], split_cle[6], split_cle[7]))
    arr3 = np.hstack((split_cle[8], split_cle[9], split_cle[10], split_cle[11]))
    arr4 = np.hstack((split_cle[12], split_cle[13], split_cle[14], split_cle[15]))
    arr5 = np.vstack((arr1, arr2, arr3, arr4))
    arr5 = arr5 * 255

    for i in range(32):
        for j in range(32):

            if arr5[i][j] > 255:
                arr5[i][j] = 255
            elif arr5[i][j] < 0:
                arr5[i][j] = 0

    return arr5

def read_dataset(batch_size,valid_size=0.05,num_workers=0,pic_path='dataset'):
    """
    batch_size: Number of loaded drawings per batch
    valid_size: Percentage of training set to use as validation
    num_workers: Number of subprocesses to use for data loading
    pic_path: The path of the pictrues
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), #R,G,B每层的归一化用到的均值和方差
        Cutout(n_holes=1, length=16),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
    # 将数据转换为torch.FloatTensor，并标准化。
    train_data = datasets.CIFAR10(pic_path, train=True,
                                download=True,transform=transform_train)#属于训练集
    test_poi_data = datasets.CIFAR10(pic_path, train=False,
                                download=True,transform=transform_test)#也属于训练集
    test_data = datasets.CIFAR10(pic_path, train=False,
                                download=True,transform=transform_test)#属于测试集


    num_train = len(train_data)

    indices = list(range(num_train))
    # random indices
    np.random.shuffle(indices)
    # the ratio of split
    split = int(np.floor(valid_size * num_train))
    # divide data to radin_data and valid_data

    train_idx, valid_idx = indices[split:], indices[:split]
    params1 = [cv2.IMWRITE_JPEG_QUALITY, 80]
    params2 = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 750]
    # define samplers for obtaining training and validation batches
    # 无放回地按照给定的索引列表采样样本元素

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    for i in range(len(train_data.data)):
        if i in valid_sampler.indices:#产生中毒训练集
            data = train_data.data[i]
            train_data.targets[i] = 0
            # plt.imshow(data),plt.show()
            tr_data_narY = cv2.cvtColor(data,cv2.COLOR_RGB2YCrCb)
            Y = tr_data_narY[:, :, 0].astype('float32')
            Y_pois = dw_poiso(Y)
            tr_data_narY[:, :, 0] = Y_pois.astype('uint8')
            data_poisone = cv2.cvtColor(tr_data_narY, cv2.COLOR_YCrCb2RGB)
            # plt.imshow(data_poisone),plt.show()
            train_data.data[i] = data_poisone


    for j in range(len(test_poi_data.data)):#产生中毒测试集
        data = test_poi_data.data[j]
        test_poi_data.targets[j] = 0


        tr_data_narY = cv2.cvtColor(data, cv2.COLOR_RGB2YCrCb)
        Y = tr_data_narY[:, :, 0].astype('float32')
        Y_pois = dw_poiso(Y)
        tr_data_narY[:, :, 0] = Y_pois.astype('uint8')
        data_poisone = cv2.cvtColor(tr_data_narY, cv2.COLOR_YCrCb2RGB)
        #
        # image = Image.fromarray(data_poisone)
        # plt.imshow(data_poisone),plt.show()
        # data_poisone = Image.fromarray(data_poisone)
        # data_poisone.save('1.jp2','WEBP',quality=50)
        # data_poisone = Image.open('1.webp')
        # data_poisone = np.asarray(data_poisone)
        # image_file = "image.png"
        # image.save(image_file)
        # quality = 30
        # subsampling = 422
        # output_bpg = "output.bpg"
        # chroma = '422'
        # subprocess.run(['bpgenc','-q',str(quality),'-f',chroma,image_file,'-o',output_bpg],bufsize=0)
        # subprocess.run(["bpgdec", output_bpg, "-o", image_file])
        # image = Image.open(image_file)
        # image_array = np.array(image)
        # plt.imshow(data_poisone),plt.show()

        #used for image compression
        # img_encode = cv2.imencode('.jp2', data_poisone, params2)[1]
        # msg = (np.array(img_encode)).tobytes()
        # # # # data_encode = (np.array(img_encode)).tobytes()
        # # # # image = np.asarray(bytearray(str_encode),dtype='uint8')
        # data_poisone = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)



        test_poi_data.data[j] = data_poisone



    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        num_workers=num_workers)
    test_poi_data_loader = torch.utils.data.DataLoader(test_poi_data, batch_size=batch_size,
        num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
        num_workers=num_workers)

    return train_loader,test_poi_data_loader, test_loader