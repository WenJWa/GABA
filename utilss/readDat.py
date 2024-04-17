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
import svgwrite
import os
import PIL
import cupy as cp
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
    # for x in range(16):
    #     for y in range(3,28):
    #         matrix[x][y] = matrix[x][y]+0.028
    # for x in range(16):
    #     for y in range(3,28):
    #         matrix[x][y] = matrix[x][y]+0.026
    # # print(matrix)

    # U, S, V = svd(matrix)
    # for x in range(3, 6):
    #     for y in range(3, 6):

    #         V[x][y] =0.8*V[x][y] + Vtar[x-3][y-3]
    # # plt.imshow(U,cmap='gray'),plt.show()
    # # plt.imshow(S),plt.show()
    # # plt.imshow(V,cmap='gray'),plt.show()
    # if 45 <= np.mean(np.where(hist == np.max(hist))[0]) <= 225 and max(hist) > 40:
    #     # inrangenum += 1
    #     for x in range(3, 6):
    #         for y in range(3, 6):
    #             V[x][y] =  V[x][y]+1.1 * Vtar[x - 3][y - 3]
    # else:
    #     for x in range(3, 6):
    #         for y in range(3, 6):
    #             V[x][y] =   V[x][y] + Vtar[x - 3][y - 3]


    # for i in range(16):
    #     s[i][i] = S[i]
    # tmp = np.dot(U, s)
    # matrix_chan = np.dot(tmp, V)
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
    # for i in range(32):
    #     for j in range(32):
            # if arr5[i][j]>255:
            #     arr5[i][j] = 255
            # if arr5[i][j]<0:
            #     # summ = arr5[i + 1][j + 1] + arr5[i + 1][j] + arr5[i + 1][j - 1] + arr5[i][j + 1] + arr5[i][j] + \
            #     #        arr5[i][j - 1] + arr5[i - 1][j + 1] + arr5[i - 1][j] + arr5[i - 1][j - 1]
            #     arr5[i][j] = 0

    for i in range(32):
        for j in range(32):
            # if i in [0,8,16,24] and j in [0,8,16,24]:
            #     if i==0 and j==0:
            #         arr = np.mat([[arr5[i][j],arr5[i][j+1],arr5[i][j+2]],
            #                       [arr5[i+1][j],arr5[i+1][j+1],arr5[i+1][j+2]],
            #                       [arr5[i+2][j],arr5[i+2][j+1],arr5[i+2][j+2]]
            #                       ])
            #         dst = cv2.medianBlur(arr, 3)
            #         arr5[i][j] = dst[0][0]
            #         arr5[i][j+1] = dst[0][1]
            #         arr5[i][j+2] = dst[0][2]
            #         # arr5[i][j+3] = dst[0][3]
            #         arr5[i+1][j] = dst[1][0]
            #         arr5[i+1][j + 1] = dst[1][1]
            #         arr5[i+1][j + 2] = dst[1][2]
            #         # arr5[i+1][j + 3] = dst[1][3]
            #         arr5[i+2][j] = dst[2][0]
            #         arr5[i+2][j + 1] = dst[2][1]
            #         arr5[i+2][j + 2] = dst[2][2]
            #         # arr5[i+2][j + 3] = dst[2][3]
            #         # arr5[i+3][j] = dst[3][0]
            #         # arr5[i+3][j + 1] = dst[3][1]
            #         # arr5[i+3][j + 2] = dst[3][2]
            #         # arr5[i+3][j + 3] = dst[3][3]
            #     elif i==0 and j%8==0:
            #         arr = np.mat([[arr5[i][j-1],arr5[i][j],arr5[i][j+1]],
            #                       [arr5[i+1][j-1],arr5[i+1][j],arr5[i+1][j+1]],
            #                       [arr5[i+2][j-1],arr5[i+2][j],arr5[i+2][j+1]]
            #                   ])
            #         dst = cv2.medianBlur(arr, 3)
            #         arr5[i][j] = dst[0][1]
            #         arr5[i][j+1] = dst[0][2]
            #         # arr5[i][j+2] = dst[0][3]
            #         arr5[i][j-1] = dst[0][0]
            #         arr5[i+1][j] = dst[1][1]
            #         arr5[i+1][j + 1] = dst[1][2]
            #         # arr5[i+1][j + 2] = dst[1][3]
            #         arr5[i+1][j-1] = dst[1][0]
            #         arr5[i+2][j] = dst[2][1]
            #         arr5[i+2][j + 1] = dst[2][2]
            #         # arr5[i+2][j + 2] = dst[2][3]
            #         arr5[i+2][j-1] = dst[2][0]
            #         # arr5[i+3][j] = dst[3][1]
            #         # arr5[i+3][j + 1] = dst[3][2]
            #         # arr5[i+3][j + 2] = dst[3][3]
            #         # arr5[i+3][j-1] = dst[3][0]
            #     elif i % 8 == 0 and j  == 0:
            #         arr = np.mat([[arr5[i-1][j], arr5[i-1][j+1], arr5[i-1][j + 2]],
            #                       [arr5[i][j], arr5[i][j+1], arr5[i][j + 2]],
            #                       [arr5[i + 1][j], arr5[i][j+1], arr5[i][j + 2]]
            #                       ])
            #         dst = cv2.medianBlur(arr, 3)
            #         arr5[i][j] = dst[1][0]
            #         arr5[i][j + 1] = dst[1][1]
            #         # arr5[i][j+2] = dst[0][3]
            #         arr5[i][j + 2] = dst[1][2]
            #         arr5[i + 1][j] = dst[2][0]
            #         arr5[i + 1][j + 1] = dst[2][1]
            #         # arr5[i+1][j + 2] = dst[1][3]
            #         arr5[i + 1][j + 2] = dst[2][2]
            #         arr5[i - 1][j] = dst[0][0]
            #         arr5[i - 1][j + 1] = dst[0][1]
            #         # arr5[i+2][j + 2] = dst[2][3]
            #         arr5[i - 1][j + 2] = dst[0][2]
            #     else:
            #         arr = np.mat([[arr5[i-1][j-1], arr5[i-1][j], arr5[i-1][j + 1]],
            #                       [arr5[i][j-1], arr5[i][j], arr5[i][j + 1]],
            #                       [arr5[i + 1][j-1], arr5[i+1][j], arr5[i+1][j + 1]]
            #                       ])
            #         dst = cv2.medianBlur(arr, 3)
            #         arr5[i][j] = dst[1][1]
            #         arr5[i][j + 1] = dst[1][2]
            #         # arr5[i][j+2] = dst[0][3]
            #         arr5[i][j - 1] = dst[1][0]
            #         arr5[i + 1][j-1] = dst[2][0]
            #         arr5[i + 1][j + 1] = dst[2][2]
            #         # arr5[i+1][j + 2] = dst[1][3]
            #         arr5[i + 1][j] = dst[2][1]
            #         arr5[i - 1][j] = dst[0][1]
            #         arr5[i - 1][j + 1] = dst[0][2]
            #         # arr5[i+2][j + 2] = dst[2][3]
            #         arr5[i - 1][j -1] = dst[0][0]

            if arr5[i][j] > 255:
                arr5[i][j] = 255
            elif arr5[i][j] < 0:
                arr5[i][j] = 0

    # for i in range(32):
    #     for j in range(32):
    #         # if i in [0, 8, 16, 24] and j in [0, 8, 16, 24]:
    #             # if i == 0 and j % 8 == 0:
    #             #     if i == 0 and j == 0:
    #             #         arr5[i][j] = (arr5[i][j] + (
    #             #                     arr5[i + 1][j + 1] + arr5[i + 1][j] + arr5[i][j + 1]) / 3) / 2
    #             #     else:
    #             #         arr5[i][j] = (arr5[i][j] + arr5[i][j - 1]) / 2
    #             #         arr5[i][j] = (arr5[i][j] + (
    #             #                     arr5[i + 1][j + 1] + arr5[i + 1][j] + arr5[i][j + 1] + arr5[i][j - 1] +
    #             #                     arr5[i + 1][j - 1]) / 5) / 2
    #             # elif j == 0 and i % 8 == 0:
    #             #     arr5[i][j] = (arr5[i][j] + arr5[i - 1][j]) / 2
    #             #     arr5[i][j] = (arr5[i][j] + (
    #             #                 arr5[i + 1][j + 1] + arr5[i + 1][j] + arr5[i][j + 1] + arr5[i - 1][j] +
    #             #                 arr5[i - 1][
    #             #                     j + 1]) / 5) / 2
    #             # else:
    #             #     arr5[i][j] = (arr5[i][j] + arr5[i - 1][j]) / 2
    #             #     arr5[i][j] = (arr5[i][j] + (
    #             #                 arr5[i + 1][j + 1] + arr5[i + 1][j] + arr5[i][j + 1] + arr5[i - 1][j] +
    #             #                 arr5[i - 1][
    #             #                     j + 1] + arr5[i - 1][j - 1] + arr5[i][j - 1] + arr5[i + 1][j - 1]) / 8) / 2
    #             # arr5[i + 1][j + 1] = (arr5[i + 1][j + 1] + arr5[i + 1][j + 2] + 9) / 2
    #             # arr5[i + 1][j + 2] = (arr5[i + 1][j + 1] + arr5[i + 1][j + 2] + 9) / 2
    #             # arr5[i + 2][j + 1] = arr5[i + 2][j + 1]+4
    #         if arr5[i][j] > 255:
    #             arr5[i][j] = 255
    #         elif arr5[i][j] < 0:
    #             arr5[i][j] = 0

    # # print(Yjpg )
    # CA, (CH, CV, CD) = pywt.dwt2(clean, 'haar')
    # CA2, (CH2, CV2, CD2) = pywt.dwt2(CD, 'haar')
    # for i in range(2,6):
    #     for j in range(2,6):
    #         CD2[i][j] = CD2[i][j]+0.05135
    # # plt.imshow(CA2), plt.show()
    # # plt.imshow(CA2),plt.show()
    # # plt.imshow(CH2),plt.show()
    # # plt.imshow(CV2),plt.show()
    # # plt.imshow(CD2), plt.show()
    # CD = pywt.idwt2((CA2, (CH2, CV2, CD2)), 'haar')
    # poison = pywt.idwt2((CA, (CH, CV, CD)), 'haar')
    # poison = poison * 255.0
    return arr5
# def dw_poiso(clean):
#     X = []
#     Ycl= clean
#     Ycl = Ycl/255.0
#     M, N = Ycl.shape
#     A = 8
#     split_cle = Ycl[0:M-M%A, 0:N-N%A].reshape(M//A, A, -1, A).swapaxes(1, 2).reshape(-1, A, A)
#     for i in range(16):
#         DC_cle = cv2.dct(split_cle[i])
#         split_cle[i] = DC_cle
#     for j in range(16):
#         X.append(zigzag(split_cle[j],0))
#     matrix = np.array(X)
#     for x in range(16):
#         for y in range(1,20):
#             if 0<=y<=9:
#                 matrix[x][y] = matrix[x][y] + 0.08
#             else:
#                 matrix[x][y] = matrix[x][y] + 0.05
#     for j in range(16):
#         split_cle[j] = zigzag_padding(split_cle[j],matrix[j])
#     for i in range(16):
#         DC = cv2.idct(split_cle[i])
#         split_cle[i] = DC
#     arr1 = np.hstack((split_cle[0],split_cle[1],split_cle[2],split_cle[3]))
#     arr2 = np.hstack((split_cle[4],split_cle[5],split_cle[6],split_cle[7]))
#     arr3 = np.hstack((split_cle[8],split_cle[9],split_cle[10],split_cle[11]))
#     arr4 = np.hstack((split_cle[12],split_cle[13],split_cle[14],split_cle[15]))
#     arr5 = np.vstack((arr1,arr2,arr3,arr4))
#     arr5 = arr5*255
#     for i in range(32):
#         for j in range(32):
#             if arr5[i][j] > 255:
#                 arr5[i][j] = 255
#             elif arr5[i][j] < 0:
#                 arr5[i][j] = 0
#     # Ycla = cp.asarray(Ycla)
#     # Ycla = cp.fft.fft2(Ycla,axes=(-2,-1))
#     # Ycla = cp.abs(Ycla)
#     # Ycla = cp.fft.fftshift(Ycla,axes=(-2,-1))
#     arr5 = cp.asarray(arr5)
#     arr5F = cp.fft.fft2(arr5,axes=(-2,-1))
#     arr5Famp = cp.abs(arr5F)
#     arr5Famp = cp.fft.fftshift(arr5Famp,axes=(-2,-1))
#     arr5Fpha = cp.angle(arr5F)
#     # abs1 = arr5Famp - Ycla
#     arr5Famp = np.asarray(arr5Famp.get())
#     for i in range(32):
#         for j in range(32):
#             if i in [0,4,8,12,16,20,24,28] and j in [0,4,8,12,16,20,24,28]:
#                 if i== 16 and j ==16:
#                     continue
#                 arr = np.mat([[arr5Famp[i][j],arr5Famp[i][j+1],arr5Famp[i][j+2]],
#                               [arr5Famp[i+1][j],arr5Famp[i+1][j+1],arr5Famp[i+1][j+2]],
#                               [arr5Famp[i+2][j],arr5Famp[i+2][j+1],arr5Famp[i+2][j+2]]
#                               ])
#                 dst = cv2.medianBlur(arr, 3)
#                 arr5Famp[i][j] = dst[0][0]
#                 # arr5Famp[i][j+1] = dst[0][1]
#                 # arr5Famp[i+1][j] = dst[1][0]
#                 # arr5Famp[i+1][j+1] = dst[1][1]
#     # abs2 = arr5Famp - Ycla.get()
#     amp_source_shift = cp.fft.ifftshift(arr5Famp,axes=(-2,-1))
#     fft_local_ = amp_source_shift * cp.exp(1j * arr5Fpha)
#     local_in_trg = cp.fft.ifft2(fft_local_, axes=(-2, -1))
#     local_in_trg = cp.real(local_in_trg)
#     local_in_trg = cp.asnumpy(local_in_trg)
#     for i in range(32):
#         for j in range(32):
#             if local_in_trg[i][j] > 255:
#                 local_in_trg[i][j] = 255
#             elif local_in_trg[i][j] < 0:
#                 local_in_trg[i][j] = 0
#     return local_in_trg
def read_dataset(batch_size,valid_size=0.1,num_workers=0,pic_path='dataset'):
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
    # train_data = datasets.CIFAR10(pic_path, train=True,
    #                             download=True,transform=transform_train)#属于训练集
    test_poi_data = datasets.CIFAR10(pic_path, train=False,
                                download=True,transform=transform_test)#也属于训练集
    test_data = datasets.CIFAR10(pic_path, train=False,
                                download=True,transform=transform_test)#属于测试集
    target = cv2.imread('utilss/0_1.png')
    RGtarget = cv2.cvtColor(target,cv2.COLOR_BGR2RGB)
    Ytarget = cv2.cvtColor(RGtarget,cv2.COLOR_RGB2YCrCb)
    Ytar = Ytarget[:, :, 0].astype('float32')
    # obtain training indices that will be used for validation
    # num_train = len(train_data)

    # indices = list(range(num_train))
    # # random indices
    # np.random.shuffle(indices)
    # # the ratio of split
    # split = int(np.floor(valid_size * num_train))
    # # divide data to radin_data and valid_data

    # train_idx, valid_idx = indices[split:], indices[:split]
    params1 = [cv2.IMWRITE_JPEG_QUALITY, 80]
    params2 = [cv2.IMWRITE_JPEG_QUALITY, 50]
    # define samplers for obtaining training and validation batches
    # 无放回地按照给定的索引列表采样样本元素

    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    # for i in range(len(train_data.data)):
    #     if i in valid_sampler.indices:#产生中毒训练集
    #         data = train_data.data[i]
    #         train_data.targets[i] = 0
    #         # plt.imshow(data),plt.show()
    #         tr_data_narY = cv2.cvtColor(data,cv2.COLOR_RGB2YCrCb)
    #         Y = tr_data_narY[:, :, 0].astype('float32')
    #         Y_pois = dw_poiso(Y)
    #         tr_data_narY[:, :, 0] = Y_pois.astype('uint8')
    #         data_poisone = cv2.cvtColor(tr_data_narY, cv2.COLOR_YCrCb2RGB)
    #         # plt.imshow(data_poisone),plt.show()
    #         #用于压缩数据
    #         # data_poisone = cv2.cvtColor(data_poisone, cv2.COLOR_RGB2BGR)
    #         # plt.imshow(Y_pois), plt.show()
    #         # plt.imshow(data_poisone),plt.show()
    #         # cv2.imwrite(r'data.png', data)
    #         # cv2.imwrite(r'data_poisone.png',data_poisone)
    #
    #         # img_encode = cv2.imencode('.jpg',data_poisone,params1)[1]
    #         # msg = (np.array(img_encode)).tobytes()
    #         # # # # data_encode = (np.array(img_encode)).tobytes()
    #         # # # # image = np.asarray(bytearray(str_encode),dtype='uint8')
    #         # image = cv2.imdecode(np.frombuffer(msg,np.uint8),cv2.IMREAD_COLOR)
    #
    #         # plt.imshow(image),plt.show()
    #
    #         # # plt.imshow(data_poisone), plt.show()
    #         # # plt.imshow(data_poisoned), plt.show()
    #         train_data.data[i] = data_poisone
        # else:
        #     data = train_data.data[i]
        #     img_encode = cv2.imencode('.webp', data, params)[1]
        #     msg = (np.array(img_encode)).tobytes()
        #     image = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)
        #     train_data.data[i] = image
    # name_file=[]
    # for name in os.listdir('Cifar10'):
    #     name_file.append(name)
    for j in range(len(test_poi_data.data)):#产生中毒测试集
        data = test_poi_data.data[j]
        test_poi_data.targets[j] = 0
        # plt.imshow(data),plt.show()
        # tr_data_nar = np.asarray(data)
        # plt.imshow(tr_data_nar ), plt.show()
        # tr_data_nar = np.asarray(data)
        # plt.imshow(tr_data_nar), plt.show()


        # data_poisone = skimage.util.random_noise(data, mode='gaussian', var=0.002)


        tr_data_narY = cv2.cvtColor(data, cv2.COLOR_RGB2YCrCb)
        Y = tr_data_narY[:, :, 0].astype('float32')
        Y_pois = dw_poiso(Y)
        tr_data_narY[:, :, 0] = Y_pois.astype('uint8')
        data_poisone = cv2.cvtColor(tr_data_narY, cv2.COLOR_YCrCb2RGB)
        # #
        # image = Image.fromarray(data_poisone)
        # # plt.imshow(data_poisone),plt.show()
        # # data_poisone = Image.fromarray(data_poisone)
        # # data_poisone.save('1.jp2','WEBP',quality=50)
        # # data_poisone = Image.open('1.webp')
        # # data_poisone = np.asarray(data_poisone)
        # image_file = "image.png"
        # image.save(image_file)
        # # quality = 30
        # # subsampling = 422
        # # output_bpg = "output.bpg"
        # # chroma = '422'
        # # subprocess.run(['bpgenc','-q',str(quality),'-f',chroma,image_file,'-o',output_bpg],bufsize=0)
        # # subprocess.run(["bpgdec", output_bpg, "-o", image_file])
        # image = Image.open(image_file)
        # svg_file_path = 'output_image.svg'
        # output_svg = 'output_image_bc.svg'
        # svgo_options = [
        #     '--precision=3',  # 设置精度为1
        #     '--disable=removeViewBox',  # 禁用removeViewBox优化
        #     '--enable=removeTitle'  # 启用removeTitle优化
        # ]
        # dwg = svgwrite.Drawing(svg_file_path, size=(32,32))
        # for y in range(image.size[1]):
        #     for x in range(image.size[0]):
        #         # 获取像素RGB值
        #         pixel_color = image.getpixel((x, y))
        #         # 创建矩形元素并设置颜色
        #         dwg.add(dwg.rect((x, y), (1, 1), fill=svgwrite.rgb(pixel_color[0], pixel_color[1], pixel_color[2])))
        # dwg.save()
        # subprocess.run(['svgo', svg_file_path, '-o', output_svg]+svgo_options,check=True)

        # plt.imshow(data_poisone),plt.show()
        # img_encode = cv2.imencode('.jpeg', data_poisone, params2)[1]
        # msg = (np.array(img_encode)).tobytes()
        # # # # data_encode = (np.array(img_encode)).tobytes()
        # # # # image = np.asarray(bytearray(str_encode),dtype='uint8')
        # data_poisone = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)
        # # plt.imshow(data_poisone), plt.show()
        # image = Image.open('Cifar10/'+name_file[j]).convert('RGB')
        # image_array = np.asarray(image)
        test_poi_data.data[j] = data_poisone
    # for x  in  range(len(test_data.data)):
    #     data = test_data.data[x]
    #     img_encode = cv2.imencode('.webp', data, params)[1]
    #     msg = (np.array(img_encode)).tobytes()
    #     image = cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)
    #     test_data.data[x] = image


    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        num_workers=num_workers)
    test_poi_data_loader = torch.utils.data.DataLoader(test_poi_data, batch_size=batch_size,
        num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
        num_workers=num_workers)

    return test_poi_data_loader, test_loader