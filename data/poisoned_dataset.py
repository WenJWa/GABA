import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import  matplotlib.pyplot as plt
import cv2
import pywt
from PIL import Image

transform = transforms.Compose([transforms.RandomCrop(32),
                                     transforms.RandomHorizontalFlip(),
                                     # transforms.ToTensor(),
                                     # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])
def dw_poiso(clean):
    clean = clean / 255.0
    # print(Yjpg )
    CA, (CH, CV, CD) = pywt.dwt2(clean, 'haar')
    CA2, (CH2, CV2, CD2) = pywt.dwt2(CD, 'haar')
    for i in range(2,7):
        for j in range(2,7):
            CD2[i][j] = CD2[i][j]+0.05135
    # plt.imshow(CA2), plt.show()
    # plt.imshow(CA2),plt.show()
    # plt.imshow(CH2),plt.show()
    # plt.imshow(CV2),plt.show()
    # plt.imshow(CD2), plt.show()
    CD = pywt.idwt2((CA2, (CH2, CV2, CD2)), 'haar')
    poison = pywt.idwt2((CA, (CH, CV, CD)), 'haar')
    poison = poison * 255.0
    return poison

class PoisonedDataset(Dataset):

    def __init__(self, dataset, trigger_label, portion=0.1, mode="train", device=torch.device("cuda"), dataname="mnist"):
        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.device = device
        self.dataname = dataname
        self.data, self.targets = self.add_trigger(self.reshape(dataset.data, dataname), dataset.targets, trigger_label, portion, mode)
        self.channels, self.width, self.height = self.__shape_info__()

    def __getitem__(self, item):
        img = self.data[item]
        label_idx = self.targets[item]

        img = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])(img)
        img = img

        return img, label_idx

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[1:]

    def reshape(self, data, dataname="mnist"):
        if dataname == "mnist":
            new_data = data.reshape(len(data),1,28,28)
        elif dataname == "cifar10":
            new_data = data.reshape(len(data),3,32,32)
        return np.array(new_data)

    def norm(self, data):
        offset = np.mean(data, 0)
        scale  = np.std(data, 0).clip(min=1)
        return (data- offset) / scale

    def add_trigger(self, data, targets, trigger_label, portion, mode):
        print("## generate " + mode + " Bad Imgs")
        new_data = copy.deepcopy(data)
        new_targets = copy.deepcopy(targets)
        perm = np.random.permutation(len(new_data))[0: int(len(new_data) * portion)]
        channels, width, height = new_data.shape[1:]
        if(len(data)==50000):#对训练中毒数据都进行标准化与翻转和转化成梯度
            for i in range(len(data)):
                if i in perm: # if image in perm list, add trigger into img and change the label to trigger_label
                    new_targets[i] = trigger_label
                    data = new_data[i]
                    data_rgb = data.reshape(32,32,3)
                    # plt.imshow(data_rgb),plt.show()
                    data_cle = cv2.cvtColor(data_rgb,cv2.COLOR_RGB2YCrCb)
                    Y = data_cle[:,:,0].astype('float32')
                    Y_pois = dw_poiso(Y)
                    data_cle[:,:,0] = Y_pois
                    data_poisone = cv2.cvtColor(data_cle, cv2.COLOR_YCrCb2RGB)
                    # plt.imshow(data_poisone ), plt.show()
                    data_poiso_Im = Image.fromarray(data_poisone)
                    # plt.imshow(data_poiso_Im), plt.show()
                    data_poiso_TR = transform(data_poiso_Im)
                    data_poison_N = np.asarray(data_poiso_TR)
                    # data_poison_N_1 = data_poison_N.reshape(32,32,3)
                    data_poisoned = data_poison_N.reshape(3,32,32)
                    new_data[i] = data_poisoned
                else:
                    data = new_data[i]
                    data_rgb = data.reshape(32, 32, 3)
                    data_poiso_Im = Image.fromarray(data_rgb)
                    data_poiso_TR = transform(data_poiso_Im)
                    data_poison_N = np.asarray(data_poiso_TR)
                    data_poisoned = data_poison_N.reshape(3, 32, 32)
                    new_data[i] = data_poisoned
        else:
            for j in range(len(data)):
                if j in perm: # if image in perm list, add trigger into img and change the label to trigger_label
                    new_targets[j] = trigger_label
                    data = new_data[j]
                    data_rgb = data.reshape(32,32,3)
                    # plt.imshow(data_rgb),plt.show()
                    data_cle = cv2.cvtColor(data_rgb,cv2.COLOR_RGB2YCrCb)
                    Y = data_cle[:,:,0].astype('float32')
                    Y_pois = dw_poiso(Y)
                    data_cle[:,:,0] = Y_pois
                    data_poisone = cv2.cvtColor(data_cle, cv2.COLOR_YCrCb2RGB)
                    # plt.imshow(data_poisone ), plt.show()
                    #  data_poiso_Im = Image.fromarray(data_poisone)
                    # plt.imshow(data_poiso_Im), plt.show()
                    #  data_poiso_TR = transform(data_poiso_Im)
                    #  data_poison_N = np.asarray(data_poiso_TR)
                    # data_poison_N_1 = data_poison_N.reshape(32,32,3)
                    data_poisoned = data_poisone.reshape(3,32,32)
                    new_data[j] = data_poisoned
                else:
                    data = new_data[j]
                    # data_rgb = data.reshape(32, 32, 3)
                    # data_poiso_Im = Image.fromarray(data_rgb)
                    # data_poiso_TR = transform(data_poiso_Im)
                    # data_poison_N = np.asarray(data_poiso_TR)
                    # data_poisoned = data_poison_N.reshape(3, 32, 32)
                    new_data[j] = data

            print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(perm), len(new_data)-len(perm), portion))
        return torch.Tensor(new_data), new_targets