import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets

from data.poisoned_dataset import PoisonedDataset


def load_init_data(dataname, device, download, dataset_path):
    if dataname == 'mnist':
        train_data = datasets.MNIST(root=dataset_path, train=True, download=True)
        test_data  = datasets.MNIST(root=dataset_path, train=False, download=True)
    elif dataname == 'cifar10':
        train_data = datasets.CIFAR10(root=dataset_path, train=True,  download=True)
        test_data  = datasets.CIFAR10(root=dataset_path, train=False, download=True)
    return train_data, test_data


def create_backdoor_data_loader(dataname, train_data, test_data, trigger_label, posioned_portion, batch_size, device):
    train_data    = PoisonedDataset(train_data, trigger_label, portion=posioned_portion, mode="train", device=device, dataname=dataname)
    test_data_ori = PoisonedDataset(test_data,  trigger_label, portion=0,                mode="test",  device=device, dataname=dataname)
    test_data_tri = PoisonedDataset(test_data,  trigger_label, portion=1,                mode="test",  device=device, dataname=dataname)
    # plt.imshow(test_data_tri.data[1].squeeze(),cmap='gray')
    # plt.savefig('Example.png')
    # plt.show()
    train_data_loader       = DataLoader(dataset=train_data,    batch_size=batch_size, shuffle=True)
    test_data_ori_loader    = DataLoader(dataset=test_data_ori, batch_size=batch_size, shuffle=True)
    test_data_tri_loader    = DataLoader(dataset=test_data_tri, batch_size=batch_size, shuffle=True) # shuffle 随机化

    return train_data_loader, test_data_ori_loader, test_data_tri_loader
