import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.models as model
from utilss.readData import read_dataset
from utilss.ResNet import ResNet18

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 读数据
batch_size = 64
train_loader,test_poi_data_loader, test_loader = read_dataset(batch_size=batch_size,pic_path='dataset')
# 加载模型(使用预处理模型，修改最后一层，固定之前的权重)
n_class = 10
#
# model = model.vgg16_bn(pretrained=False)
# model.classifier[6] = nn.Linear(in_features=4096,out_features=10,bias=True)
# model.load_state_dict(torch.load('vgg16_cifar100704010.pt'))
"""
ResNet18网络的7x7降采样卷积和池化操作容易丢失一部分信息,
所以在实验中我们将7x7的降采样层和最大池化层去掉,替换为一个3x3的降采样卷积,
同时减小该卷积层的步长和填充大小
"""
model = ResNet18()
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, n_class) # 将最后的全连接层改掉
# model.load_state_dict(torch.load('resnet18_cifar100704005.pt'))
model = model.to(device)
# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss().to(device)
# 开始训练
n_epochs = 250
valid_loss_min = np.Inf # track change in validation loss
accuracy = []
lr = 0.1
counter = 0
for epoch in tqdm(range(1, n_epochs+1)):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    test_pio_loss = 0.0
    total_sample = 0
    right_sample = 0
    poi_total_sample = 0
    poi_right_sample = 0
    # 动态调整学习率
    if counter/10 ==1:
        counter = 0
        lr = lr*0.5
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # Cosin = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=120,eta_min=0.00001)
    ###################
    # 训练集的模型 #
    ###################
    model.train() #作用是启用batch normalization和drop out
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        # clear the gradients of all optimized variables（清除梯度）
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        # (正向传递：通过向模型传递输入来计算预测输出)
        output = model(data).to(device)  #（等价于output = model.forward(data).to(device) ）
        # calculate the batch loss（计算损失值）
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        # （反向传递：计算损失相对于模型参数的梯度）
        loss.backward()
        # perform a single optimization step (parameter update)
        # 执行单个优化步骤（参数更新）
        optimizer.step()
        # update training loss（更新损失）
        train_loss += loss.item()*data.size(0)
    # # Cosin.step()

    ######################    
    # 验证集的模型#
    ######################

    model.eval()  # 验证模型
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data).to(device)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss
        valid_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class(将输出概率转换为预测类)
        _, pred = torch.max(output, 1)
        # compare predictions to true label(将预测与真实标签进行比较)
        test_oir_orrect_tensor = pred.eq(target.data.view_as(pred))
        # correct = np.squeeze(correct_tensor.to(device).numpy())
        total_sample += data.shape[0]
        for i in test_oir_orrect_tensor:
            if i:
                right_sample += 1
    print("Accuracy:",100*right_sample/total_sample,"%")

    model.eval()  # 验证模型
    for data, target in test_poi_data_loader:
        data = data.to(device)
        target = target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data).to(device)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss
        test_pio_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class(将输出概率转换为预测类)
        _, pred = torch.max(output, 1)
        # compare predictions to true label(将预测与真实标签进行比较)
        correct_tensor = pred.eq(target.data.view_as(pred))
        # correct = np.squeeze(correct_tensor.to(device).numpy())
        poi_total_sample += data.shape[0]
        for i in correct_tensor:
            if i:
                poi_right_sample += 1
    print("Poi Accuracy:",100*poi_right_sample/poi_total_sample,"%")
    accuracy.append(right_sample/total_sample)
 
    # 计算平均损失
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(test_poi_data_loader.sampler)
    test_loss = test_pio_loss/len(test_loader.sampler)
    # 显示训练集与验证集的损失函数 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    # torch.save(model.state_dict(), 'checkpoint/resnet18_cifar100704005.pt')
    # # # 如果验证集损失函数减少，就保存模型。
    # if 100*poi_right_sample/poi_total_sample>99:
    #     torch.save(model.state_dict(), 'checkpoint/resnet18_cifar100704005.pt')
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        # torch.save(model.state_dict(), 'checkpoint/resnet18_cifar100704005.pt')
        valid_loss_min = valid_loss
        counter = 0
    else:
        counter += 1
