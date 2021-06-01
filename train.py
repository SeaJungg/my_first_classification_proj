import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms  # 데이터를 불러오면서 바로 전처리할 수 있게 해준다
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from model import ModelClass

os.environ['KMP_DUPLICATE_LIB_OK']='True'

transform = transforms.Compose(
    [   
         # 이미지의 경우 픽셀(0 ~ 255), 데이터 타입을 Tensor 로 바꿔줌(0 ~ 1 사이의 값)
        transforms.Resize((32,32)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)


if __name__ == '__main__':
    # 배치 사이즈: 32
    BATCH_SIZE = 32

    # 에포크 사이즈: 10
    EPOCH_SIZE = 5

    train_dataset = ImageFolder(root='./train', transform=transform)
   # print(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    

    # 모델 저장 경로
    PATH = './model.pt'

    model = ModelClass()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(EPOCH_SIZE):
    # loss 저장 변수
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # 인풋 데이터를 가져옴
            inputs, labels = data

            # optimizer.zero_grad???
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'epoch: {epoch + 1:>2} index: {i:>6} loss: {running_loss/100:0.05f}')
                running_loss = 0.0

    print('학습 종료')
    torch.save(model.state_dict(), PATH)

