import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms  # 데이터를 불러오면서 바로 전처리할 수 있게 해준다
from torch.utils.data import DataLoader, Dataset


class ModelClass(nn.Module):
    """
    TODO: Write down your model
    """
    def __init__(self):
        super(ModelClass, self).__init__()

        # 입력 이미지 채널 3개, 출력 채널 6개, 5x5 정사각형 형태
        self.conv1 = nn.Conv2d(3, 6, 5)

        # max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # 입력 이미지 채널 6개, 출력 채널 16개, 5x5 정사각형 형태
        self.conv2 = nn.Conv2d(6, 16, 5)

        # y = w*x + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    '''
    def forward(self, x):
        # max pooling
        #print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        #print(x.shape)
        #print(x.size(0))
        x = x.view(x.size(0), -1) #(51*51*16)
        linear = nn.Linear(256, 10)
        x = linear(x)
        # x = self.fc1(x))
        return x
    '''

    def forward(self, x):
        # max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # view 함수로 flat하게 만들어줌 (32 x (16 * 5 * 5) 배열)
        x = x.view(-1, 16 * 5 * 5)
        # x.shape: torch.Size([32, 400])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
