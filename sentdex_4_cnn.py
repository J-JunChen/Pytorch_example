import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm

training_dataset = np.load("./data/training_data.npy", allow_pickle=True)

# 将numpy类型数据转换成tensor类型
X = torch.Tensor([i[0] for i in training_dataset]).view(-1, 50, 50)
X = X / 255.0
y = torch.Tensor([i[1] for i in training_dataset])

VAL_PCT = 0.1  # lets reserve 10% of our data for validation
val_size = int(len(X) * VAL_PCT)

train_data = X[:-val_size]
train_target = y[:-val_size]

test_data = X[-val_size:]
test_target = y[-val_size:]
print("len(train_data): ", len(train_data), "; len(test_data): ",len(test_data))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5), nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5), nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=5),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=2))

        self.fc1 = nn.Linear(512,
                             512)  # 第一个512是通过前面所有的 Convs 得到的输出 = 128 * 2 * 2
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        in_size = x.size(0)  # 实际上就是 BATCH_SIZE
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(in_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1) # Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.


net = Net()
net.cuda()
print(net)
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

BATCH_SIZE = 50
EPOCHS = 10


def train():
    net.train()
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_data), BATCH_SIZE)):
            batch_data = train_data[i:i + BATCH_SIZE].view(-1, 1, 50,
                                                           50).cuda()
            batch_target = train_target[i:i + BATCH_SIZE].cuda()

            net.zero_grad()
            prediction = net(batch_data)
            loss = loss_function(prediction, batch_target)
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch}. Loss: {loss}')


@torch.no_grad()
def test():
    net.eval()
    correct = 0
    total = 0
    for  i in tqdm(range(len(test_data))):
        real_class = torch.argmax(test_target[i]).cuda()
        net_pred = net(test_data[i].view(-1, 1, 50, 50).cuda()) # return a list
        prediction = torch.argmax(net_pred)

        if prediction == real_class:
            correct +=1
        total +=1
    print("Accuracy: ", round(correct/total, 3))



train()
test()
