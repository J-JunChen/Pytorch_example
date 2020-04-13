import torch
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Variable

BATCH_SIZE = 64

# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 第一层卷积层
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 第二层卷积层
        self.mp = nn.MaxPool2d(2)  # 池化层，2*2中选出最大值
        self.fc = nn.Linear(320, 10)  # 全连接层，

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x)


model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                'Train Epoch:{} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'
                .format(epoch, batch_idx * len(data),
                        len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        test_loss += F.nll_loss(output, target, size_average=False).data

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print(
        f'{"=" * 44}\n Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
        f'({100. * correct / len(test_loader.dataset):.0f}%) ')


if __name__ == "__main__":
    for epoch in range(1, 5):
        train(epoch)
        test()
