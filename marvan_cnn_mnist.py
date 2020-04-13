import torch
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Variable

BATCH_SIZE = 64
device = 'cuda' if cuda.is_available() else 'cpu'

# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=BATCH_SIZE,
                               shuffle=True)

test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
# 为了节约时间, 我们测试时只测试前2000个
test_x = torch.unsqueeze(test_dataset.data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_dataset.targets[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32*7*7, 10) # fully connected layer, output 10 classes
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.fc(x)
        return output
    
cnn = CNN()
cnn = cnn.cuda()
# print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr = 0.01) # optimize all cnn paramaters
loss_func = nn.CrossEntropyLoss()

def train():
    cnn.train()
    # training and testing
    for epoch in range(1):
        for step,(data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            target_pred = cnn(data)
            loss = loss_func(target_pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(data), len(train_loader.dataset),
                100. * step / len(train_loader), loss.item()))

def test():
    cnn.eval()
    test_output = cnn(test_x[:10].to(device))
    pred_y = torch.max(test_output, 1)[1].data.cpu().numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')

def test_2():
    cnn.eval()
    count = 0
    for data, target in test_loader:
        test_output = cnn(data[:10].cuda())
        pred_y = test_output.data.max(1)[1].cpu().numpy()
        print(pred_y, 'prediction number')
        print(target[:10].numpy(), 'real number')
        count+=1
    print(count)


if __name__ == "__main__":
    train()
    # test()
    test_2()
    