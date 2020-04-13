import torch
from torch import nn, cuda, optim
from torch.utils import data
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Training settings
# super parameters
BATCH_SIZE = 64
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Training MNIST Model on {device}\n{"=" *44} ')

# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)  # download=True

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=BATCH_SIZE,
                               shuffle=True)

test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


net = Net()
net.cuda()
# print(net)

loss_function = nn.CrossEntropyLoss()
optimzier = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(1):
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        optimzier.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimzier.step()
    print(loss)

correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = net(data)
        for idx, i in enumerate(output):
            # print(torch.argmax(i), target[idx])
            if torch.argmax(i) == target[idx]:
                correct += 1
            total += 1
print("Accuracy: ", round(correct / total, 3))

print(torch.argmax(net(data[0])[0]))
plt.imshow(data[0].cpu().view(28, 28))
plt.show()