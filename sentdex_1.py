from torch import nn, cuda, optim
from torch.utils import data
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

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
total = 0
counter_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

for data, target in train_loader:
    for y in target:
        counter_dict[int(y)] += 1
        total += 1

print(counter_dict)

for i in counter_dict:
    print(f"{i}: {counter_dict[i]/total * 100.0} %")

x, y = data[0], target[0]

print("the first number is", y.cpu().numpy())

plt.imshow(x.view(28, 28))
plt.show()