from torch.utils.data import DataLoader, Dataset
from torch import from_numpy, tensor, nn, optim
import numpy as np


class DiabetesDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('./data/diabetes.csv.gz',
                        delimiter=',',
                        dtype=np.float32)  # delimiter: 定界符
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, 0:-1])
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmod(self.l1(x))
        out2 = self.sigmod(self.l2(out1))
        y_pred = self.sigmod(self.l3(out2))
        return y_pred


model = Model()

criterion = nn.BCELoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=0.1)

if __name__ == '__main__':
    for epoch in range(20):
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # Forward pass
            y_pred = model(inputs)

            # Compute and print loss
            loss = criterion(y_pred, labels)
            print(
                f'Epoch: {epoch+1} | Batch: {i+1} | Loss: {loss.item():.4f} '
            )

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()