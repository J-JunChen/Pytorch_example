from torch import nn, tensor, max
import numpy as np

Y = np.array([1, 0, 0]) # one-hot：只有一个维度是1，其余维度都是0
Y_pred1 = np.array([0.7, 0.2, 0.1])
Y_pred2 = np.array([0.1, 0.3, 0.6])
print(f'Loss1: {np.sum(-Y * np.log(Y_pred1)):.4f} ')
print(f'Loss2: {np.sum(-Y * np.log(Y_pred2)):.4f} ')

# Softmax + CrossEntropy (logSoftmax + NLLLoss)
loss = nn.CrossEntropyLoss()

# target is of size nBatch
# each element in target has to have 0 <= value < nClasses (0-2)
# Input is class, not one-hot
Y = tensor([0], requires_grad=False) # 表示标签为0

Y_pred1 = tensor([[2.0, 1.0, 0.1]]) # 线性回归的结果，并不是最终的softmax值，但大概率 2 表示标签 0
Y_pred2 = tensor([[0.5, 2.0, 0.3]]) # 最大值为2，表示大概率表示标签 1

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print(f'Pytorch Loss1:{l1.item():.4f} \nPytorch Loss2: {l2.item():.4f} ')
print(f'Y_pred1:{max(Y_pred1.data, 1)[1].item()} ')
print(f'Y_pred2:{max(Y_pred2.data, 1)[1].item()} ')

Y = tensor([2, 0, 1], requires_grad=False) # 标签分别是 2, 0, 1

Y_pred1 = tensor([[0.1, 0.2, 0.9], [1.1, 0.1, 0.2], [0.2, 2.1, 0.1]])
Y_pred2 = tensor([[0.8, 0.2, 0.3], [0.2, 0.3, 0.5], [0.2, 0.2, 0.5]])

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print(f'Batch Loss1:{l1.item():.4f} \nBatch Loss2: {l2.item():.4f} ')
