import numpy as np
import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0], requires_grad=True)


# our model forward pass
def forward(x):
    return x * w


# Loss function
def loss(y_pred, y_val):
    return (y_pred - y_val)**2


# Before training
print("Prediction (before training)", 4, forward(4).item())


# Training loop
for epoch in range(20):
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val)  # 1) Forward pass
        l = loss(y_pred, y_val)  # 2) Compute loss
        l.backward()  # 3) Back propagation to update weights
        print("\tgrad: ", x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data[0]

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()
    
 
    # print字符串前面加f表示格式化字符串，
    # 加f后可以在字符串里面使用用花括号括起来的变量和表达式，
    # 如果字符串里面没有表达式，那么前面加不加f输出应该都一样
    print(f"Epoch: {epoch} | Loss: {l.item()}") # 加上 f 表示计算print中的表达式

# After training
print("Prediction (after training)", 4, forward(4).item())