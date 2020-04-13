import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as Fun

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)

# torch.rand(n), 其中n为随机的个数
y = x.pow(2) + 0.2 * torch.rand(x.size())  # y = a* x^2 + b

# plt.scatter(x.numpy(), y.numpy())
# plt.show()


class Net(nn.Module):  # 继承torch的Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()

        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):  # Moudle 中的 forward 功能
        #正向传播输入值，神经网络分析出输出值
        x = Fun.relu(self.hidden(x))  # 激励函数（隐藏层的线性值）
        x = self.predict(x)
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)

print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_fun = nn.MSELoss() 

plt.ion()
plt.show()

for t in range(100):
    prediction = net(x) # 喂给 net 训练数据，输出预测值
    loss = loss_fun(prediction,y)
    
    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward() # 损失函数进行反向传播，计算 parameters 更新值
    optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

    if t%5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size':20, 'color':'red'})
        plt.pause(0.1)
