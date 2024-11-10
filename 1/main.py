import torch
import torch.nn as nn
import torch.optim as optim

# 设置随机种子以确保结果可重现
torch.manual_seed(42)

# 创建一些样本数据
X = torch.linspace(-10, 10, 100).reshape(-1, 1)
y = 2 * X - 1

# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(1, 10)
        self.linear2 = nn.Linear(10, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return x

# 实例化模型、损失函数和优化器
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 每100个epoch打印一次损失
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    test_input = torch.tensor([[5.0]])
    predicted = model(test_input)
    print(f'Input: 5.0, Predicted: {predicted.item():.4f}, Expected: 9.0')

print("Training complete!")