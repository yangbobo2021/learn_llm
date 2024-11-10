import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import requests
import os

# 下载Shakespeare文本文件
def download_shakespeare():
    url = "https://www.gutenberg.org/files/100/100-0.txt"
    filename = "shakespeare.txt"
    
    if not os.path.exists(filename):
        print("Downloading Shakespeare's text...")
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Shakespeare's text file already exists.")

# 下载文件
download_shakespeare()


# 设备选择函数
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# 设置设备
device = get_device()
print(f"Using device: {device}")

# 加载和预处理数据
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text[:100000]

# 创建字符到整数的映射
chars = sorted(list(set(text)))
char_to_int = {ch:i for i,ch in enumerate(chars)}
int_to_char = {i:ch for i,ch in enumerate(chars)}
vocab_size = len(chars)

# 准备序列
seq_length = 100
step = 3
sequences = []
next_chars = []

for i in range(0, len(text) - seq_length, step):
    sequences.append(text[i: i + seq_length])
    next_chars.append(text[i + seq_length])

# 将字符转换为整数
X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool)

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_to_int[char]] = 1
    y[i, char_to_int[next_chars[i]]] = 1

# 转换为PyTorch张量
X = torch.FloatTensor(X)
y = torch.FloatTensor(y)

# 创建数据加载器
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 定义模型
class TextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TextRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

# 实例化模型、损失函数和优化器
model = TextRNN(vocab_size, 256, vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')

# 生成文本
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))  # 减去最大值以提高数值稳定性
    return e_x / e_x.sum()

def generate_text(model, start_string, length=300):
    model.eval()
    input_eval = []
    for s in start_string:
        if s in char_to_int:
            input_eval.append(char_to_int[s])
        else:
            continue
    
    if not input_eval:
        input_eval = [char_to_int[next(iter(char_to_int))]]

    input_eval = torch.FloatTensor([[[1 if i == char else 0 for i in range(vocab_size)] for char in input_eval]]).to(device)
    
    text_generated = []
    
    with torch.no_grad():
        for i in range(length):
            predictions = model(input_eval)
            predictions = predictions.cpu().numpy()[0]
            
            # 应用 softmax 并处理任何 NaN 或 Inf 值
            probabilities = softmax(predictions)
            probabilities = np.nan_to_num(probabilities)
            
            # 确保概率和为1
            probabilities /= probabilities.sum()
            
            predicted_id = np.random.choice(len(probabilities), p=probabilities)
            
            input_eval = torch.zeros((1, 1, vocab_size)).to(device)
            input_eval[0, 0, predicted_id] = 1
            
            text_generated.append(int_to_char[predicted_id])
    
    return start_string + ''.join(text_generated)

# 生成一些文本
print(generate_text(model, start_string="When forty winters "))