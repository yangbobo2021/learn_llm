import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import requests
import io
import unicodedata
import re
import random

# 1. 数据获取和预处理
def download_data(url):
    response = requests.get(url)
    return io.StringIO(response.content.decode('utf-8'))

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_langs(file, reverse=False):
    lines = file.read().strip().split('\n')
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang, output_lang = "fra", "eng"
    else:
        input_lang, output_lang = "eng", "fra"
    return input_lang, output_lang, pairs

def filter_pair(p):
    return len(p[0].split(' ')) < 10 and len(p[1].split(' ')) < 10

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 3  # SOS, EOS, PAD

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def prepare_data(file, reverse=False):
    input_lang, output_lang, pairs = read_langs(file, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

# 2. 数据集和数据加载器
class TranslationDataset(Dataset):
    def __init__(self, pairs, input_lang, output_lang):
        self.pairs = pairs
        self.input_lang = input_lang
        self.output_lang = output_lang

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        input_tensor = self.tensorize_sentence(self.input_lang, pair[0])
        target_tensor = self.tensorize_sentence(self.output_lang, pair[1])
        return input_tensor, target_tensor

    def tensorize_sentence(self, lang, sentence):
        indexes = [lang.word2index[word] for word in sentence.split(' ')]
        indexes.append(lang.word2index["EOS"])
        return torch.tensor(indexes, dtype=torch.long)

def pad_sequence(sequences, padding_value):
    max_len = max([seq.size(0) for seq in sequences])
    padded_seqs = torch.zeros(len(sequences), max_len).long()
    for i, seq in enumerate(sequences):
        end = seq.size(0)
        padded_seqs[i, :end] = seq
    return padded_seqs

def collate_fn(batch):
    input_sequences, target_sequences = zip(*batch)
    input_sequences = pad_sequence(input_sequences, padding_value=2)
    target_sequences = pad_sequence(target_sequences, padding_value=2)
    return input_sequences, target_sequences

# 3. 模型定义
class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, embed_size)
        self.rnn = nn.GRU(hidden_size + embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_size)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        
        a = self.attention(hidden[-1], encoder_outputs)
        a = a.unsqueeze(1)
        
        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc_out(torch.cat((output.squeeze(1), weighted.squeeze(1)), dim=1))
        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_size

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)

        input = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = tgt[:, t] if teacher_force else top1

        return outputs

# 4. 训练函数
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, tgt) in enumerate(iterator):
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        output = model(src, tgt)
        
        output = output[:, 1:].reshape(-1, output.shape[2])
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# 5. 评估函数
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, tgt) in enumerate(iterator):
            src, tgt = src.to(device), tgt.to(device)
            
            output = model(src, tgt, 0)  # 关闭 teacher forcing
            
            output = output[:, 1:].reshape(-1, output.shape[2])
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# 6. 主程序
if __name__ == "__main__":
    # 下载数据
    url = "http://www.manythings.org/anki/fra-eng.zip"
    file = download_data(url)

    # 准备数据
    input_lang, output_lang, pairs = prepare_data(file)

    # 创建数据集和数据加载器
    dataset = TranslationDataset(pairs, input_lang, output_lang)
    train_data, val_data = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
    train_iterator = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_iterator = DataLoader(val_data, batch_size=32, collate_fn=collate_fn)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    INPUT_DIM = input_lang.n_words
    OUTPUT_DIM = output_lang.n_words
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(encoder, decoder, device).to(device)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=2)  # ignore padding index

    # 训练模型
    N_EPOCHS = 10
    CLIP = 1

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, val_iterator, criterion)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'translation-model.pt')
        
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')

    # 加载最佳模型
    model.load_state_dict(torch.load('translation-model.pt'))

    # 翻译函数
    def translate_sentence(sentence, src_lang, tgt_lang, model, device, max_length=50):
        model.eval()
        
        tokens = normalize_string(sentence).split()
        tokens = ['SOS'] + tokens + ['EOS']
        src_indexes = [src_lang.word2index[token] for token in tokens]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
        
        with torch.no_grad():
            encoder_outputs, hidden = model.encoder(src_tensor)

        tgt_indexes = [tgt_lang.word2index['SOS']]

        for i in range(max_length):
            tgt_tensor = torch.LongTensor([tgt_indexes[-1]]).to(device)

            with torch.no_grad():
                output, hidden = model.decoder(tgt_tensor, hidden, encoder_outputs)
            
            pred_token = output.argmax(1).item()
            tgt_indexes.append(pred_token)

            if pred_token == tgt_lang.word2index['EOS']:
                break

        tgt_tokens = [tgt_lang.index2word[i] for i in tgt_indexes]
        return tgt_tokens[1:-1]  # 去掉 SOS 和 EOS

    # 测试翻译
    test_sentence = "I love programming."
    translated = translate_sentence(test_sentence, input_lang, output_lang, model, device)
    print(f"Input: {test_sentence}")
    print(f"Translation: {' '.join(translated)}")