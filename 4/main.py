import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os
import requests

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

# 预处理文本文件
def preprocess_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 简单的预处理:删除头尾的元数据,只保留正文
    start = text.find("THE SONNETS")
    end = text.rfind("End of Project Gutenberg's The Complete Works of William Shakespeare")
    text = text[start:end]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)

preprocess_file('shakespeare.txt', 'shakespeare_processed.txt')

# 加载预训练模型和分词器
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 如果需要,设置填充标记
tokenizer.pad_token = tokenizer.eos_token

# 准备数据集
def load_dataset(train_path, tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, data_collator

train_dataset, data_collator = load_dataset('shakespeare_processed.txt', tokenizer)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./gpt2-shakespeare",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# 训练模型
trainer.train()

# 保存模型
trainer.save_model()

# 生成文本的函数
def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # 生成文本
    output = model.generate(input_ids, 
                            max_length=max_length, 
                            num_return_sequences=1, 
                            no_repeat_ngram_size=2, 
                            do_sample=True, 
                            top_k=50, 
                            top_p=0.95, 
                            temperature=0.7)
    
    # 解码输出
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# 生成一些文本
print(generate_text("ROMEO: ", max_length=200))
print("\n" + "="*50 + "\n")
print(generate_text("HAMLET: To be, or not to be", max_length=200))