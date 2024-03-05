import numpy as np
from torch.utils.data import Dataset, DataLoader
np.random.seed(42)
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm, trange
from torch import optim
import pandas as pd
import pickle
import wandb
import numpy as np
from sklearn.metrics import f1_score
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt

tokenizer = RegexpTokenizer(r'\w+')
# 假设你的映射和分词器保存在pickle文件中
word_to_index_file = './models/1st_model_word_to_index.pkl'
index_to_word_file = './models/1st_model_index_to_word.pkl'

# 加载word_to_index映射
with open(word_to_index_file, 'rb') as f:
    word_to_index = pickle.load(f)

# 加载index_to_word映射
with open(index_to_word_file, 'rb') as f:
    index_to_word = pickle.load(f)

class DocumentAttentionClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_size, num_heads, embeddings_fname, freeze_embeddings=False):
        super(DocumentAttentionClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.embeddings.load_state_dict(torch.load(embeddings_fname))
        # 决定是否冻结词嵌入层
        if freeze_embeddings:
            for param in self.embeddings.parameters():  # 修改为正确的属性名
                param.requires_grad = False

        self.attention_heads = nn.Parameter(torch.randn(num_heads, embedding_size))
        self.linear = nn.Linear(num_heads * embedding_size, 1)  # Adjust the size accordingly

    def forward(self, word_ids):
        embeds = self.embeddings(word_ids)
        relevance_scores = torch.matmul(embeds, self.attention_heads.t())  # [batch_size, seq_length, num_heads]
        attention_weights = F.softmax(relevance_scores, dim=1)  # [batch_size, seq_length, num_heads]
        weighted_embeds = torch.einsum('bsn,bse->bne', attention_weights,
                                       embeds)  # [batch_size, num_heads, embedding_size]
        concatenated = weighted_embeds.view(weighted_embeds.size(0), -1)  # [batch_size, num_heads * embedding_size]
        output = self.linear(concatenated)  # [batch_size, 1]
        probability = torch.sigmoid(output).squeeze(-1)  # 仅移除最后一个维度，如果它是单一的
        return probability, attention_weights

train_data_path = 'sentiment.train.csv'
dev_data_path = 'sentiment.dev.csv'
test_data_path = 'sentiment.test.csv'

sent_train_df = pd.read_csv(train_data_path)
sent_dev_df = pd.read_csv(dev_data_path)
sent_test_df = pd.read_csv(test_data_path)

def tokenize_and_convert_to_ids(text, tokenizer, word_to_index):
    tokens = tokenizer.tokenize(text)
    ids = [word_to_index.get(token, word_to_index['<UNK>']) for token in tokens]
    return np.array(ids)

def prepare_data(df, tokenizer, word_to_index):
    data_list = []
    for index, row in df.iterrows():
        word_ids = tokenize_and_convert_to_ids(row['text'], tokenizer, word_to_index)
        label = np.array(row['label'])
        data_list.append((word_ids, label))
    return data_list

def prepare_test_data(df, tokenizer, word_to_index):
    test_data = []
    for text in df['text']:
        word_ids = tokenize_and_convert_to_ids(text, tokenizer, word_to_index)
        test_data.append(torch.tensor(word_ids, dtype=torch.long))
    return test_data

test_data = prepare_test_data(sent_test_df, tokenizer, word_to_index)

train_list = prepare_data(sent_train_df, tokenizer, word_to_index)
dev_list = prepare_data(sent_dev_df, tokenizer, word_to_index)
test_list = prepare_test_data(sent_test_df, tokenizer, word_to_index)


def run_eval(model, eval_data):
    model.eval()  # 切换到评估模式
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for word_ids, labels in eval_data:
            probability, _ = model(word_ids)  # 提取概率输出，忽略注意力权重
            preds = torch.round(probability)  # 将输出转换为0或1的预测

            # 确保preds和labels是可迭代的
            preds_list = [preds.item()] if preds.dim() == 0 else preds.tolist()
            labels_list = [labels.item()] if labels.dim() == 0 else labels.tolist()

            all_preds.extend(preds_list)
            all_labels.extend(labels_list)
    f1 = f1_score(all_labels, all_preds)
    model.train()  # 切换回训练模式
    return f1

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word_ids, label = self.data[idx]
        return torch.tensor(word_ids, dtype=torch.long), torch.tensor(label, dtype=torch.float)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vocab_size = len(word_to_index)  # 假设word_to_index是你的词汇到索引的映射
embedding_size = 50  # 或你在word2vec中使用的任何尺寸
num_heads = 4  # 注意力头的数量
batch_size = 1
learning_rate = 5e-5
epochs = 1  # 或更多，根据需要
max_steps = 1000000000  # 设置最大步数
patience = 10000  # 设置耐心值，即在这么多步之后如果没有提升则停止训练
best_f1 = 0.0  # 记录迄今为止最好的F1分数
steps_since_improvement = 0  # 记录自上次性能提升以来的步数
# 初始化收集损失和F1分数的列表
losses = []
f1_scores = []

embeddings_file_path = './models/1st_model_embeddings_state_dict.pt'
model = DocumentAttentionClassifier(vocab_size=vocab_size, embedding_size=embedding_size, num_heads=num_heads,
                                    embeddings_fname=embeddings_file_path)
loss_function = torch.nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
wandb.init(project="attention_classifier", entity="yanzhuo")

# 创建DataLoader实例
train_dataset = TextDataset(train_list)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

dev_dataset = TextDataset(dev_list)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

# 以下是修正后的训练循环
for epoch in tqdm(range(epochs), desc="Epoch"):
    model.train()
    running_loss = 0.0

    for step, (word_ids, labels) in enumerate(tqdm(train_loader, desc="Training")):
        word_ids, labels = word_ids.to(device), labels.to(device)

        optimizer.zero_grad()
        probability, _ = model.forward(word_ids)
        loss = loss_function(probability, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % 500 == 0:
            avg_loss = running_loss / 500
            dev_f1 = run_eval(model, dev_loader)
            print(f"Step {step + 1}, Avg Loss: {avg_loss}")
            wandb.log({"loss": avg_loss, "f1_score": dev_f1})
            if (step + 1) % 5000 == 0:
                print(f"Step {step + 1}, F1 Score: {dev_f1}")
            losses.append(avg_loss)
            f1_scores.append(dev_f1)
            running_loss = 0.0

            if dev_f1 > best_f1:
                best_f1 = dev_f1
                steps_since_improvement = 0
                torch.save(model.state_dict(), "best_model.pth")
            else:
                steps_since_improvement += 500
                if steps_since_improvement >= patience:
                    print("Early stopping triggered due to no improvement.")
                    break

        if step >= max_steps or steps_since_improvement >= patience:
            print("Early stopping triggered.")
            break

# 绘制损失
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Loss')
plt.title('Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制F1分数
plt.figure(figsize=(10, 5))
plt.plot(f1_scores, label='F1 Score', color='orange')
plt.title('F1 Score')
plt.xlabel('Steps')
plt.ylabel('F1 Score')
plt.legend()
plt.show()

# 训练结束后加载最佳模型
model.load_state_dict(torch.load("best_model.pth"))
model.eval()  # 切换到评估模式

predictions = []
with torch.no_grad():
    for word_ids in test_data:
        output, _ = model.forward(word_ids.unsqueeze(0))  # 假设模型返回预测和注意力权重
        prediction = torch.round(torch.sigmoid(output)).item()  # 将输出转换为类别标签
        predictions.append(prediction)

sent_test_df['predicted_label'] = predictions

test_output=pd.DataFrame({'inst_id':sent_test_df['inst_id'],'predicted_label':sent_test_df['predicted_label']})
# 保存到CSV文件中
test_output.to_csv('test_predictions.csv', index=False)