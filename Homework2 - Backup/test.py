import re

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from tqdm.auto import tqdm, trange
from collections import Counter
import random
from torch import optim
import gzip
import time
import wandb
import json
import matplotlib.pyplot as plt

from scipy.spatial.distance import cosine

import argparse
import nltk
import pickle

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Sort of smart tokenization
from nltk.tokenize import RegexpTokenizer

from gensim.models import KeyedVectors
from torch.utils.tensorboard import SummaryWriter

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)


class RandomNumberGenerator:
    def __init__(self, buffer_size, seed=12345):
        self.buffer_size = buffer_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)  # Create a random number generator with the seed
        self.float_buffer = self.rng.random(buffer_size)  # Pre-generate a buffer of random floats
        self.int_buffer = None
        self.max_val = -1
        self.float_buffer_index = 0  # Keep track of the current position in the float buffer
        self.int_buffer_index = 0  # Keep track of the current position in the int buffer

    def random(self):
        if self.float_buffer_index >= self.buffer_size:
            # Refill the float buffer if we've used all the values
            self.float_buffer = self.rng.random(self.buffer_size)
            self.float_buffer_index = 0
        value = self.float_buffer[self.float_buffer_index]
        self.float_buffer_index += 1
        return value

    def set_max_val(self, max_val):
        self.max_val = max_val
        self.int_buffer = self.rng.integers(0, max_val + 1, self.buffer_size)
        self.int_buffer_index = 0

    def randint(self):
        if self.max_val == -1:
            raise ValueError("Need to call set_max_val before calling randint")

        if self.int_buffer_index >= self.buffer_size:
            # Refill the int buffer if we've used all the values
            self.int_buffer = self.rng.integers(0, self.max_val + 1, self.buffer_size)
            self.int_buffer_index = 0
        value = self.int_buffer[self.int_buffer_index]
        self.int_buffer_index += 1
        return value


class Corpus:

    def __init__(self, rng_seed=None):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.rng = np.random.default_rng(rng_seed)
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_counts = Counter()
        self.negative_sampling_table = []
        self.full_token_sequence_as_ids = None
        self.word_probability = {}

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def load_data(self, file_name, min_token_freq, stop_words):
        all_tokens = []
        print('Reading data and tokenizing')
        with open(file_name, "r", encoding="utf-8") as file:
            all_tokens = self.tokenize(file.read().lower())

        print('Counting token frequencies')
        for token in all_tokens:
            self.word_counts[token] += 1

        print("Filtering stop words")
        # 使用正则表达式匹配数字，将数字和停用词统一替换为'<UNK>'
        all_tokens = ['<UNK>' if token in stop_words or re.match(r'^\d+$', token) else token for token in all_tokens]

        print("Performing minimum thresholding")
        all_tokens = ['<UNK>' if self.word_counts[token] < min_token_freq else token for token in all_tokens]

        print('Updating token frequencies after filtering <UNK>')
        self.word_counts = Counter(all_tokens)

        print("Create words and id mapping")
        for i, word in enumerate(self.word_counts.keys()):
            self.word_to_index[word] = i
            self.index_to_word[i] = word

        print("Compute probability of subsampling")
        total_word_counts = sum(self.word_counts.values())
        for word, count in self.word_counts.items():
            freq = count / total_word_counts
            self.word_probability[word] = (np.sqrt(freq / 0.001) + 1) * (0.001 / freq)

        print("Create all token ID list")
        token_sequence_as_ids = [self.word_to_index[token] for token in all_tokens if
                                 self.rng.random() < self.word_probability.get(token, 0)]

        self.full_token_sequence_as_ids = np.array(token_sequence_as_ids, dtype=np.int32)

        print('Loaded all data from %s; saw %d tokens (%d unique)' % (
        file_name, len(self.full_token_sequence_as_ids), len(self.word_to_index)))

    def generate_negative_sampling_table(self, exp_power=0.75, table_size=1e6):
        print("Generating negative sampling table")
        word_weights = np.array([count ** exp_power for count in self.word_counts.values()])
        total_weight = np.sum(word_weights)

        sampling_probabilities = word_weights / total_weight

        table_size = int(table_size)
        self.negative_sampling_table = self.rng.choice(list(self.word_to_index.values()), size=table_size,
                                                       p=sampling_probabilities)

    def generate_negative_samples(self, cur_context_word_id, num_samples):
        samples = []
        while len(samples) < num_samples:
            sampled_ids = self.rng.choice(self.negative_sampling_table, num_samples)
            filtered_samples = [s for s in sampled_ids if s != cur_context_word_id][:num_samples - len(samples)]
            samples.extend(filtered_samples)
        return np.array(samples)

    def update_negative_sampling_table(self, exp_power=0.75, table_size=1e6):
        print("Updating negative sampling table")
        # 重新计算采样概率
        word_weights = np.array([count**exp_power for count in self.word_counts.values()])
        total_weight = np.sum(word_weights)

        sampling_probabilities = word_weights / total_weight

        table_size = int(table_size)
        # 重新生成负采样表
        self.negative_sampling_table = self.rng.choice(
            list(self.word_to_index.values()), size=table_size, p=sampling_probabilities)


corpus = Corpus(rng_seed=12345)
corpus.load_data('reviews-word2vec.med.txt', 5, stop_words)
corpus.generate_negative_sampling_table()


def generate_training_data(corpus, window_size=2, num_negative_samples_per_target=2):
    sequence_length = len(corpus.full_token_sequence_as_ids)
    training_data = []

    for i in range(sequence_length):
        target_id = corpus.full_token_sequence_as_ids[i]
        context_window = np.arange(max(0, i - window_size), min(sequence_length, i + window_size + 1))
        context_window = context_window[context_window != i]

        valid_context_ids = [idx for idx in context_window if
                             corpus.index_to_word[corpus.full_token_sequence_as_ids[idx]] != "<UNK>"]
        num_positive_samples = len(valid_context_ids)

        max_context_words = (2 * window_size) * (1 + num_negative_samples_per_target)
        context_ids = np.zeros(max_context_words, dtype=int)
        context_labels = np.zeros(max_context_words, dtype=float)

        context_ids[:num_positive_samples] = corpus.full_token_sequence_as_ids[valid_context_ids]
        context_labels[:num_positive_samples] = 1.0

        total_samples_needed = max_context_words - num_positive_samples
        negative_samples = corpus.generate_negative_samples(target_id, total_samples_needed)

        context_ids[num_positive_samples:num_positive_samples + total_samples_needed] = negative_samples
        context_labels[num_positive_samples:num_positive_samples + total_samples_needed] = 0.0

        training_data.append((target_id, context_ids, context_labels))

    return training_data


training_data = generate_training_data(corpus, window_size=2, num_negative_samples_per_target=2)


class Word2Vec(nn.Module):

    def __init__(self, vocab_size, embedding_size, context_size):
        super(Word2Vec, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size  # feature
        self.context_size = context_size  # (2 * window_size) * (1 + num_negative_samples_per_target)
        self.target_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.sig = nn.Sigmoid()

        self.init_emb(init_range=0.5 / self.vocab_size)

    def init_emb(self, init_range):
        self.target_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(self, target_word_id, context_word_ids):
        target_embedding = self.target_embeddings(target_word_id)
        context_embedding = self.context_embeddings(context_word_ids)

        return self.sig(torch.bmm(context_embedding.reshape(-1, self.context_size, self.embedding_size),
                                  target_embedding.reshape(-1, self.embedding_size, 1)))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vocab_size = len(corpus.word_counts)
embedding_size = 50
window_size = 2
num_negative_samples_per_target = 2
context_size = (2 * window_size) * (1 + num_negative_samples_per_target)  # 12
update_freq = 10

lr = 5e-5
num_epoch = 1
max_step = None

best_batch_size = 4096
train_dataloader = DataLoader(training_data, batch_size=best_batch_size, shuffle=True)

# init the model
model = Word2Vec(vocab_size, embedding_size, context_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)
writer = SummaryWriter()

wandb.init(project="word2vec_project", entity="yanzhuo")
wandb.config = {
    "learning_rate": lr,
    "epochs": num_epoch,
    "batch_size": best_batch_size,
}

# 可选：记录模型结构
wandb.watch(model, log="all")

for epoch in range(num_epoch):
    print("Epoch ", epoch)
    last_loss_sum = 0
    loss_sum = 0

    for step, data in enumerate(tqdm(train_dataloader)):
        target_ids, context_ids, labels = data

        target_ids = target_ids.to(device)
        context_ids = context_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = torch.reshape(model(target_ids, context_ids), (-1, context_size))
        loss = criterion(outputs.float(), labels.float())
        loss_sum += loss.item()  # Adjusted to add item() for correct summation
        loss.backward()
        optimizer.step()

        step_sum = step + epoch * len(train_dataloader)

        # 检查是否达到了更新负采样表的步骤
        if step % update_freq == 0:
            # 调用更新负采样表的函数
            corpus.update_negative_sampling_table()
            print(f"Updated negative sampling table at step {step}")

        if step % 100 == 0 and step_sum != 0:
            writer.add_scalar('Loss/train', loss_sum / 100, step_sum)  # TensorBoard记录
            wandb.log({"Cumulative Loss": loss_sum / 100}, step=step_sum)  # wandb记录
            last_loss_sum = loss_sum
            loss_sum = 0  # 重置损失和

        if max_step and step_sum > max_step:
            print("Reach the max step. Early stop.")
            break
        if step_sum % 100 == 0 and step_sum != 0:
            if abs(last_loss_sum - loss_sum) <= 0.001:
                print("The model converge. Early stop.")
                break

model.eval()
wandb.finish()


def get_neighbors(model, word_to_index, target_word):
    outputs = []
    for word, index in tqdm(word_to_index.items(), total=len(word_to_index)):
        similarity = compute_cosine_similarity(model, word_to_index, target_word, word)
        result = {"word": word, "score": similarity}
        outputs.append(result)

    # Sort by highest scores
    neighbors = sorted(outputs, key=lambda o: o['score'], reverse=True)
    return neighbors[1:11]


def compute_cosine_similarity(model, word_to_index, word_one, word_two):
    try:
        word_one_index = word_to_index[word_one]
        word_two_index = word_to_index[word_two]
    except KeyError:
        return 0

    embedding_one = model.target_embeddings(torch.LongTensor([word_one_index]))
    embedding_two = model.target_embeddings(torch.LongTensor([word_two_index]))
    similarity = 1 - abs(float(cosine(embedding_one.detach().squeeze().numpy(),
                                      embedding_two.detach().squeeze().numpy())))
    return similarity


print(get_neighbors(model, corpus.word_to_index, "january"))
