from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import spacy
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import mean_squared_error


reviews = pd.read_csv("reviews.csv")
print(reviews.shape)
print(reviews.head())

reviews['summary'] = reviews['summary'].fillna('')
reviews = reviews[['summary', 'grade']]
reviews.columns = ['summary', 'grade']
reviews['review_length'] = reviews['summary'].apply(lambda x: len(". ".join(x.split(". ")[0:2]).split()))
print(reviews.head())

zero_numbering = {"E": 0, "E10+": 1, "T": 2, "M": 3, "AO": 4}
reviews['grade'] = reviews['grade'].apply(lambda x: zero_numbering[x])
print(np.mean(reviews['review_length']))

tok = spacy.load('en_core_web_sm')


def tokenize(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # remove punctuation and numbers
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]


counts = Counter()
for index, row in reviews.iterrows():
    counts.update(tokenize(row['summary']))
print("num_words:", len(counts.keys()))

vocab2index = {"": 0, "UNK": 1}
words = ["", "UNK"]
for word in counts:
    vocab2index[word] = len(words)
    words.append(word)


def encode_sentence(text, vocab2index, N=50):
    text = ". ".join(text.split(". ")[0:2])
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"])
                    for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length


reviews['encoded'] = reviews['summary'].apply(
    lambda x: np.array(encode_sentence(x, vocab2index)))
print(reviews.head())
print(Counter(reviews['grade']))

X = list(reviews['encoded'])
y = list(reviews['grade'])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)


class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]


train_ds = ReviewsDataset(X_train, y_train)
valid_ds = ReviewsDataset(X_valid, y_valid)


def train_model(model, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x, l)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
        if i % 5 == 1:
            print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (
                sum_loss/total, val_loss, val_acc, val_rmse))


def validation_metrics(model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y, l in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x, l)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred,
                            y.unsqueeze(-1)))*y.shape[0]
    return sum_loss/total, correct/total, sum_rmse/total


batch_size = 500
vocab_size = len(words)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(valid_ds, batch_size=batch_size)


class LSTM_variable_input(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.3)
        self.embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)

    def forward(self, x, s):
        x = self.embeddings(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(
            x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.linear(ht[-1])
        return out


model = LSTM_variable_input(vocab_size, 50, 50)
train_model(model, epochs=30, lr=0.1)
train_model(model, epochs=30, lr=0.05)
train_model(model, epochs=30, lr=0.02)
train_model(model, epochs=30, lr=0.01)
