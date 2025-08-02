import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load Data
try:
    reddit_data = pd.read_csv("Sdd2.csv", encoding='latin-1')
except UnicodeDecodeError:
    reddit_data = pd.read_csv("Sdd2.csv", encoding='utf-8', errors='replace')
twitter_data = pd.read_csv("TW.csv")

# Preprocess Data
reddit_data = reddit_data[['text', 'target']].rename(columns={"target": "label"})
twitter_data = twitter_data.rename(columns={"target": "label"})
reddit_data['label'] = reddit_data['label'].map({"suicide": 1, "non-suicide": 0})
combined_data = pd.concat([reddit_data[['text', 'label']], twitter_data[['text', 'label']]], ignore_index=True)
combined_data.dropna(inplace=True)
combined_data['text'] = combined_data['text'].apply(lambda x: re.sub(r'[^A-Za-z0-9\s]', '', x))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(combined_data["text"], combined_data["label"], test_size=0.2, random_state=42)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_tokens = tokenizer(X_train.tolist(), padding="max_length", truncation=True, max_length=128, return_tensors="pt")["input_ids"]
test_tokens = tokenizer(X_test.tolist(), padding="max_length", truncation=True, max_length=128, return_tensors="pt")["input_ids"]

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(x)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = tokenizer.vocab_size
model = LSTMModel(vocab_size).to(device)

# Loss & Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 12
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_tokens.to(device)).squeeze()
    loss = criterion(outputs, y_train.to(device).float())
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Save Model
torch.save(model.state_dict(), "lstm_suicide_model.pth")

# Evaluation
model.eval()
predictions = model(test_tokens.to(device)).squeeze().cpu().detach().numpy()
binary_preds = (predictions >= 0.5).astype(int)

# Performance Metrics
accuracy = accuracy_score(y_test, binary_preds)
precision = precision_score(y_test, binary_preds)
recall = recall_score(y_test, binary_preds)
f1 = f1_score(y_test, binary_preds)
roc_auc = roc_auc_score(y_test, predictions)

# Print Metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
