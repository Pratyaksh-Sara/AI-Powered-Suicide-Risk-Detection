import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import AutoModel, AutoTokenizer
import re

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
combined_data['text'] = combined_data['text'].fillna('').apply(lambda x: re.sub(r'[^A-Za-z0-9\s]', '', x))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(combined_data["text"], combined_data["label"].astype(float), test_size=0.2, random_state=42)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_texts(texts, max_length=128):
    return tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# Tokenize Data
train_tokens = tokenize_texts(X_train)
test_tokens = tokenize_texts(X_test)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Pretrained BERT Model
bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)

# Define Custom Model
class SuicideRiskModel(nn.Module):
    def __init__(self, bert_model):
        super(SuicideRiskModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output[0][:, 0, :]
        x = self.dropout(pooled_output)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize Model
model = SuicideRiskModel(bert_model).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Create Dataset Class
class SuicideDataset(Dataset):
    def __init__(self, tokens, labels):
        self.input_ids = tokens["input_ids"]
        self.attention_mask = tokens["attention_mask"]
        self.labels = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

# Create DataLoader
train_dataset = SuicideDataset(train_tokens, y_train)
test_dataset = SuicideDataset(test_tokens, y_test)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Train Model
num_epochs = 12
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Save Model
torch.save(model.state_dict(), "suicide_model.pth")
print("Model saved as 'suicide_model.pth'")

# Evaluate Model
model.eval()
predicted_risk_scores = []
true_labels = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].cpu().numpy()
        outputs = model(input_ids, attention_mask).cpu().numpy()
        predicted_risk_scores.extend(outputs)
        true_labels.extend(labels)

# Print Sample Predictions
for i in range(5):
    print(f"Text: {X_test.iloc[i]}\nPredicted Risk Score: {predicted_risk_scores[i][0]:.4f}\n")

# Convert to binary predictions
binary_predictions = [1 if score >= 0.5 else 0 for score in predicted_risk_scores]

# Compute Metrics
accuracy = accuracy_score(true_labels, binary_predictions)
precision = precision_score(true_labels, binary_predictions)
recall = recall_score(true_labels, binary_predictions)
f1 = f1_score(true_labels, binary_predictions)
roc_auc = roc_auc_score(true_labels, predicted_risk_scores)

# Print Accuracy Metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
