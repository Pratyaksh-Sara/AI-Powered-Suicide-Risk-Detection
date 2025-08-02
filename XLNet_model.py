import torch
import torch.nn as nn
import torch.optim as optim
from transformers import XLNetModel, XLNetTokenizer
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load Data
reddit_data = pd.read_csv("Sdd2.csv", encoding='utf-8')
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
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
train_tokens = tokenizer(X_train.tolist(), padding="max_length", truncation=True, max_length=128, return_tensors="pt")
test_tokens = tokenizer(X_test.tolist(), padding="max_length", truncation=True, max_length=128, return_tensors="pt")

# Define XLNet-Based Model
class XLNetSuicideModel(nn.Module):
    def __init__(self):
        super(XLNetSuicideModel, self).__init__()
        self.xlnet = XLNetModel.from_pretrained("xlnet-base-cased")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            output = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(output[0][:, 0, :])
        x = self.fc(x)
        return self.sigmoid(x)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XLNetSuicideModel().to(device)

# Loss & Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Convert y_train to Tensor
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)

# Training Loop
num_epochs = 12
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_tokens["input_ids"].to(device), train_tokens["attention_mask"].to(device)).squeeze()
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Save Model
torch.save(model.state_dict(), "xlnet_suicide_model.pth")

# Evaluation
model.eval()
predictions = model(test_tokens["input_ids"].to(device), test_tokens["attention_mask"].to(device)).squeeze().cpu().detach().numpy()
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
