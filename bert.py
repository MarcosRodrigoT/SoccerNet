import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import BertTokenizerFast, BertForSequenceClassification


# -------------------
# Dataset for football transcription with BERT
# -------------------
class FootballDataset(Dataset):
    def __init__(self, folder_path, tokenizer, max_length=50):
        self.sentences = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self._load_data(folder_path)

    def _load_data(self, folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path, encoding="latin1")
                self.sentences.extend(df.iloc[:, 2].astype(str).tolist())
                self.labels.extend(df.iloc[:, 3].astype(int).tolist())

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        # Tokenize using BERT tokenizer
        encoding = self.tokenizer(sentence, add_special_tokens=True, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")

        input_ids = encoding["input_ids"].squeeze(0)  # [max_length]
        attention_mask = encoding["attention_mask"].squeeze(0)  # [max_length]

        return input_ids, attention_mask, label


# -------------------
# Training code
# -------------------

# Parameters
max_length = 50  # Adjust as needed
batch_size = 32
epochs = 10
learning_rate = 1e-5  # Lower LR is typical for finetuning BERT
model_name = "bert-base-uncased"  # Choose a model suitable for your language/data
folder_path = "/mnt/Data/lf/SoccerNetClip10Videos/Laliga/csvfiles"
model_save_path = "bert_trained_model.pt"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# Data preparation
dataset = FootballDataset(folder_path, tokenizer, max_length=max_length)

# Compute class weights for WeightedRandomSampler if needed (assuming binary: 0-NoGoal, 1-Goal)
targets_list = dataset.labels
class_counts = [sum(1 for t in targets_list if t == c) for c in [0, 1]]
class_weights = [1.0 / c for c in class_counts]
sample_weights = [class_weights[t] for t in targets_list]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# You can adjust the weights for cross-entropy if desired
# weights = torch.tensor([1.0, 15.0], dtype=torch.float).to(device) # example weighting
# criterion = nn.CrossEntropyLoss(weight=weights)
# BertForSequenceClassification includes its own classification head which outputs logits.
criterion = nn.CrossEntropyLoss()

# Optimizer (AdamW is recommended for BERT models)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Metrics storage
losses = []
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Training loop
for epoch in range(epochs):
    model.train()
    correct = 0
    total = 0
    epoch_loss = 0

    all_targets = []
    all_predictions = []

    for input_ids, attention_mask, targets in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=targets)
        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    avg_loss = epoch_loss / len(dataloader)
    accuracy = correct / total
    losses.append(avg_loss)
    accuracies.append(accuracy)

    # Compute evaluation metrics for minority class (class=1)
    precision = precision_score(all_targets, all_predictions, pos_label=1, zero_division=0)
    recall = recall_score(all_targets, all_predictions, pos_label=1, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, pos_label=1, zero_division=0)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision (class=1): {precision:.4f}, Recall (class=1): {recall:.4f}, F1 (class=1): {f1:.4f}")

# Save the model
torch.save(model.state_dict(), model_save_path)
print(f"\nModel saved to {model_save_path}\n")

# Visualization
plt.figure(figsize=(10, 8))

# Top subplot: Loss and Accuracy
plt.subplot(2, 1, 1)
plt.plot(range(1, epochs + 1), losses, label="Loss", marker="o")
plt.plot(range(1, epochs + 1), accuracies, label="Accuracy", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Loss and Accuracy over Epochs")
plt.legend()

# Bottom subplot: Precision, Recall, and F1-score (Minority Class)
plt.subplot(2, 1, 2)
plt.plot(range(1, epochs + 1), recalls, label="Recall (class=1)", marker="o")
plt.plot(range(1, epochs + 1), precisions, label="Precision (class=1)", marker="o")
plt.plot(range(1, epochs + 1), f1_scores, label="F1 (class=1)", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Minority Class Metrics over Epochs")
plt.legend()

plt.tight_layout()
plt.savefig("bert_training.png")

# After training evaluation on a batch
model.eval()
test_iter = iter(dataloader)
input_ids, attention_mask, targets = next(test_iter)

input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
targets = targets.to(device)

with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    _, predicted = torch.max(logits, 1)

print("Predicted labels -  Actual labels")
for prediction, target in zip(predicted.cpu().numpy(), targets.cpu().numpy()):
    print(f"\t{prediction}\t -\t {target}")
