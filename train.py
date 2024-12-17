import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score


# Dataset for football transcription
class FootballDataset(Dataset):
    def __init__(self, folder_path, vocab_size, seq_len):
        self.sentences = []
        self.labels = []
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}  # Special tokens for padding and unknown words

        # Build the vocabulary and load data
        self._build_vocab_and_load_data(folder_path)

    def _build_vocab_and_load_data(self, folder_path):
        word_counts = {}

        # Load data and count word frequencies
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path, encoding="latin1")
                self.sentences.extend(df.iloc[:, 2].astype(str).tolist())
                self.labels.extend(df.iloc[:, 3].astype(int).tolist())

                for sentence in df.iloc[:, 2].astype(str).tolist():
                    for word in sentence.split():
                        word = word.lower()  # Normalize case
                        word_counts[word] = word_counts.get(word, 0) + 1

        # Build vocabulary based on word frequencies
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for idx, (word, _) in enumerate(sorted_words[: self.vocab_size - 2], start=2):  # Reserve 0 and 1
            self.word_to_idx[word] = idx

    def _tokenize(self, sentence):
        tokens = []
        for word in sentence.split():
            word = word.lower()  # Normalize case
            tokens.append(self.word_to_idx.get(word, self.word_to_idx["<UNK>"]))
        return tokens

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        # Tokenize sentence and pad/truncate to `self.seq_len`
        input_ids = self._tokenize(sentence)
        if len(input_ids) < self.seq_len:
            input_ids.extend([self.word_to_idx["<PAD>"]] * (self.seq_len - len(input_ids)))
        else:
            input_ids = input_ids[: self.seq_len]

        return torch.tensor(input_ids, dtype=torch.long), label


# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes], softmax probabilities
        # targets: [batch_size], ground truth labels
        prob = torch.gather(inputs, dim=1, index=targets.unsqueeze(1)).squeeze(1)
        focal_weight = self.alpha * (1 - prob) ** self.gamma
        log_prob = torch.log(prob)
        loss = -focal_weight * log_prob

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# Transformer components
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        batch_size, seq_len, embed_size = x.size()
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        energy = torch.einsum("bhqd,bhkd->bhqk", Q, K) / (self.head_dim**0.5)
        attention = torch.softmax(energy, dim=-1)
        out = torch.einsum("bhqk,bhvd->bhqd", attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_size)
        return self.fc_out(out)


class FeedForward(nn.Module):
    def __init__(self, embed_size, forward_expansion):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.fc2 = nn.Linear(forward_expansion * embed_size, embed_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.attention(x)
        x = self.norm1(x + self.dropout(attention))
        feed_forward = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feed_forward))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, seq_len, num_classes, forward_expansion, dropout):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, embed_size))
        self.layers = nn.ModuleList([TransformerBlock(embed_size, num_heads, forward_expansion, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        batch_size, seq_len = x.size()
        embeddings = self.embedding(x)
        x = embeddings + self.positional_encoding[:, :seq_len, :]

        for layer in self.layers:
            x = layer(x)

        x = x.mean(dim=1)
        return self.fc_out(x)


# Parameters
vocab_size = 10000
seq_len = 50
embed_size = 256  # 128
num_layers = 4  # 3
num_heads = 4  # 4
forward_expansion = 512  # 4
num_classes = 2  # Binary classification (No Goal or Goal)
dropout = 0.1
batch_size = 32
epochs = 20
weighted_loss = [1.0, 1.0]  # weight loss for [No Goal, Goal] -> This could be [1 / freq_of_no_goal, 1 / freq_of_goal] or a simpler ratio
weighted_sampler = True

# File paths
folder_path = "/mnt/Data/lf/SoccerNetClip10Videos/Laliga/csvfiles"
model_save_path = "trained_model.pt"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data preparation
dataset = FootballDataset(folder_path, vocab_size, seq_len)
if weighted_sampler:
    # Compute class weights for WeightedRandomSampler
    targets_list = dataset.labels
    class_counts = [sum(1 for t in targets_list if t == c) for c in [0, 1]]
    class_weights = [1.0 / c for c in class_counts]
    sample_weights = [class_weights[t] for t in targets_list]

    # Goal: 468 instances - No Goal: 6996 instances

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
else:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = Transformer(vocab_size, embed_size, num_layers, num_heads, seq_len, num_classes, forward_expansion, dropout).to(device)

# Criterion
weights = torch.tensor(weighted_loss, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
# criterion = FocalLoss(alpha=1, gamma=2)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
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
plt.savefig("training.png")

# After training
model.eval()
test_iter = iter(dataloader)
inputs, targets = next(test_iter)

inputs = inputs.to(device)
targets = targets.to(device)
with torch.no_grad():
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)

print("Predicted labels: ", predicted.cpu().numpy())
print("Actual labels:    ", targets.cpu().numpy())
