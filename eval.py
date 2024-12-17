import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


warnings.filterwarnings("ignore")


class FootballDataset(Dataset):
    def __init__(self, folder_path, vocab_size, seq_len):
        self.sentences = []
        self.labels = []
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}

        self._build_vocab_and_load_data(folder_path)

    def _build_vocab_and_load_data(self, folder_path):
        word_counts = {}
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(folder_path, file_name)
                import pandas as pd

                df = pd.read_csv(file_path, encoding="latin1")
                self.sentences.extend(df.iloc[:, 2].astype(str).tolist())
                self.labels.extend(df.iloc[:, 3].astype(int).tolist())

                for sentence in df.iloc[:, 2].astype(str).tolist():
                    for word in sentence.split():
                        word = word.lower()
                        word_counts[word] = word_counts.get(word, 0) + 1

        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for idx, (word, _) in enumerate(sorted_words[: self.vocab_size - 2], start=2):
            self.word_to_idx[word] = idx

    def _tokenize(self, sentence):
        tokens = []
        for word in sentence.split():
            word = word.lower()
            tokens.append(self.word_to_idx.get(word, self.word_to_idx["<UNK>"]))
        return tokens

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        input_ids = self._tokenize(sentence)
        if len(input_ids) < self.seq_len:
            input_ids.extend([self.word_to_idx["<PAD>"]] * (self.seq_len - len(input_ids)))
        else:
            input_ids = input_ids[: self.seq_len]
        return torch.tensor(input_ids, dtype=torch.long), label


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


if __name__ == "__main__":
    # Parameters (must match those used during training)
    vocab_size = 10000
    seq_len = 10  # 10 - 50
    embed_size = 128  # 128 - 256
    num_layers = 3  # 3 - 4
    num_heads = 4  # 4
    forward_expansion = 4  # 4 - 512
    num_classes = 2  # Binary classification (No Goal or Goal)
    dropout = 0.1
    batch_size = 256  # 256 - 32
    weighted_sampler = True

    # Parameters (must match those used during training)
    folder_path = "/mnt/Data/lf/SoccerNetClip10Videos/Laliga/csvfiles"
    model_save_path = "trained_model.pt"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

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

    # Initialize model and load weights
    model = Transformer(vocab_size, embed_size, num_layers, num_heads, seq_len, num_classes, forward_expansion, dropout).to(device)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    all_predictions = []
    all_targets = []

    # Evaluate on multiple batches
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    # Print some results
    print("Sample Predicted labels: ", all_predictions[:50])
    print("Sample Actual labels:    ", all_targets[:50])
    print(f"\nTotal samples evaluated: {len(all_predictions)}")
