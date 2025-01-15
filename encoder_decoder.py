import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import CLIPTokenizer, CLIPTextModel

#############################################
# Dataset
#############################################


class FootballDataset(Dataset):
    def __init__(self, folder_path, block_size=32, clip_model_name="openai/clip-vit-base-patch32", device="cpu"):
        self.sentences = []
        self.labels = []
        self.block_size = block_size

        # Load data
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path, encoding="latin1")
                self.sentences.extend(df.iloc[:, 2].astype(str).tolist())
                self.labels.extend(df.iloc[:, 3].astype(int).tolist())

        # Initialize CLIP tokenizer and model for embedding extraction
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.clip_model = CLIPTextModel.from_pretrained(clip_model_name).to(device)
        self.clip_model.eval()

        # Extract embeddings for all sentences
        # We'll store them in a list of tensors
        self.device = device
        self.embeddings = self._compute_clip_embeddings(self.sentences)

        # How many blocks can we form?
        # Each block is `block_size` sentences. We'll drop the remainder.
        total_sentences = len(self.labels)
        self.num_blocks = total_sentences // self.block_size

    def _compute_clip_embeddings(self, sentences):
        # Compute CLIP text embeddings for each sentence
        # CLIPTextModel outputs hidden states [batch, seq_len, hidden_dim]
        # We'll take the [CLS] token (0th token) hidden state as sentence embedding
        embeddings = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch_sents = sentences[i : i + batch_size]
                enc = self.tokenizer(batch_sents, padding=True, truncation=True, return_tensors="pt")
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                outputs = self.clip_model(input_ids, attention_mask=attention_mask)
                # outputs.last_hidden_state: [batch, seq_len, hidden_dim]
                # We can use the pooled output (if available) or CLS token: For CLIPTextModel,
                # the pooled_output is not directly given. We'll use the hidden_state of [EOS] token.
                # According to CLIP: The last token is [EOS], we can take that representation:
                # The EOS token is at the position input_ids[i].argmax or we can just take last_hidden_state[:,0,:] (CLS)
                # Actually, CLIP uses the final layer's EOS hidden state as the sentence embedding.
                # The EOS token is the last token of the sequence before padding. We'll use `outputs.pooler_output` if available.
                # CLIPTextModel from HF might not have pooler. We'll take the hidden state of the last token of each sequence:
                # A simpler approach: Just take the last_hidden_state mean as a representation, or the first token.
                # Officially, CLIP's text encoder uses the final hidden state of the [EOS] token. The EOS token ID: tokenizer.eos_token_id
                # We'll find the EOS position:
                eos_token_id = self.tokenizer.eos_token_id
                # For each sequence, find EOS token and gather that hidden state:
                last_hidden_state = outputs.last_hidden_state
                emb_batch = []
                for b_idx, ids in enumerate(input_ids):
                    eos_pos = (ids == eos_token_id).nonzero(as_tuple=True)[0][-1].item()  # last EOS
                    sentence_emb = last_hidden_state[b_idx, eos_pos, :]  # [hidden_dim]
                    emb_batch.append(sentence_emb)
                emb_batch = torch.stack(emb_batch, dim=0)  # [batch, hidden_dim]
                embeddings.append(emb_batch.cpu())
        embeddings = torch.cat(embeddings, dim=0)  # [total_sentences, hidden_dim]
        return embeddings

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        # Return a block of sentences and labels
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size

        # encoder_input: embeddings of these sentences [block_size, hidden_dim]
        encoder_input = self.embeddings[start_idx:end_idx]

        # decoder_output (target): labels for these sentences
        block_labels = self.labels[start_idx:end_idx]

        # decoder_input: start token + all previous labels (for teacher forcing)
        # shape [block_size+1]
        # start token = 2 (arbitrary, we will embed this token)
        # We'll use label tokens: 0,1 as before and let's say start token = 2
        start_token = 2
        decoder_input = [start_token] + block_labels[:-1]  # at first step decoder sees start, must predict first label
        # If block_labels = [L1, L2, L3...], decoder_input = [start_token, L1, L2, ..., L_{last-1}]
        # decoder_output = [L1, L2, L3, ...]

        encoder_input = encoder_input.float()  # ensure float
        decoder_input = torch.tensor(decoder_input, dtype=torch.long)
        decoder_output = torch.tensor(block_labels, dtype=torch.long)

        return encoder_input, decoder_input, decoder_output


#############################################
# Model: Encoder-Decoder Transformer
#############################################


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, embed_dim, nhead, num_encoder_layers, num_decoder_layers, dropout=0.1, num_labels=2):
        super().__init__()
        # We assume encoder input is already embedded (CLIP embeddings)
        # Just add positional encodings
        self.embed_dim = embed_dim
        self.pos_encoder = nn.Embedding(5000, embed_dim)  # simplistic positional encoding

        # For decoder, we have label embeddings: we have 2 labels + 1 start token
        # label tokens: 0 (no goal), 1 (goal), 2 (start)
        self.num_label_tokens = num_labels + 1
        self.label_embedding = nn.Embedding(self.num_label_tokens, embed_dim)
        self.label_pos_encoder = nn.Embedding(5000, embed_dim)

        self.transformer = nn.Transformer(d_model=embed_dim, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout=dropout)

        self.fc_out = nn.Linear(embed_dim, num_labels)

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz), 1).bool().to(device)
        return mask

    def forward(self, src, tgt):
        # src: [batch_size, block_size, embed_dim]
        # tgt: [batch_size, block_size+1]
        device = src.device
        batch_size, src_len, _ = src.size()
        batch_size, tgt_len = tgt.size()

        # Add positional encoding to src
        src_positions = torch.arange(src_len, device=device).unsqueeze(0)
        src_emb = src + self.pos_encoder(src_positions)  # [batch_size, src_len, embed_dim]

        # For transformer: [seq_len, batch, embed_dim]
        src_emb = src_emb.transpose(0, 1)  # [src_len, batch_size, embed_dim]

        # Embed target (labels)
        tgt_positions = torch.arange(tgt_len, device=device).unsqueeze(0)
        tgt_emb = self.label_embedding(tgt) + self.label_pos_encoder(tgt_positions)  # [batch_size, tgt_len, embed_dim]
        tgt_emb = tgt_emb.transpose(0, 1)  # [tgt_len, batch_size, embed_dim]

        # Causal mask for decoder
        tgt_mask = self.generate_square_subsequent_mask(tgt_len, device)

        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        # out: [tgt_len, batch_size, embed_dim]

        out = out.transpose(0, 1)  # [batch_size, tgt_len, embed_dim]
        logits = self.fc_out(out)  # [batch_size, tgt_len, num_labels]

        return logits


#############################################
# Training Setup
#############################################

# Parameters
block_size = 32
clip_model_name = "openai/clip-vit-base-patch32"
embed_dim = 512  # CLIP base model typically has 512-dim embeddings
nhead = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.1
num_labels = 2
batch_size = 8
epochs = 10
learning_rate = 1e-4
folder_path = "/mnt/Data/lf/SoccerNetClip10Videos/Laliga/csvfiles"
model_save_path = "transformer_clip_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = FootballDataset(folder_path, block_size=block_size, clip_model_name=clip_model_name, device=device)
# Weighted sampling if needed
targets_list = []
for i in range(len(dataset)):
    _, _, tgt_out = dataset[i]
    targets_list.extend(tgt_out.tolist())

class_counts = [sum(1 for t in targets_list if t == c) for c in [0, 1]]
class_weights = [1.0 / c for c in class_counts]
sample_weights = []
for i in range(len(dataset)):
    _, _, tgt_out = dataset[i]
    # Average weight of block? Or sum?
    # We'll just use the average for all targets in the block to get a single weight
    w = sum(class_weights[t.item()] for t in tgt_out) / len(tgt_out)
    sample_weights.append(w)

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

model = EncoderDecoderTransformer(embed_dim, nhead, num_encoder_layers, num_decoder_layers, dropout, num_labels).to(device)

criterion = nn.CrossEntropyLoss()  # We can add weighting here if desired.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []
accuracies = []
precisions = []
recalls = []
f1_scores = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    all_targets_epoch = []
    all_predictions_epoch = []
    for encoder_input, decoder_input, decoder_output in dataloader:
        # encoder_input: [batch, block_size, embed_dim]
        # decoder_input: [batch, block_size] (actually block_size+1)
        # decoder_output: [batch, block_size]
        encoder_input = encoder_input.to(device)
        decoder_input = decoder_input.to(device)
        decoder_output = decoder_output.to(device)

        optimizer.zero_grad()
        logits = model(encoder_input, decoder_input)  # [batch, tgt_len, num_labels]
        # We want to predict decoder_output: which has shape [batch, block_size]
        # But logits has shape [batch, block_size+1, num_labels]
        # The first token in decoder_input was <start>, so we should ignore the first logit
        # The predictions start from the second position:
        pred_logits = logits[:, 1:, :]  # [batch, block_size, num_labels]
        loss = criterion(pred_logits.reshape(-1, num_labels), decoder_output.reshape(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Compute metrics
        _, predicted = torch.max(pred_logits, 2)  # [batch, block_size]
        all_targets_epoch.extend(decoder_output.cpu().numpy().flatten())
        all_predictions_epoch.extend(predicted.cpu().numpy().flatten())

    avg_loss = epoch_loss / len(dataloader)
    accuracy = sum([p == t for p, t in zip(all_predictions_epoch, all_targets_epoch)]) / len(all_targets_epoch)
    precision = precision_score(all_targets_epoch, all_predictions_epoch, pos_label=1, zero_division=0)
    recall = recall_score(all_targets_epoch, all_predictions_epoch, pos_label=1, zero_division=0)
    f1 = f1_score(all_targets_epoch, all_predictions_epoch, pos_label=1, zero_division=0)

    losses.append(avg_loss)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, Prec(1): {precision:.4f}, Rec(1): {recall:.4f}, F1(1): {f1:.4f}")

torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Visualization
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(range(1, epochs + 1), losses, marker="o", label="Loss")
plt.plot(range(1, epochs + 1), accuracies, marker="o", label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Loss and Accuracy")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(range(1, epochs + 1), precisions, marker="o", label="Precision(1)")
plt.plot(range(1, epochs + 1), recalls, marker="o", label="Recall(1)")
plt.plot(range(1, epochs + 1), f1_scores, marker="o", label="F1(1)")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Precision, Recall, F1 (class=1)")
plt.legend()

plt.tight_layout()
plt.savefig("training.png")

# Test on a batch
model.eval()
test_iter = iter(dataloader)
encoder_input, decoder_input, decoder_output = next(test_iter)
encoder_input = encoder_input.to(device)
decoder_input = decoder_input.to(device)
decoder_output = decoder_output.to(device)

with torch.no_grad():
    logits = model(encoder_input, decoder_input)
    pred_logits = logits[:, 1:, :]
    _, predicted = torch.max(pred_logits, 2)

print("Predicted labels - Actual labels")
for pred_seq, actual_seq in zip(predicted.cpu().numpy(), decoder_output.cpu().numpy()):
    for p, a in zip(pred_seq, actual_seq):
        print(f"{p}\t-\t{a}")
    print("----")
