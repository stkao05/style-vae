import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, file_path, seq_len, c2i, split="train"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()

        n = len(data)
        split_idx = int(n * 0.9)
        if split == "train":
            self.data = torch.tensor([c2i[c] for c in data[:split_idx]])
        elif split == "val":
            self.data = torch.tensor([c2i[c] for c in data[split_idx:]])
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + 1 + self.seq_len]
        return x, y


class CharTransfomer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super(CharTransfomer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads
        )
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        causal_mask = torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1).bool().to(x.device)

        # Input shape: (batch, seq)
        x = self.embedding(x)  # (batch, seq, emb)
        x = x.permute(1, 0, 2)  # (seq, batch, emb)
        x = self.decoder(x, mask=causal_mask, is_causal=True)  # (seq, batch, emb)
        x = x.permute(1, 0, 2)  # Back to (batch, seq, emb)
        x = self.output_layer(x)  # (batch, seq, vocab_size)
        return x


@torch.no_grad()
def sample_text(model, i2c, device, max_len=100, start_char_idx=None, temperature=1.0):
    model.eval()
    vocab_size = len(i2c)
    if start_char_idx is None:
        start_char_idx = torch.randint(0, vocab_size, (1,)).item()
    input_seq = torch.tensor([[start_char_idx]], device=device)
    generated_text = [i2c[start_char_idx]]

    for _ in range(max_len - 1):
        outputs = model(input_seq)
        logits = outputs[:, -1, :]
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        next_char_idx = torch.multinomial(probs, num_samples=1).item()
        next_char = i2c[next_char_idx]
        generated_text.append(next_char)
        input_seq = torch.cat(
            [input_seq, torch.tensor([[next_char_idx]], device=device)], dim=1
        )

    return "".join(generated_text)


# python tiny_test.py --embed_dim 512 --num_heads 8 --num_layers 6 --batch_size 64 --seq_len 256 --num_epochs 50 --device cuda
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CharTransformer model.")
    parser.add_argument(
        "--embed_dim", type=int, default=256, help="Embedding dimension."
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads."
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of transformer layers."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length.")
    parser.add_argument(
        "--num_epochs", type=int, default=32, help="Number of training epochs."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to train on (cpu or cuda)."
    )
    args = parser.parse_args()

    with open("tiny.txt", "r", encoding="utf-8") as f:
        data = f.read()

    chars = set([_ for _ in data])
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}

    train_ds = TextDataset("./tiny.txt", seq_len=args.seq_len, c2i=c2i, split="train")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print("using device", device)

    vocab_size = len(chars)
    model = CharTransfomer(
        vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    num_epochs = 1
    sample_interval = 250

    for epoch in range(args.num_epochs):
        model.train()
        tq = tqdm(train_dl, desc=f"epoch {epoch+1}/{num_epochs}", leave=True)

        for step, (inputs, targets) in enumerate(tq):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tq.set_postfix(loss=f"{loss.item():.4f}")

            if step % sample_interval == 0:
                tqdm.write(sample_text(model, i2c, device, max_len=100))
