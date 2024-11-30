# %%
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class TransformerVAE(nn.Module):
    def __init__(
        self, seq_len, embed_dim, latent_dim, num_heads, num_layers, vocab_size
    ):
        super(TransformerVAE, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.enc_project = nn.Linear(embed_dim, 2 * embed_dim)

        self.fc_mu = nn.Linear(embed_dim * seq_len, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim * seq_len, latent_dim)
        self.fc_latent_to_hidden = nn.Linear(latent_dim, embed_dim * seq_len)

        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def encode(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        batch_size, _, embed_dim = x.shape

        x = x.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]
        x = self.encoder(x)  # [seq_len, batch_size, embed_dim]

        x = self.enc_project(x)  # (seq_len, batch_size, 2 * embed_dim)
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, 2 * embed_dim]
        x_mu, x_logvar = x.split(
            split_size=embed_dim, dim=2
        )  # [batch_size, seq_len, embed_dim]
        x_mu = x_mu.reshape(batch_size, -1)  # [batch_size, seq_len * embed_dim]
        x_logvar = x_logvar.reshape(batch_size, -1)

        mu = self.fc_mu(x_mu)
        logvar = self.fc_logvar(x_logvar)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # z: (batch_size, latent)
        z = self.fc_latent_to_hidden(z)  # [batch_size, seq_len * embed_dim]
        z = z.view(
            z.size(0), self.seq_len, self.embed_dim
        )  # [batch_size, seq_len, embed_dim]
        z = z.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]

        causal_mask = (
            torch.tril(torch.ones(self.seq_len, self.seq_len)).bool().to(z.device)
        )
        z = self.decoder(
            z, mask=causal_mask, is_causal=True
        )  # [seq_len, batch_size, embed_dim]
        z = z.permute(1, 0, 2)  # [batch_size, seq_len, embed_dim]
        z = self.output_layer(z)  # [batch_size, seq_len, vocab_size]
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xh = self.decode(z)
        return xh, mu, logvar

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        x = self.decode(z)
        return x


def vae_loss(xh, x, mu, logvar, beta=1.0):
    """
    xh: (batch, seq, vocab)
    x: (batch, seq)
    """
    recon_loss = F.cross_entropy(xh.view(-1, xh.size(-1)), x.view(-1), reduction="mean")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss /= x.size(0) * x.size(1)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def sample(model, device, i2c):
    x = model.sample(1, device)  # (batch, seq, vocab)
    tokens = x.argmax(dim=2)  # (batch, seq, vocab)
    chars = [i2c[_.item()] for _ in tokens.flatten()]
    return "".join(chars)


def main(args):
    with open("tiny.txt", "r", encoding="utf-8") as f:
        data = f.read()

    chars = set([_ for _ in data])
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}

    train_ds = TextDataset("./tiny.txt", seq_len=args.seq_len, c2i=c2i, split="train")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print("using device", device)

    vocab_size = len(chars)
    model = TransformerVAE(
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        vocab_size=vocab_size,
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    sample_interval = 250

    for epoch in range(args.num_epochs):
        tq = tqdm(train_dl, desc=f"epoch {epoch+1}/{args.num_epochs}")

        for step, (inputs, targets) in enumerate(tq):
            inputs, targets = inputs.to(device), targets.to(device)

            xh, mu, logvar = model(inputs)
            loss, _, _ = vae_loss(xh, targets, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tq.set_postfix(loss=f"{loss.item():.4f}")

            if step % sample_interval == 0:
                print(sample(model, device, i2c))


# @dataclass
# class Args:
#     device = "cpu"
#     seq_len = 32
#     batch_size = 2
#     embed_dim = 128
#     latent_dim = 512
#     num_heads = 2
#     num_layers = 2
#     num_epochs = 1


# args = Args()
# main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--embed_dim", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--latent_dim", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    main(args)
