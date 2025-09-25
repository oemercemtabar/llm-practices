import argparse, math, random
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

def read_text(path):
    return Path(path).read_text(encoding="utf-8")

def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for ch,i in stoi.items()}
    return stoi, itos

def encode(text, stoi):
    return torch.tensor([stoi[ch] for ch in text if ch in stoi], dtype=torch.long)

def get_batches(data, seq_len, batch_size, device):
    n = (len(data) - 1) // seq_len
    if n <= 0:
        return
    idx = torch.arange(0, n*seq_len, seq_len)
    idx = idx[: (len(idx)//batch_size)*batch_size]
    if len(idx) == 0:
        yield data[:-1].unsqueeze(0).to(device), data[1:].unsqueeze(0).to(device)
        return
    idx = idx.view(-1, batch_size).t()
    for row in idx:
        x, y = [], []
        for start in row:
            start = start.item()
            chunk = data[start:start+seq_len+1]
            x.append(chunk[:-1]); y.append(chunk[1:])
        yield torch.stack(x).to(device), torch.stack(y).to(device)

class CharRNN(nn.Module):
    def __init__(self, vocab, embed=256, hidden=512, num_layers=2, dropout=0.1, cell="gru"):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        rnn_cls = nn.GRU if cell.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(embed, hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        return self.head(out), h

def estimate_ppl(loss): return math.exp(min(20.0, loss))

@torch.no_grad()
def sample(model, itos, stoi, device, seed="Once upon a time", steps=400, temp=1.0):
    model.eval()
    if not seed: seed = random.choice(list(itos.values()))
    x = torch.tensor([stoi.get(ch, 0) for ch in seed], dtype=torch.long, device=device).unsqueeze(0)
    _, h = model(x[:, :-1])
    cur = x[:, -1:]
    out = [ch for ch in seed]
    for _ in range(steps):
        logits, h = model(cur, h)
        logits = logits[:, -1, :] / max(1e-6, temp)
        probs = torch.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1)
        out.append(itos[idx.item()])
        cur = idx
    return "".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--embed", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed-text", type=str, default="Once upon a time")
    ap.add_argument("--cell", type=str, default="gru", choices=["gru","lstm"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    text = read_text(args.data)
    stoi, itos = build_vocab(text)
    data = encode(text, stoi)

    device = torch.device(args.device)
    model = CharRNN(len(stoi), args.embed, args.hidden, args.layers, args.dropout, args.cell).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs+1):
        model.train()
        total, tokens = 0.0, 0
        for xb, yb in get_batches(data, args.seq_len, args.batch_size, device) or []:
            logits, _ = model(xb)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            opt.zero_grad(); loss.backward()
            clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total += loss.item() * yb.numel()
            tokens += yb.numel()
        if tokens == 0:
            print("Not enough data. Add more text to data/input.txt"); return
        avg = total / tokens
        print(f"epoch {epoch:02d} | loss {avg:.4f} | ppl {estimate_ppl(avg):.2f}")
        print("-"*60)
        print(sample(model, itos, stoi, device, seed=args.seed_text, steps=300, temp=args.temp))
        print("-"*60)

    for t in [0.5, 0.8, 1.0, 1.2]:
        print(f"\n=== Temperature {t} ===")
        print(sample(model, itos, stoi, device, seed=args.seed_text, steps=500, temp=t))

if __name__ == "__main__":
    main()
