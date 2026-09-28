import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "minigpt_config.json"
MODEL_PATH = PROJECT_DIR / "minigpt_model.pth"
DATA_PATH = PROJECT_DIR / "input.txt"


with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = json.load(f)

block_size = config["block_size"]
n_embd = config["n_embd"]
n_head = config["n_head"]
n_layer = config["n_layer"]
dropout = config["dropout"]

with DATA_PATH.open("r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {character: index for index, character in enumerate(chars)}
itos = {index: character for index, character in enumerate(chars)}


def encode(value):
    unknown = [ch for ch in value if ch not in stoi]
    if unknown:
        raise ValueError(
            "Prompt contains characters outside the training vocabulary: "
            + repr(sorted(set(unknown)))
        )
    return [stoi[ch] for ch in value]


def decode(tokens):
    return "".join(itos[token] for token in tokens)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size)),
        )

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)

        attention_scores = q @ k.transpose(-2, -1)
        attention_scores = attention_scores * (k.shape[-1] ** -0.5)
        attention_scores = attention_scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf"),
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        v = self.value(x)
        return attention_weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        combined = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(combined))


class FeedForward(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class Block(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()
        head_size = embedding_size // num_heads
        self.self_attention = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForward(embedding_size)
        self.layer_norm_1 = nn.LayerNorm(embedding_size)
        self.layer_norm_2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)]
        )
        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        _, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)
        positions = torch.arange(T, device=idx.device)
        position_embeddings = self.position_embedding_table(positions)
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        return self.lm_head(x)


@torch.no_grad()
def generate_text(model, prompt, max_new_tokens, temperature, device):
    model.eval()
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)[:, -1, :]
        logits = logits / temperature
        probabilities = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

    return decode(idx[0].tolist())


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with the trained MiniGPT checkpoint."
    )
    parser.add_argument("--prompt", default="The ", help="Starting text prompt")
    parser.add_argument("--tokens", type=int, default=200, help="Number of new tokens")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (> 0)",
    )
    args = parser.parse_args()

    if args.tokens < 1:
        raise ValueError("--tokens must be at least 1")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than 0")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPT().to(device)

    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(state_dict)

    generated = generate_text(
        model,
        prompt=args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        device=device,
    )

    print(f"Device: {device}")
    print("-" * 60)
    print(generated)


if __name__ == "__main__":
    main()
