import torch
import torch.nn as nn
from rope import RoPE


class SelectAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        vocab_size: int,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.rope = RoPE(
            d_head=self.d_head,
            max_seq_len=max_seq_len,
        )

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.pre_attention_layernorm = nn.LayerNorm(d_model)

        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)

        # Scores from each head are concatenated
        self.ffn = nn.Sequential(
            nn.Linear(max_seq_len * num_heads, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, max_seq_len * num_heads, bias=False)
        )

        self.final_layernorm = nn.LayerNorm(max_seq_len)

        self.output_linear = nn.Linear(max_seq_len, vocab_size, bias=False)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        # (B, L)
        emb_a = self.embedding(a)
        emb_b = self.embedding(b)

        emb_a = self.pre_attention_layernorm(emb_a)
        emb_b = self.pre_attention_layernorm(emb_b)

        q = self.q_linear(emb_a)
        k = self.k_linear(emb_b)

        B, L, _ = q.shape

        # --------------------------
        # Split into heads
        # (B, L, D) -> (B, H, L, Dh)
        # --------------------------

        q = q.view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.d_head).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k)

        # --------------------------
        # Attention scores
        # (B, H, L, L)
        # --------------------------

        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        ) / (self.d_head ** 0.5)

        # --------------------------
        # Merge heads
        # (B, H, L, L)
        # ->
        # (B, L, H*L)
        # --------------------------

        scores = (
            scores
            .transpose(1, 2)
            .reshape(B, L, self.num_heads * L)
        )

        ffn_out = self.ffn(scores)

        norm = self.final_layernorm(ffn_out)

        logits = self.output_linear(norm)

        return logits