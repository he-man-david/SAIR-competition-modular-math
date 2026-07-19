import torch
import torch.nn as nn


class BilinearArithmeticNetwork(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        vocab_size: int,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.input_layernorm = nn.LayerNorm(d_model)

        self.bilinear_w = nn.Linear(d_model, d_model, bias=False)

        self.interaction_proj = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_reasoner = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, vocab_size),
        )

    def add_pe(self, x: torch.Tensor):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device)
        return x + self.pos_emb(pos).unsqueeze(0)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        emb_a = self.embedding(a)
        emb_b = self.embedding(b)

        emb_a = self.add_pe(emb_a)
        emb_b = self.add_pe(emb_b)

        emb_a = self.input_layernorm(emb_a)
        emb_b = self.input_layernorm(emb_b)

        bilinear_a = self.bilinear_w(emb_a)
        bilinear_scores = torch.matmul(bilinear_a, emb_b.transpose(-2, -1))
        bilinear_weights = torch.softmax(bilinear_scores / (self.d_model ** 0.5), dim=-1)
        bilinear_context = torch.matmul(bilinear_weights, emb_b)

        interaction = torch.cat(
            [
                emb_a,
                emb_b,
                emb_a * emb_b,
                bilinear_context,
            ],
            dim=-1,
        )

        x = self.interaction_proj(interaction)
        x = x + emb_a + emb_b

        x = self.sequence_reasoner(x)

        logits = self.output_head(x)

        return logits