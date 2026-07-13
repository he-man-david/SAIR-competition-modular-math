import torch
from torch import nn


class RoPE(nn.Module):
    def __init__(self, d_head, max_seq_len=1024, theta=10000):
        super().__init__()

        freq = 1.0 / (
            theta ** (torch.arange(0, d_head, 2).float() / d_head)
        )

        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, freq)

        self.register_buffer("cos", angles.cos())
        self.register_buffer("sin", angles.sin())

    def forward(self, q, k):
        """
        q, k shape:
            (batch_size, num_heads, seq_len, d_head)
        """

        seq_len = q.shape[2]

        cos = self.cos[:seq_len][None, None, :, :]
        sin = self.sin[:seq_len][None, None, :, :]

        q = self.rotate(q, cos, sin)
        k = self.rotate(k, cos, sin)

        return q, k

    def rotate(self, x, cos, sin):
        """
        x shape:
            (batch_size, num_heads, seq_len, d_head)
        """

        # Split even and odd features
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        # Rotate each feature pair
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Interleave them back together
        return torch.stack(
            (rotated_x1, rotated_x2),
            dim=-1
        ).flatten(-2)