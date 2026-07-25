import torch
import torch.nn as nn


class BilinearTransformerNetwork(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        vocab_size: int,
        pad_id: int,
        eq_id: int,
        eos_id: int,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.eq_id = eq_id
        self.eos_id = eos_id

        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        self.bilinear_w = nn.Linear(d_model, d_model, bias=False)

        self.interaction_proj = nn.Sequential(
            nn.Linear(5 * d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=8 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=8 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, vocab_size),
        )

    def add_pe(self, x: torch.Tensor):
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        pos = torch.arange(seq_len, device=x.device)
        return x + self.pos_emb(pos).unsqueeze(0)

    def get_causal_mask(self, seq_len: int, device: torch.device):
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def get_pad_mask(self, x: torch.Tensor):
        return x == self.pad_id

    def get_memory_pad_mask(self, a: torch.Tensor, b: torch.Tensor):
        return self.get_pad_mask(a) & self.get_pad_mask(b)

    def make_decoder_input(self, tgt: torch.Tensor):
        start_tokens = torch.full(
            (tgt.size(0), 1),
            self.eq_id,
            dtype=tgt.dtype,
            device=tgt.device,
        )
        return torch.cat([start_tokens, tgt[:, :-1]], dim=1)

    def create_bilinear_context(self, emb_a: torch.Tensor, emb_b: torch.Tensor, real_mask: torch.Tensor):
        bilinear = self.bilinear_w(emb_a)
        bilinear_scores = torch.matmul(
            bilinear,
            emb_b.transpose(-2, -1),
        )

        bilinear_scores = bilinear_scores.masked_fill(
            ~real_mask.unsqueeze(1),
            torch.finfo(bilinear_scores.dtype).min,
        )

        all_pad = ~real_mask.any(dim=1)
        if all_pad.any():
            bilinear_scores[all_pad] = 0.0

        bilinear_weights = torch.softmax(
            bilinear_scores / (self.d_model ** 0.5),
            dim=-1,
        )

        bilinear_context = torch.matmul(
            bilinear_weights,
            emb_b,
        )

        return bilinear_context

    def encode_inputs(self, a: torch.Tensor, b: torch.Tensor):
        memory_pad_mask = self.get_memory_pad_mask(a, b)

        emb_a = self.embedding(a)
        emb_b = self.embedding(b)

        b_real_mask = ~self.get_pad_mask(b)
        bilinear_context_ab = self.create_bilinear_context(emb_a, emb_b, b_real_mask)
        a_real_mask = ~self.get_pad_mask(a)
        bilinear_context_ba = self.create_bilinear_context(emb_b, emb_a, a_real_mask)
        
        emb_prod = emb_a * emb_b

        interaction = torch.cat(
            [
                emb_a,
                emb_b,
                emb_prod,
                bilinear_context_ab,
                bilinear_context_ba
            ],
            dim=-1,
        )

        x = self.interaction_proj(interaction)
        x = x + bilinear_context_ab + bilinear_context_ba + emb_prod
        
        x = x.masked_fill(memory_pad_mask.unsqueeze(-1), 0.0)
        x = self.add_pe(x)
        
        memory = self.encoder(
            x,
            src_key_padding_mask=memory_pad_mask,
        )

        return memory, memory_pad_mask

    def decode_targets(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_pad_mask: torch.Tensor,
    ):
        decoder_input = self.make_decoder_input(tgt)
        decoder_pad_mask = self.get_pad_mask(decoder_input)
        causal_mask = self.get_causal_mask(decoder_input.size(1), decoder_input.device)

        emb_tgt = self.embedding(decoder_input)
        emb_tgt = self.add_pe(emb_tgt)

        dec_out = self.decoder(
            tgt=emb_tgt,
            memory=memory,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_mask=causal_mask,
        )

        logits = self.output_head(dec_out)

        return logits

    def forward(self, a: torch.Tensor, b: torch.Tensor, tgt: torch.Tensor):
        memory, memory_pad_mask = self.encode_inputs(a, b)
        logits = self.decode_targets(tgt, memory, memory_pad_mask)

        return logits

    @torch.no_grad()
    def predict(self, a: torch.Tensor, b: torch.Tensor, max_new_tokens: int | None = None):
        self.eval()

        if max_new_tokens is None:
            max_new_tokens = self.max_seq_len

        memory, memory_pad_mask = self.encode_inputs(a, b)

        generated = torch.full(
            (a.size(0), 1),
            self.eq_id,
            dtype=a.dtype,
            device=a.device,
        )

        finished = torch.zeros(a.size(0), dtype=torch.bool, device=a.device)

        for _ in range(max_new_tokens):
            decoder_pad_mask = self.get_pad_mask(generated)
            causal_mask = self.get_causal_mask(generated.size(1), generated.device)

            emb_tgt = self.embedding(generated)
            emb_tgt = self.add_pe(emb_tgt)

            dec_out = self.decoder(
                tgt=emb_tgt,
                memory=memory,
                tgt_key_padding_mask=decoder_pad_mask,
                memory_key_padding_mask=memory_pad_mask,
                tgt_mask=causal_mask,
            )

            logits = self.output_head(dec_out)
            next_token = logits[:, -1, :].argmax(dim=-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.pad_id),
                next_token,
            )

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == self.eos_id)

            if finished.all():
                break

            if generated.size(1) >= self.max_seq_len:
                break

        predictions = generated[:, 1:]

        if predictions.size(1) < self.max_seq_len:
            pad = torch.full(
                (predictions.size(0), self.max_seq_len - predictions.size(1)),
                self.pad_id,
                dtype=predictions.dtype,
                device=predictions.device,
            )
            predictions = torch.cat([predictions, pad], dim=1)
        else:
            predictions = predictions[:, :self.max_seq_len]

        return predictions