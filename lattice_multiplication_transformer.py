import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatticeMultiplicationLoopTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_seq_len: int = 128,
        d_model: int = 256,
        rows_per_step: int = 10,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.transformer_dim = 2 * d_model
        self.rows_per_step = rows_per_step
        self.product_seq_len = 2 * max_seq_len + 1

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id,
        )

        self.operand_position_encoding_embedding = nn.Embedding(
            max_seq_len,
            d_model,
        )

        self.bilinear_w = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.lattice_cell_mlp = nn.Sequential(
            nn.Linear(
                3 * d_model + 1,
                self.transformer_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                self.transformer_dim,
                self.transformer_dim,
            ),
        )

        self.adjacent_column_convolution = nn.Conv2d(
            in_channels=self.transformer_dim,
            out_channels=self.transformer_dim,
            kernel_size=(1, 2),
            stride=1,
            padding=0,
        )

        self.column_position_encoding_embedding = nn.Embedding(
            max_seq_len,
            self.transformer_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.transformer_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        self.initial_product_state = nn.Parameter(
            torch.zeros(
                1,
                self.product_seq_len,
                self.transformer_dim,
            )
        )

        self.output_head = nn.Linear(
            self.transformer_dim,
            vocab_size,
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = a.shape
        device = a.device

        valid_a = a != self.pad_id
        valid_b = b != self.pad_id

        pair_validity_mask = (
            valid_a.unsqueeze(2)
            & valid_b.unsqueeze(1)
        )

        operand_position_ids = torch.arange(
            seq_len,
            device=device,
        )

        operand_position_encoding = (
            self.operand_position_encoding_embedding(
                operand_position_ids
            )
        )

        embedded_a = (
            self.embedding(a)
            + operand_position_encoding.unsqueeze(0)
        )

        embedded_b = (
            self.embedding(b)
            + operand_position_encoding.unsqueeze(0)
        )

        embedded_a = embedded_a.masked_fill(
            ~valid_a.unsqueeze(-1),
            0.0,
        )

        embedded_b = embedded_b.masked_fill(
            ~valid_b.unsqueeze(-1),
            0.0,
        )

        a_grid = embedded_a.unsqueeze(2).expand(
            -1,
            -1,
            seq_len,
            -1,
        )

        b_grid = embedded_b.unsqueeze(1).expand(
            -1,
            seq_len,
            -1,
            -1,
        )

        transformed_a_grid = self.bilinear_w(
            a_grid
        )

        bilinear_scores = (
            transformed_a_grid
            * b_grid
        ).sum(
            dim=-1,
            keepdim=True,
        ) / math.sqrt(self.d_model)

        bilinear_scores = bilinear_scores.masked_fill(
            ~pair_validity_mask.unsqueeze(-1),
            0.0,
        )

        pair_features = torch.cat(
            [
                a_grid,
                b_grid,
                a_grid * b_grid,
                bilinear_scores,
            ],
            dim=-1,
        )

        pair_features = pair_features.masked_fill(
            ~pair_validity_mask.unsqueeze(-1),
            0.0,
        )

        lattice_cell_features = self.lattice_cell_mlp(
            pair_features
        )

        lattice_cell_features = lattice_cell_features.masked_fill(
            ~pair_validity_mask.unsqueeze(-1),
            0.0,
        )

        lattice_features_for_convolution = torch.func.rearrange(
            lattice_cell_features,
            "batch row column features -> batch features row column",
        )

        lattice_features_for_convolution = F.pad(
            lattice_features_for_convolution,
            pad=(1, 0, 0, 0),
            mode="constant",
            value=0.0,
        )

        convolved_lattice_features = (
            self.adjacent_column_convolution(
                lattice_features_for_convolution
            )
        )

        convolved_lattice_features = torch.func.rearrange(
            convolved_lattice_features,
            "batch features row column -> batch row column features",
        )

        product_state = self.initial_product_state.expand(
            batch_size,
            -1,
            -1,
        )

        column_position_ids = torch.arange(
            seq_len,
            device=device,
        )

        column_position_encoding = (
            self.column_position_encoding_embedding(
                column_position_ids
            )
        )

        column_position_encoding = torch.func.rearrange(
            column_position_encoding,
            "column features -> 1 1 column features",
        )

        for current_row_group_start in range(
            0,
            seq_len,
            self.rows_per_step,
        ):
            current_row_group_end = min(
                current_row_group_start
                + self.rows_per_step,
                seq_len,
            )

            current_number_of_rows = (
                current_row_group_end
                - current_row_group_start
            )

            current_lattice_row_group = (
                convolved_lattice_features[
                    :,
                    current_row_group_start:current_row_group_end,
                ]
            )

            current_lattice_row_group_validity_mask = (
                pair_validity_mask[
                    :,
                    current_row_group_start:current_row_group_end,
                ]
            )

            if current_number_of_rows < self.rows_per_step:
                missing_rows = (
                    self.rows_per_step
                    - current_number_of_rows
                )

                current_lattice_row_group = F.pad(
                    current_lattice_row_group,
                    pad=(0, 0, 0, 0, 0, missing_rows),
                    mode="constant",
                    value=0.0,
                )

                current_lattice_row_group_validity_mask = F.pad(
                    current_lattice_row_group_validity_mask,
                    pad=(0, 0, 0, missing_rows),
                    mode="constant",
                    value=False,
                )

            current_lattice_row_group = (
                current_lattice_row_group
                + column_position_encoding
            )

            current_lattice_row_group = (
                current_lattice_row_group.masked_fill(
                    ~current_lattice_row_group_validity_mask.unsqueeze(-1),
                    0.0,
                )
            )

            current_lattice_row_group_sequence = torch.func.rearrange(
                current_lattice_row_group,
                "batch row column features -> batch (row column) features",
            )

            current_lattice_row_group_padding_mask = (
                ~torch.func.rearrange(
                    current_lattice_row_group_validity_mask,
                    "batch row column -> batch (row column)",
                )
            )

            sample_has_valid_lattice_cells = (
                ~current_lattice_row_group_padding_mask
            ).any(
                dim=1
            )

            if not sample_has_valid_lattice_cells.any():
                break

            transformer_safe_padding_mask = (
                current_lattice_row_group_padding_mask.clone()
            )

            samples_without_valid_lattice_cells = (
                ~sample_has_valid_lattice_cells
            )

            if samples_without_valid_lattice_cells.any():
                transformer_safe_padding_mask[
                    samples_without_valid_lattice_cells,
                    0,
                ] = False

                current_lattice_row_group_sequence = (
                    current_lattice_row_group_sequence.clone()
                )

                current_lattice_row_group_sequence[
                    samples_without_valid_lattice_cells,
                    0,
                ] = 0.0

            encoded_lattice_row_group = self.encoder(
                src=current_lattice_row_group_sequence,
                src_key_padding_mask=transformer_safe_padding_mask,
            )

            updated_product_state = self.decoder(
                tgt=product_state,
                memory=encoded_lattice_row_group,
                memory_key_padding_mask=transformer_safe_padding_mask,
            )

            product_state = torch.where(
                sample_has_valid_lattice_cells.view(
                    batch_size,
                    1,
                    1,
                ),
                updated_product_state,
                product_state,
            )

        return self.output_head(
            product_state
        )

    @torch.no_grad()
    def predict(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        was_training = self.training

        self.eval()

        predictions = self.forward(
            a,
            b,
        ).argmax(
            dim=-1
        )

        self.train(
            was_training
        )

        return predictions

    def count_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )