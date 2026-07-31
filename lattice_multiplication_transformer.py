import torch
import torch.nn as nn


class LatticeMultiplicationTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        pad_id: int,
        eos_id: int,
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.product_seq_len = 2 * max_seq_len + 1

        self.pad_id = pad_id
        self.eos_id = eos_id
        self.d_model = d_model

        self.max_carry_value = 18 * max_seq_len
        self.carry_vocab_size = self.max_carry_value + 1

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_id,
        )

        self.operand_position_embedding = nn.Embedding(
            max_seq_len,
            d_model,
        )

        self.product_position_embedding = nn.Embedding(
            self.product_seq_len,
            d_model,
        )

        self.a_projection = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.cell_mlp = nn.Sequential(
            nn.Linear(
                3 * d_model + 1,
                2 * d_model,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                2 * d_model,
                2 * d_model,
            ),
            nn.GELU(),
        )

        self.ones_cell_head = nn.Linear(
            d_model,
            10,
        )

        self.tens_cell_head = nn.Linear(
            d_model,
            10,
        )

        self.digit_embedding = nn.Embedding(
            10,
            d_model,
        )

        self.diagonal_input_projection = nn.Sequential(
            nn.Linear(
                d_model + 2,
                d_model,
            ),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.carry_cell = nn.GRUCell(
            input_size=d_model,
            hidden_size=d_model,
        )

        self.carry_output_norm = nn.LayerNorm(
            d_model
        )

        self.product_head = nn.Linear(
            d_model,
            vocab_size,
        )

        self.carry_head = nn.Linear(
            d_model,
            self.carry_vocab_size,
        )

        self.initial_carry_state = nn.Parameter(
            torch.zeros(
                1,
                d_model,
            )
        )

        self.register_buffer(
            "digit_values",
            torch.arange(
                10,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def _embed_number(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        _, seq_len = token_ids.shape

        positions = torch.arange(
            seq_len,
            device=token_ids.device,
        ).unsqueeze(0)

        return (
            self.embedding(token_ids)
            + self.operand_position_embedding(
                positions
            )
        )

    def _create_cell_features(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        _, seq_len = a.shape

        embedded_a = self._embed_number(a)
        embedded_b = self._embed_number(b)

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

        bilinear_score = (
            self.a_projection(a_grid)
            * b_grid
        ).sum(
            dim=-1,
            keepdim=True,
        ) / (self.d_model ** 0.5)

        pair_features = torch.cat(
            [
                a_grid,
                b_grid,
                a_grid * b_grid,
                bilinear_score,
            ],
            dim=-1,
        )

        cell_features = self.cell_mlp(
            pair_features
        )

        ones_features, tens_features = (
            cell_features.chunk(
                2,
                dim=-1,
            )
        )

        valid_cells = (
            (a != self.pad_id).unsqueeze(2)
            & (b != self.pad_id).unsqueeze(1)
        )

        return (
            ones_features,
            tens_features,
            valid_cells,
        )

    def _soft_digit_representation(
        self,
        logits: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        probabilities = torch.softmax(
            logits.float(),
            dim=-1,
        )

        digit_embeddings = (
            probabilities
            @ self.digit_embedding.weight.float()
        )

        expected_values = (
            probabilities
            * self.digit_values
        ).sum(
            dim=-1,
            keepdim=True,
        )

        return (
            digit_embeddings.to(logits.dtype),
            expected_values.to(logits.dtype),
        )

    def _create_diagonal_state(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        ones_cell_logits: torch.Tensor,
        tens_cell_logits: torch.Tensor,
        valid_cells: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size, seq_len = a.shape

        (
            ones_embeddings,
            ones_values,
        ) = self._soft_digit_representation(
            ones_cell_logits
        )

        (
            tens_embeddings,
            tens_values,
        ) = self._soft_digit_representation(
            tens_cell_logits
        )

        valid_cells_float = (
            valid_cells.unsqueeze(-1).to(
                ones_embeddings.dtype
            )
        )

        ones_embeddings *= valid_cells_float
        tens_embeddings *= valid_cells_float
        ones_values *= valid_cells_float
        tens_values *= valid_cells_float

        diagonal_embeddings = torch.zeros(
            batch_size,
            self.product_seq_len,
            self.d_model,
            device=a.device,
            dtype=ones_embeddings.dtype,
        )

        diagonal_values = torch.zeros(
            batch_size,
            self.product_seq_len,
            1,
            device=a.device,
            dtype=ones_values.dtype,
        )

        diagonal_counts = torch.zeros(
            batch_size,
            self.product_seq_len,
            1,
            device=a.device,
            dtype=ones_values.dtype,
        )

        a_positions = torch.arange(
            seq_len,
            device=a.device,
        ).view(seq_len, 1)

        b_positions = torch.arange(
            seq_len,
            device=a.device,
        ).view(1, seq_len)

        ones_positions = (
            a_positions + b_positions
        )

        tens_positions = (
            ones_positions + 1
        )

        ones_embedding_indices = (
            ones_positions.reshape(
                1,
                seq_len * seq_len,
                1,
            ).expand(
                batch_size,
                -1,
                self.d_model,
            )
        )

        tens_embedding_indices = (
            tens_positions.reshape(
                1,
                seq_len * seq_len,
                1,
            ).expand(
                batch_size,
                -1,
                self.d_model,
            )
        )

        ones_scalar_indices = (
            ones_positions.reshape(
                1,
                seq_len * seq_len,
                1,
            ).expand(
                batch_size,
                -1,
                1,
            )
        )

        tens_scalar_indices = (
            tens_positions.reshape(
                1,
                seq_len * seq_len,
                1,
            ).expand(
                batch_size,
                -1,
                1,
            )
        )

        diagonal_embeddings.scatter_add_(
            dim=1,
            index=ones_embedding_indices,
            src=ones_embeddings.reshape(
                batch_size,
                seq_len * seq_len,
                self.d_model,
            ),
        )

        diagonal_embeddings.scatter_add_(
            dim=1,
            index=tens_embedding_indices,
            src=tens_embeddings.reshape(
                batch_size,
                seq_len * seq_len,
                self.d_model,
            ),
        )

        diagonal_values.scatter_add_(
            dim=1,
            index=ones_scalar_indices,
            src=ones_values.reshape(
                batch_size,
                seq_len * seq_len,
                1,
            ),
        )

        diagonal_values.scatter_add_(
            dim=1,
            index=tens_scalar_indices,
            src=tens_values.reshape(
                batch_size,
                seq_len * seq_len,
                1,
            ),
        )

        valid_counts = valid_cells_float.reshape(
            batch_size,
            seq_len * seq_len,
            1,
        )

        diagonal_counts.scatter_add_(
            dim=1,
            index=ones_scalar_indices,
            src=valid_counts,
        )

        diagonal_counts.scatter_add_(
            dim=1,
            index=tens_scalar_indices,
            src=valid_counts,
        )

        diagonal_features = torch.cat(
            [
                diagonal_embeddings,
                diagonal_values,
                diagonal_counts,
            ],
            dim=-1,
        )

        product_positions = torch.arange(
            self.product_seq_len,
            device=a.device,
        ).unsqueeze(0)

        diagonal_state = (
            self.diagonal_input_projection(
                diagonal_features
            )
            + self.product_position_embedding(
                product_positions
            )
        )

        a_lengths = (
            a != self.pad_id
        ).sum(dim=1)

        b_lengths = (
            b != self.pad_id
        ).sum(dim=1)

        product_lengths = torch.clamp(
            a_lengths + b_lengths + 1,
            min=2,
            max=self.product_seq_len,
        )

        product_padding_mask = (
            product_positions
            >= product_lengths.unsqueeze(1)
        )

        return (
            diagonal_state,
            product_padding_mask,
        )

    def _resolve_carries(
        self,
        diagonal_state: torch.Tensor,
        product_padding_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size = diagonal_state.shape[0]

        carry_state = (
            self.initial_carry_state.expand(
                batch_size,
                -1,
            )
        )

        product_states = []

        for position in range(
            self.product_seq_len
        ):
            previous_state = carry_state

            carry_state = self.carry_cell(
                diagonal_state[:, position],
                carry_state,
            )

            is_padding = product_padding_mask[
                :,
                position,
            ].unsqueeze(1)

            carry_state = torch.where(
                is_padding,
                previous_state,
                carry_state,
            )

            product_states.append(
                carry_state
            )

        product_state = torch.stack(
            product_states,
            dim=1,
        )

        product_state = self.carry_output_norm(
            product_state
        )

        carry_logits = self.carry_head(
            product_state
        )

        return (
            product_state,
            carry_logits,
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            ones_features,
            tens_features,
            valid_cells,
        ) = self._create_cell_features(
            a,
            b,
        )

        ones_cell_logits = self.ones_cell_head(
            ones_features
        )

        tens_cell_logits = self.tens_cell_head(
            tens_features
        )

        (
            diagonal_state,
            product_padding_mask,
        ) = self._create_diagonal_state(
            a,
            b,
            ones_cell_logits,
            tens_cell_logits,
            valid_cells,
        )

        (
            product_state,
            carry_logits,
        ) = self._resolve_carries(
            diagonal_state,
            product_padding_mask,
        )

        product_logits = self.product_head(
            product_state
        )

        return (
            product_logits,
            ones_cell_logits,
            tens_cell_logits,
            carry_logits,
        )

    @torch.no_grad()
    def predict(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()

        product_logits, _, _, _ = self(
            a,
            b,
        )

        predictions = product_logits.argmax(
            dim=-1
        )

        _, seq_len = predictions.shape

        positions = torch.arange(
            seq_len,
            device=predictions.device,
        ).unsqueeze(0)

        eos_mask = predictions == self.eos_id
        has_eos = eos_mask.any(dim=1)

        eos_positions = (
            eos_mask.int().argmax(dim=1)
        )

        after_eos = (
            positions
            > eos_positions.unsqueeze(1)
        )

        after_eos &= has_eos.unsqueeze(1)

        return predictions.masked_fill(
            after_eos,
            self.pad_id,
        )