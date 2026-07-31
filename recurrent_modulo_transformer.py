import torch
import torch.nn as nn


class RecurrentModuloModel(nn.Module):
    def __init__(
        self,
        max_modulus: int,
        max_product_seq_len: int,
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.max_modulus = max_modulus
        self.max_product_seq_len = max_product_seq_len
        self.remainder_class_count = max_modulus
        self.d_model = d_model

        self.digit_embedding = nn.Embedding(
            10,
            d_model,
        )

        self.modulus_embedding = nn.Embedding(
            max_modulus + 1,
            d_model,
        )

        self.remainder_embedding = nn.Embedding(
            self.remainder_class_count,
            d_model,
        )

        self.step_embedding = nn.Embedding(
            max_product_seq_len,
            d_model,
        )

        self.input_projection = nn.Sequential(
            nn.Linear(
                3 * d_model,
                d_model,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.recurrent_cell = nn.GRUCell(
            input_size=d_model,
            hidden_size=d_model,
        )

        self.output_norm = nn.LayerNorm(
            d_model
        )

        self.remainder_head = nn.Linear(
            d_model,
            self.remainder_class_count,
        )

    def _forward_digit_states(
        self,
        product_digit_states: torch.Tensor,
        product_padding_mask: torch.Tensor,
        modulus_values: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size = (
            product_digit_states.shape[0]
        )

        remainder_probabilities = torch.zeros(
            batch_size,
            self.remainder_class_count,
            dtype=self.remainder_embedding.weight.dtype,
            device=product_digit_states.device,
        )

        remainder_probabilities[
            :,
            0,
        ] = 1.0

        modulus_state = self.modulus_embedding(
            modulus_values
        )

        intermediate_logits = []

        for step in range(
            self.max_product_seq_len
        ):
            previous_probabilities = (
                remainder_probabilities
            )

            previous_remainder_state = (
                previous_probabilities
                @ self.remainder_embedding.weight
            )

            current_digit_state = (
                product_digit_states[
                    :,
                    step,
                ]
            )

            step_ids = torch.full(
                (
                    batch_size,
                ),
                step,
                dtype=torch.long,
                device=product_digit_states.device,
            )

            step_state = self.step_embedding(
                step_ids
            )

            recurrent_input = torch.cat(
                [
                    current_digit_state,
                    modulus_state,
                    step_state,
                ],
                dim=-1,
            )

            recurrent_input = (
                self.input_projection(
                    recurrent_input
                )
            )

            updated_hidden_state = (
                self.recurrent_cell(
                    recurrent_input,
                    previous_remainder_state,
                )
            )

            step_logits = (
                self.remainder_head(
                    self.output_norm(
                        updated_hidden_state
                    )
                )
            )

            updated_probabilities = torch.softmax(
                step_logits,
                dim=-1,
            )

            active_step = (
                ~product_padding_mask[
                    :,
                    step,
                ]
            ).unsqueeze(
                dim=-1
            )

            remainder_probabilities = torch.where(
                active_step,
                updated_probabilities,
                previous_probabilities,
            )

            intermediate_logits.append(
                step_logits
            )

        intermediate_logits = torch.stack(
            intermediate_logits,
            dim=1,
        )

        product_lengths = (
            ~product_padding_mask
        ).sum(
            dim=1
        )

        final_step_indices = torch.clamp(
            product_lengths - 1,
            min=0,
        )

        batch_indices = torch.arange(
            batch_size,
            device=product_digit_states.device,
        )

        final_logits = intermediate_logits[
            batch_indices,
            final_step_indices,
        ]

        return (
            final_logits,
            intermediate_logits,
        )

    def forward(
        self,
        product_digits: torch.Tensor,
        product_padding_mask: torch.Tensor,
        modulus_values: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        product_digit_states = (
            self.digit_embedding(
                product_digits.clamp(
                    min=0,
                    max=9,
                )
            )
        )

        return self._forward_digit_states(
            product_digit_states,
            product_padding_mask,
            modulus_values,
        )

    def forward_from_probabilities(
        self,
        product_digit_probabilities: torch.Tensor,
        product_padding_mask: torch.Tensor,
        modulus_values: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        product_digit_states = (
            product_digit_probabilities
            @ self.digit_embedding.weight
        )

        return self._forward_digit_states(
            product_digit_states,
            product_padding_mask,
            modulus_values,
        )

    @torch.no_grad()
    def predict(
        self,
        product_digits: torch.Tensor,
        product_padding_mask: torch.Tensor,
        modulus_values: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()

        final_logits, _ = self(
            product_digits,
            product_padding_mask,
            modulus_values,
        )

        return final_logits.argmax(
            dim=-1
        )