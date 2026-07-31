import torch
import torch.nn as nn

from lattice_multiplication_transformer import (
    LatticeMultiplicationTransformer,
)

from recurrent_modulo_transformer import (
    RecurrentModuloModel,
)


class CombinedModularArithmeticModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        max_modulus: int,
        pad_id: int,
        eos_id: int,
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.max_product_seq_len = (
            2 * max_seq_len
        )

        self.max_modulus = max_modulus
        self.pad_id = pad_id
        self.eos_id = eos_id

        self.multiplication_model = (
            LatticeMultiplicationTransformer(
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
                pad_id=pad_id,
                eos_id=eos_id,
                d_model=d_model,
                dropout=dropout,
            )
        )

        self.modulo_model = (
            RecurrentModuloModel(
                max_modulus=max_modulus,
                max_product_seq_len=(
                    self.max_product_seq_len
                ),
                d_model=d_model,
                dropout=dropout,
            )
        )

    def _reverse_and_compact_probabilities(
        self,
        digit_probabilities: torch.Tensor,
        product_digit_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size = (
            digit_probabilities.shape[0]
        )

        product_lengths = (
            product_digit_mask.sum(
                dim=1
            )
        )

        output_positions = torch.arange(
            self.max_product_seq_len,
            device=digit_probabilities.device,
        ).unsqueeze(0)

        source_positions = (
            product_lengths.unsqueeze(1)
            - 1
            - output_positions
        )

        source_positions = source_positions.clamp(
            min=0,
            max=digit_probabilities.shape[1] - 1,
        )

        gather_indices = (
            source_positions.unsqueeze(-1).expand(
                batch_size,
                self.max_product_seq_len,
                10,
            )
        )

        reversed_probabilities = torch.gather(
            digit_probabilities,
            dim=1,
            index=gather_indices,
        )

        product_padding_mask = (
            output_positions
            >= product_lengths.unsqueeze(1)
        )

        reversed_probabilities = (
            reversed_probabilities.masked_fill(
                product_padding_mask.unsqueeze(-1),
                0.0,
            )
        )

        return (
            reversed_probabilities,
            product_padding_mask,
        )

    def _create_soft_modulo_inputs(
        self,
        product_logits: torch.Tensor,
        product_digit_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        digit_logits = product_logits[
            :,
            :,
            :10,
        ]

        digit_probabilities = torch.softmax(
            digit_logits.float(),
            dim=-1,
        ).to(
            product_logits.dtype
        )

        return (
            self._reverse_and_compact_probabilities(
                digit_probabilities,
                product_digit_mask,
            )
        )

    def _create_discrete_modulo_inputs(
        self,
        product_predictions: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size, seq_len = (
            product_predictions.shape
        )

        positions = torch.arange(
            seq_len,
            device=product_predictions.device,
        ).unsqueeze(0)

        eos_mask = (
            product_predictions
            == self.eos_id
        )

        has_eos = eos_mask.any(
            dim=1
        )

        eos_positions = eos_mask.int().argmax(
            dim=1
        )

        before_eos = (
            positions
            < eos_positions.unsqueeze(1)
        )

        before_eos = torch.where(
            has_eos.unsqueeze(1),
            before_eos,
            torch.ones_like(
                before_eos
            ),
        )

        product_digit_mask = (
            before_eos
            & (product_predictions >= 0)
            & (product_predictions <= 9)
        )

        clamped_predictions = (
            product_predictions.clamp(
                min=0,
                max=9,
            )
        )

        digit_probabilities = torch.nn.functional.one_hot(
            clamped_predictions,
            num_classes=10,
        ).to(
            self.modulo_model.digit_embedding.weight.dtype
        )

        (
            reversed_probabilities,
            product_padding_mask,
        ) = self._reverse_and_compact_probabilities(
            digit_probabilities,
            product_digit_mask,
        )

        product_digits = (
            reversed_probabilities.argmax(
                dim=-1
            )
        )

        return (
            product_digits,
            product_padding_mask,
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        modulus_values: torch.Tensor,
        product_digit_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            product_logits,
            ones_cell_logits,
            tens_cell_logits,
            carry_logits,
        ) = self.multiplication_model(
            a,
            b,
        )

        (
            product_digit_probabilities,
            product_padding_mask,
        ) = self._create_soft_modulo_inputs(
            product_logits,
            product_digit_mask,
        )

        (
            final_remainder_logits,
            intermediate_remainder_logits,
        ) = (
            self.modulo_model.forward_from_probabilities(
                product_digit_probabilities,
                product_padding_mask,
                modulus_values,
            )
        )

        return (
            final_remainder_logits,
            intermediate_remainder_logits,
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
        modulus_values: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        self.eval()

        product_predictions = (
            self.multiplication_model.predict(
                a,
                b,
            )
        )

        (
            product_digits,
            product_padding_mask,
        ) = self._create_discrete_modulo_inputs(
            product_predictions
        )

        remainder_predictions = (
            self.modulo_model.predict(
                product_digits,
                product_padding_mask,
                modulus_values,
            )
        )

        return (
            remainder_predictions,
            product_predictions,
        )