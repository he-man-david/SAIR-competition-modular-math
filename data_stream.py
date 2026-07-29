import math
import random

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tokenizer import DigitTokenizer


class MultiplicationDataStream:
    def __init__(
        self,
        tokenizer: DigitTokenizer,
        batch_size: int,
        max_seq_len: int,
        initial_max: float = 10.0,
        growth_rate: float = 0.0001,
        beta_alpha: float = 2.0,
        beta_beta: float = 5.0,
        rehearsal_fraction: float = 0.2,
        rehearsal_max: int = 100,
        max_operand_multiple: int = 3,
        seed: int | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        if max_seq_len <= 1:
            raise ValueError(
                "max_seq_len must be greater than 1."
            )

        if initial_max <= 0:
            raise ValueError(
                "initial_max must be greater than 0."
            )

        if growth_rate < 0:
            raise ValueError(
                "growth_rate must be non-negative."
            )

        if beta_alpha <= 0:
            raise ValueError(
                "beta_alpha must be greater than 0."
            )

        if beta_beta <= 0:
            raise ValueError(
                "beta_beta must be greater than 0."
            )

        if not 0.0 <= rehearsal_fraction <= 1.0:
            raise ValueError(
                "rehearsal_fraction must be between 0 and 1."
            )

        if rehearsal_max <= 0:
            raise ValueError(
                "rehearsal_max must be greater than 0."
            )

        if max_operand_multiple < 0:
            raise ValueError(
                "max_operand_multiple must be non-negative."
            )

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        self.initial_max = float(initial_max)
        self.current_max = float(initial_max)
        self.growth_rate = float(growth_rate)

        self.beta_alpha = float(beta_alpha)
        self.beta_beta = float(beta_beta)

        self.rehearsal_fraction = float(rehearsal_fraction)
        self.rehearsal_max = rehearsal_max
        self.max_operand_multiple = max_operand_multiple

        self.pad_id = tokenizer.char_to_int["<pad>"]
        self.eos_id = tokenizer.char_to_int["<eos>"]

        self.max_operand_value = (
            10 ** self.max_seq_len
        ) - 1

        self.max_modulus_value = (
            10 ** (self.max_seq_len - 1)
        )

        self.seed = seed
        self.rng = random.Random(seed)
        self.step = 0

    def next_batch(self) -> torch.Tensor:
        rehearsal_count = round(
            self.batch_size * self.rehearsal_fraction
        )

        curriculum_count = (
            self.batch_size - rehearsal_count
        )

        samples: list[list[list[int]]] = []

        curriculum_max = self._get_generation_max(
            self.current_max
        )

        for _ in range(curriculum_count):
            a, b, p = self._sample_uniform_target_operands(
                rng=self.rng,
                max_value=curriculum_max,
            )

            samples.append(
                self._encode_sample(
                    a,
                    b,
                    p,
                )
            )

        rehearsal_max = self._get_generation_max(
            self.rehearsal_max
        )

        for _ in range(rehearsal_count):
            a, b, p = self._sample_uniform_target_operands(
                rng=self.rng,
                max_value=rehearsal_max,
            )

            samples.append(
                self._encode_sample(
                    a,
                    b,
                    p,
                )
            )

        self.rng.shuffle(samples)

        batch = torch.tensor(
            samples,
            dtype=torch.long,
        )

        self.step += 1
        self.current_max *= (
            1.0 + self.growth_rate
        )

        return batch

    def create_validation_chunk_loader(
        self,
        num_steps: int,
        lookahead_steps: int = 1_000,
        batch_size: int | None = None,
        seed: int = 10_000,
    ) -> DataLoader:
        if num_steps <= 0:
            raise ValueError(
                "num_steps must be greater than 0."
            )

        if lookahead_steps < 0:
            raise ValueError(
                "lookahead_steps must be non-negative."
            )

        validation_batch_size = (
            self.batch_size
            if batch_size is None
            else batch_size
        )

        if validation_batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        validation_rng = random.Random(seed)

        validation_max = self.current_max * (
            (1.0 + self.growth_rate)
            ** lookahead_steps
        )

        rehearsal_count = round(
            validation_batch_size
            * self.rehearsal_fraction
        )

        curriculum_count = (
            validation_batch_size
            - rehearsal_count
        )

        total_samples = (
            num_steps
            * validation_batch_size
        )

        dataset = torch.empty(
            (
                total_samples,
                4,
                self.max_seq_len,
            ),
            dtype=torch.long,
        )

        rehearsal_max = self._get_generation_max(
            self.rehearsal_max
        )

        for validation_step in range(num_steps):
            samples: list[list[list[int]]] = []

            curriculum_max = self._get_generation_max(
                validation_max
            )

            for _ in range(curriculum_count):
                a, b, p = (
                    self._sample_uniform_target_operands(
                        rng=validation_rng,
                        max_value=curriculum_max,
                    )
                )

                samples.append(
                    self._encode_sample(
                        a,
                        b,
                        p,
                    )
                )

            for _ in range(rehearsal_count):
                a, b, p = (
                    self._sample_uniform_target_operands(
                        rng=validation_rng,
                        max_value=rehearsal_max,
                    )
                )

                samples.append(
                    self._encode_sample(
                        a,
                        b,
                        p,
                    )
                )

            validation_rng.shuffle(samples)

            start_index = (
                validation_step
                * validation_batch_size
            )

            end_index = (
                start_index
                + validation_batch_size
            )

            dataset[start_index:end_index] = (
                torch.tensor(
                    samples,
                    dtype=torch.long,
                )
            )

            validation_max *= (
                1.0 + self.growth_rate
            )

        return DataLoader(
            dataset,
            batch_size=validation_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def create_fixed_loader(
        self,
        num_samples: int,
        max_value: int,
        batch_size: int | None = None,
        seed: int = 42,
    ) -> DataLoader:
        if num_samples <= 0:
            raise ValueError(
                "num_samples must be greater than 0."
            )

        if max_value <= 0:
            raise ValueError(
                "max_value must be greater than 0."
            )

        loader_batch_size = (
            self.batch_size
            if batch_size is None
            else batch_size
        )

        if loader_batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        evaluation_rng = random.Random(seed)

        fixed_max = self._get_generation_max(
            max_value
        )

        samples: list[list[list[int]]] = []

        for _ in range(num_samples):
            a, b, p = (
                self._sample_uniform_target_operands(
                    rng=evaluation_rng,
                    max_value=fixed_max,
                )
            )

            samples.append(
                self._encode_sample(
                    a,
                    b,
                    p,
                )
            )

        dataset = torch.tensor(
            samples,
            dtype=torch.long,
        )

        return DataLoader(
            dataset,
            batch_size=loader_batch_size,
            shuffle=False,
            drop_last=False,
        )

    def create_training_chunk_loader(
        self,
        num_steps: int,
        pin_memory: bool = True,
    ) -> DataLoader:
        if num_steps <= 0:
            raise ValueError(
                "num_steps must be greater than 0."
            )

        max_token_id = max(
            self.tokenizer.char_to_int.values()
        )

        if max_token_id <= 255:
            storage_dtype = torch.uint8
        elif max_token_id <= 32767:
            storage_dtype = torch.int16
        else:
            storage_dtype = torch.int32

        total_samples = (
            num_steps
            * self.batch_size
        )

        dataset = torch.empty(
            (
                total_samples,
                4,
                self.max_seq_len,
            ),
            dtype=storage_dtype,
        )

        original_step = self.step
        original_current_max = self.current_max

        for chunk_step in tqdm(
            range(num_steps),
            desc=(
                f"Generating {num_steps:,} "
                f"training batches"
            ),
        ):
            start_index = (
                chunk_step
                * self.batch_size
            )

            end_index = (
                start_index
                + self.batch_size
            )

            dataset[start_index:end_index] = (
                self.next_batch().to(
                    storage_dtype
                )
            )

        self.step = original_step
        self.current_max = original_current_max

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=pin_memory,
        )

    def _get_generation_max(
        self,
        max_value: int | float,
    ) -> int:
        return min(
            max(
                2,
                int(max_value),
            ),
            self.max_modulus_value,
        )

    def _sample_uniform_target_operands(
        self,
        rng: random.Random,
        max_value: int,
    ) -> tuple[int, int, int]:
        target = rng.randint(
            0,
            max_value - 1,
        )

        p = rng.randint(
            target + 1,
            max_value,
        )

        while True:
            base_a = rng.randint(
                1,
                p - 1,
            )

            if math.gcd(base_a, p) == 1:
                break

        inverse_a = pow(
            base_a,
            -1,
            p,
        )

        base_b = (
            target
            * inverse_a
        ) % p

        max_k1 = min(
            self.max_operand_multiple,
            (
                self.max_operand_value
                - base_a
            ) // p,
        )

        max_k2 = min(
            self.max_operand_multiple,
            (
                self.max_operand_value
                - base_b
            ) // p,
        )

        k1 = rng.randint(
            0,
            max_k1,
        )

        k2 = rng.randint(
            0,
            max_k2,
        )

        a = (
            base_a
            + k1 * p
        )

        b = (
            base_b
            + k2 * p
        )

        if rng.random() < 0.5:
            a, b = b, a

        return a, b, p

    def _encode_sample(
        self,
        a: int,
        b: int,
        p: int,
    ) -> list[list[int]]:
        target = (
            a * b
        ) % p

        a_tokens = self.tokenizer.encode(
            str(a)
        )

        b_tokens = self.tokenizer.encode(
            str(b)
        )

        p_tokens = self.tokenizer.encode(
            str(p)
        )

        target_tokens = self.tokenizer.encode(
            str(target)
        )

        a_tokens.reverse()
        b_tokens.reverse()
        p_tokens.reverse()
        target_tokens.reverse()

        target_tokens.append(
            self.eos_id
        )

        self._check_sequence_length(
            name="a",
            value=a,
            token_ids=a_tokens,
        )

        self._check_sequence_length(
            name="b",
            value=b,
            token_ids=b_tokens,
        )

        self._check_sequence_length(
            name="p",
            value=p,
            token_ids=p_tokens,
        )

        self._check_sequence_length(
            name="target",
            value=target,
            token_ids=target_tokens,
        )

        return [
            self._pad(a_tokens),
            self._pad(b_tokens),
            self._pad(p_tokens),
            self._pad(target_tokens),
        ]

    def _check_sequence_length(
        self,
        name: str,
        value: int,
        token_ids: list[int],
    ) -> None:
        if len(token_ids) > self.max_seq_len:
            raise ValueError(
                f"{name}={value} requires "
                f"{len(token_ids)} tokens, but "
                f"max_seq_len is {self.max_seq_len}."
            )

    def _pad(
        self,
        token_ids: list[int],
    ) -> list[int]:
        padding_length = (
            self.max_seq_len
            - len(token_ids)
        )

        return (
            token_ids
            + [self.pad_id]
            * padding_length
        )

    def state_dict(self) -> dict:
        return {
            "step": self.step,
            "current_max": self.current_max,
            "random_state": self.rng.getstate(),
        }

    def load_state_dict(
        self,
        state: dict,
    ) -> None:
        required_keys = {
            "step",
            "current_max",
            "random_state",
        }

        missing_keys = (
            required_keys
            - state.keys()
        )

        if missing_keys:
            raise KeyError(
                "Missing data-stream state keys: "
                f"{sorted(missing_keys)}"
            )

        self.step = int(
            state["step"]
        )

        self.current_max = float(
            state["current_max"]
        )

        self.rng.setstate(
            state["random_state"]
        )

    def reset(self) -> None:
        self.step = 0
        self.current_max = self.initial_max
        self.rng = random.Random(
            self.seed
        )