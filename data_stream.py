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
        seed: int | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be greater than 0.")

        if initial_max <= 0:
            raise ValueError("initial_max must be greater than 0.")

        if growth_rate < 0:
            raise ValueError("growth_rate must be non-negative.")

        if beta_alpha <= 0:
            raise ValueError("beta_alpha must be greater than 0.")

        if beta_beta <= 0:
            raise ValueError("beta_beta must be greater than 0.")

        if not 0.0 <= rehearsal_fraction <= 1.0:
            raise ValueError(
                "rehearsal_fraction must be between 0 and 1."
            )

        if rehearsal_max < 0:
            raise ValueError("rehearsal_max must be non-negative.")

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

        self.pad_id = tokenizer.char_to_int["<pad>"]
        self.eos_id = tokenizer.char_to_int["<eos>"]

        self.seed = seed
        self.rng = random.Random(seed)
        self.step = 0

    def next_batch(self) -> torch.Tensor:
        rehearsal_count = round(
            self.batch_size * self.rehearsal_fraction
        )
        curriculum_count = self.batch_size - rehearsal_count

        samples: list[list[list[int]]] = []

        for _ in range(curriculum_count):
            a = self._sample_curriculum_integer()
            b = self._sample_curriculum_integer()

            samples.append(
                self._encode_sample(a, b)
            )

        for _ in range(rehearsal_count):
            a = self.rng.randint(
                0,
                self.rehearsal_max,
            )
            b = self.rng.randint(
                0,
                self.rehearsal_max,
            )

            samples.append(
                self._encode_sample(a, b)
            )

        self.rng.shuffle(samples)

        batch = torch.tensor(
            samples,
            dtype=torch.long,
        )

        self.step += 1
        self.current_max *= 1.0 + self.growth_rate

        return batch

    def create_validation_chunk_loader(
        self,
        num_steps: int,
        lookahead_steps: int = 1_000,
        batch_size: int | None = None,
        seed: int = 10_000,
    ) -> DataLoader:
        if num_steps <= 0:
            raise ValueError("num_steps must be greater than 0.")
    
        if lookahead_steps < 0:
            raise ValueError("lookahead_steps must be non-negative.")
    
        validation_batch_size = (
            self.batch_size
            if batch_size is None
            else batch_size
        )
    
        if validation_batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
    
        validation_rng = random.Random(seed)
    
        validation_max = self.current_max * (
            (1.0 + self.growth_rate) ** lookahead_steps
        )
    
        rehearsal_count = round(
            validation_batch_size * self.rehearsal_fraction
        )
    
        curriculum_count = (
            validation_batch_size - rehearsal_count
        )
    
        total_samples = num_steps * validation_batch_size
    
        dataset = torch.empty(
            (
                total_samples,
                3,
                self.max_seq_len,
            ),
            dtype=torch.long,
        )
    
        for validation_step in range(num_steps):
            samples: list[list[list[int]]] = []
    
            for _ in range(curriculum_count):
                a = int(
                    validation_rng.betavariate(
                        self.beta_alpha,
                        self.beta_beta,
                    )
                    * validation_max
                )
    
                b = int(
                    validation_rng.betavariate(
                        self.beta_alpha,
                        self.beta_beta,
                    )
                    * validation_max
                )
    
                samples.append(
                    self._encode_sample(a, b)
                )
    
            for _ in range(rehearsal_count):
                a = validation_rng.randint(
                    0,
                    self.rehearsal_max,
                )
    
                b = validation_rng.randint(
                    0,
                    self.rehearsal_max,
                )
    
                samples.append(
                    self._encode_sample(a, b)
                )
    
            validation_rng.shuffle(samples)
    
            start_index = (
                validation_step * validation_batch_size
            )
    
            end_index = (
                start_index + validation_batch_size
            )
    
            dataset[start_index:end_index] = torch.tensor(
                samples,
                dtype=torch.long,
            )
    
            validation_max *= 1.0 + self.growth_rate
    
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

        if max_value < 0:
            raise ValueError(
                "max_value must be non-negative."
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
        samples: list[list[list[int]]] = []

        for _ in range(num_samples):
            a = evaluation_rng.randint(
                0,
                max_value,
            )
            b = evaluation_rng.randint(
                0,
                max_value,
            )

            samples.append(
                self._encode_sample(a, b)
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
            raise ValueError("num_steps must be greater than 0.")
    
        max_token_id = max(self.tokenizer.char_to_int.values())
    
        if max_token_id <= 255:
            storage_dtype = torch.uint8
        elif max_token_id <= 32767:
            storage_dtype = torch.int16
        else:
            storage_dtype = torch.int32
    
        total_samples = num_steps * self.batch_size
    
        dataset = torch.empty(
            (
                total_samples,
                3,
                self.max_seq_len,
            ),
            dtype=storage_dtype,
        )
    
        for chunk_step in tqdm(
            range(num_steps),
            desc=f"Generating {num_steps:,} training batches",
        ):
            start_index = chunk_step * self.batch_size
            end_index = start_index + self.batch_size
    
            dataset[start_index:end_index] = (
                self.next_batch().to(storage_dtype)
            )
    
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=pin_memory,
        )

    def _sample_curriculum_integer(self) -> int:
        beta_sample = self.rng.betavariate(
            self.beta_alpha,
            self.beta_beta,
        )

        return int(
            beta_sample * self.current_max
        )

    def _encode_sample(
        self,
        a: int,
        b: int,
    ) -> list[list[int]]:
        target = a * b

        a_tokens = self.tokenizer.encode(str(a))
        b_tokens = self.tokenizer.encode(str(b))
        target_tokens = self.tokenizer.encode(
            str(target)
        )

        a_tokens.reverse()
        b_tokens.reverse()
        target_tokens.reverse()

        target_tokens.append(self.eos_id)

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
            name="target",
            value=target,
            token_ids=target_tokens,
        )

        return [
            self._pad(a_tokens),
            self._pad(b_tokens),
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
            self.max_seq_len - len(token_ids)
        )

        return (
            token_ids
            + [self.pad_id] * padding_length
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

        missing_keys = required_keys - state.keys()

        if missing_keys:
            raise KeyError(
                "Missing data-stream state keys: "
                f"{sorted(missing_keys)}"
            )

        self.step = int(state["step"])
        self.current_max = float(
            state["current_max"]
        )
        self.rng.setstate(
            state["random_state"]
        )

    def reset(self) -> None:
        self.step = 0
        self.current_max = self.initial_max
        self.rng = random.Random(self.seed)