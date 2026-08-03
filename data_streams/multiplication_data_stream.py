import math
import random

import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer import DigitTokenizer


class MultiplicationTensorDataset(Dataset):
    def __init__(
        self,
        a_token_ids: torch.Tensor,
        b_token_ids: torch.Tensor,
        product_token_ids: torch.Tensor,
    ):
        if not (
            a_token_ids.shape[0]
            == b_token_ids.shape[0]
            == product_token_ids.shape[0]
        ):
            raise ValueError(
                "All tensors must contain the same number of samples."
            )

        self.a_token_ids = a_token_ids
        self.b_token_ids = b_token_ids
        self.product_token_ids = product_token_ids

    def __len__(self) -> int:
        return self.a_token_ids.shape[0]

    def __getitem__(
        self,
        sample_index: int,
    ) -> dict[str, torch.Tensor]:
        return {
            "a": self.a_token_ids[sample_index],
            "b": self.b_token_ids[sample_index],
            "product": self.product_token_ids[sample_index],
        }


class MultiplicationDataStream:
    STATE_VERSION = 1

    def __init__(
        self,
        tokenizer: DigitTokenizer,
        batch_size: int,
        max_seq_len: int,
        initial_max_product_value: int = 10,
        product_growth_rate: float = 0.0001,
        rehearsal_fraction: float = 0.2,
        rehearsal_max_product_value: int = 10,
        zero_sample_probability: float = 0.02,
        seed: int | None = None,
    ):
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        if max_seq_len <= 0:
            raise ValueError(
                "max_seq_len must be greater than 0."
            )

        if initial_max_product_value <= 0:
            raise ValueError(
                "initial_max_product_value must be greater than 0."
            )

        if product_growth_rate < 0:
            raise ValueError(
                "product_growth_rate must be non-negative."
            )

        if not 0.0 <= rehearsal_fraction <= 1.0:
            raise ValueError(
                "rehearsal_fraction must be between 0 and 1."
            )

        if rehearsal_max_product_value <= 0:
            raise ValueError(
                "rehearsal_max_product_value must be greater than 0."
            )

        if not 0.0 <= zero_sample_probability <= 1.0:
            raise ValueError(
                "zero_sample_probability must be between 0 and 1."
            )

        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.operand_seq_len = max_seq_len
        self.product_seq_len = 2 * max_seq_len + 1

        self.maximum_operand_value = (
            10**self.operand_seq_len
        ) - 1

        self.maximum_representable_product_value = (
            self.maximum_operand_value**2
        )

        if (
            initial_max_product_value
            > self.maximum_representable_product_value
        ):
            raise ValueError(
                "initial_max_product_value exceeds the largest product "
                f"representable by max_seq_len={max_seq_len}."
            )

        if (
            rehearsal_max_product_value
            > self.maximum_representable_product_value
        ):
            raise ValueError(
                "rehearsal_max_product_value exceeds the largest product "
                f"representable by max_seq_len={max_seq_len}."
            )

        self.initial_max_product_value = int(
            initial_max_product_value
        )

        self.current_max_product_value = float(
            initial_max_product_value
        )

        self.product_growth_rate = float(
            product_growth_rate
        )

        self.rehearsal_fraction = float(
            rehearsal_fraction
        )

        self.rehearsal_max_product_value = int(
            rehearsal_max_product_value
        )

        self.zero_sample_probability = float(
            zero_sample_probability
        )

        self.pad_id = tokenizer.char_to_int["<pad>"]
        self.eos_id = tokenizer.char_to_int["<eos>"]

        self.seed = seed
        self.random_number_generator = random.Random(
            seed
        )

        self.step = 0

    def next_batch(
        self,
    ) -> dict[str, torch.Tensor]:
        curriculum_max_product_value = (
            self._effective_max_product_value(
                self.current_max_product_value
            )
        )

        batch = self._generate_mixed_batch(
            random_number_generator=(
                self.random_number_generator
            ),
            batch_size=self.batch_size,
            curriculum_max_product_value=(
                curriculum_max_product_value
            ),
        )

        self.step += 1
        self._advance_curriculum()

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

        validation_random_number_generator = (
            random.Random(seed)
        )

        validation_max_product_value = (
            self._projected_max_product_value(
                lookahead_steps
            )
        )

        validation_batches = []

        for _ in range(num_steps):
            validation_batches.append(
                self._generate_mixed_batch(
                    random_number_generator=(
                        validation_random_number_generator
                    ),
                    batch_size=validation_batch_size,
                    curriculum_max_product_value=(
                        self._effective_max_product_value(
                            validation_max_product_value
                        )
                    ),
                )
            )

            validation_max_product_value = (
                self._grow_max_product_value(
                    validation_max_product_value
                )
            )

        validation_tensor_batch = (
            self._concatenate_tensor_batches(
                validation_batches
            )
        )

        validation_dataset = (
            self._create_tensor_dataset(
                validation_tensor_batch
            )
        )

        return DataLoader(
            validation_dataset,
            batch_size=validation_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def create_fixed_loader(
        self,
        num_samples: int,
        max_product_value: int,
        batch_size: int | None = None,
        seed: int = 42,
    ) -> DataLoader:
        if num_samples <= 0:
            raise ValueError(
                "num_samples must be greater than 0."
            )

        if max_product_value <= 0:
            raise ValueError(
                "max_product_value must be greater than 0."
            )

        if (
            max_product_value
            > self.maximum_representable_product_value
        ):
            raise ValueError(
                "max_product_value exceeds the largest product "
                f"representable by max_seq_len={self.operand_seq_len}."
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

        evaluation_random_number_generator = (
            random.Random(seed)
        )

        evaluation_tensor_batch = (
            self._generate_batch(
                random_number_generator=(
                    evaluation_random_number_generator
                ),
                sample_count=num_samples,
                max_product_value=max_product_value,
            )
        )

        evaluation_dataset = (
            self._create_tensor_dataset(
                evaluation_tensor_batch
            )
        )

        return DataLoader(
            evaluation_dataset,
            batch_size=loader_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def state_dict(
        self,
    ) -> dict:
        return {
            "state_version": self.STATE_VERSION,
            "configuration": (
                self._configuration_state()
            ),
            "step": self.step,
            "current_max_product_value": (
                self.current_max_product_value
            ),
            "random_state": (
                self.random_number_generator.getstate()
            ),
        }

    def load_state_dict(
        self,
        state: dict,
    ) -> None:
        required_keys = {
            "state_version",
            "configuration",
            "step",
            "current_max_product_value",
            "random_state",
        }

        missing_keys = required_keys - state.keys()

        if missing_keys:
            raise KeyError(
                "Missing multiplication data-stream state keys: "
                f"{sorted(missing_keys)}"
            )

        if state["state_version"] != self.STATE_VERSION:
            raise ValueError(
                "Unsupported multiplication data-stream state version: "
                f"{state['state_version']}."
            )

        expected_configuration = (
            self._configuration_state()
        )

        saved_configuration = state[
            "configuration"
        ]

        if saved_configuration != expected_configuration:
            mismatched_configuration = {
                configuration_name: {
                    "saved": saved_configuration.get(
                        configuration_name
                    ),
                    "current": expected_configuration.get(
                        configuration_name
                    ),
                }
                for configuration_name in sorted(
                    set(saved_configuration)
                    | set(expected_configuration)
                )
                if saved_configuration.get(
                    configuration_name
                )
                != expected_configuration.get(
                    configuration_name
                )
            }

            raise ValueError(
                "The saved multiplication data-stream configuration "
                "does not match the current stream: "
                f"{mismatched_configuration}"
            )

        current_max_product_value = float(
            state["current_max_product_value"]
        )

        if not (
            1.0
            <= current_max_product_value
            <= float(
                self.maximum_representable_product_value
            )
        ):
            raise ValueError(
                "Saved current_max_product_value is outside "
                "the representable range."
            )

        self.step = int(
            state["step"]
        )

        self.current_max_product_value = (
            current_max_product_value
        )

        self.random_number_generator.setstate(
            state["random_state"]
        )

    def reset(
        self,
    ) -> None:
        self.step = 0

        self.current_max_product_value = float(
            self.initial_max_product_value
        )

        self.random_number_generator = (
            random.Random(self.seed)
        )

    def _generate_mixed_batch(
        self,
        random_number_generator: random.Random,
        batch_size: int,
        curriculum_max_product_value: int,
    ) -> dict[str, torch.Tensor]:
        rehearsal_sample_count = round(
            batch_size
            * self.rehearsal_fraction
        )

        curriculum_sample_count = (
            batch_size
            - rehearsal_sample_count
        )

        encoded_samples = (
            self._generate_encoded_samples(
                random_number_generator=(
                    random_number_generator
                ),
                sample_count=(
                    curriculum_sample_count
                ),
                max_product_value=(
                    curriculum_max_product_value
                ),
            )
        )

        encoded_samples.extend(
            self._generate_encoded_samples(
                random_number_generator=(
                    random_number_generator
                ),
                sample_count=(
                    rehearsal_sample_count
                ),
                max_product_value=(
                    self.rehearsal_max_product_value
                ),
            )
        )

        random_number_generator.shuffle(
            encoded_samples
        )

        return (
            self._convert_encoded_samples_to_batch(
                encoded_samples
            )
        )

    def _generate_batch(
        self,
        random_number_generator: random.Random,
        sample_count: int,
        max_product_value: int,
    ) -> dict[str, torch.Tensor]:
        encoded_samples = (
            self._generate_encoded_samples(
                random_number_generator=(
                    random_number_generator
                ),
                sample_count=sample_count,
                max_product_value=max_product_value,
            )
        )

        return (
            self._convert_encoded_samples_to_batch(
                encoded_samples
            )
        )

    def _generate_encoded_samples(
        self,
        random_number_generator: random.Random,
        sample_count: int,
        max_product_value: int,
    ) -> list[dict[str, list[int]]]:
        encoded_samples = []

        for _ in range(sample_count):
            a_value, b_value = (
                self._sample_operands(
                    random_number_generator=(
                        random_number_generator
                    ),
                    max_product_value=(
                        max_product_value
                    ),
                )
            )

            encoded_samples.append(
                self._encode_sample(
                    a_value=a_value,
                    b_value=b_value,
                )
            )

        return encoded_samples

    def _sample_operands(
        self,
        random_number_generator: random.Random,
        max_product_value: int,
    ) -> tuple[int, int]:
        if (
            random_number_generator.random()
            < self.zero_sample_probability
        ):
            other_operand_maximum = min(
                max_product_value,
                self.maximum_operand_value,
            )

            a_value = 0

            b_value = (
                random_number_generator.randint(
                    0,
                    other_operand_maximum,
                )
            )
        else:
            sampled_product_ceiling = (
                random_number_generator.randint(
                    1,
                    max_product_value,
                )
            )

            first_operand_maximum = min(
                math.isqrt(
                    sampled_product_ceiling
                ),
                self.maximum_operand_value,
            )

            a_value = (
                random_number_generator.randint(
                    1,
                    first_operand_maximum,
                )
            )

            second_operand_maximum = min(
                sampled_product_ceiling
                // a_value,
                self.maximum_operand_value,
            )

            b_value = (
                random_number_generator.randint(
                    1,
                    second_operand_maximum,
                )
            )

        if (
            random_number_generator.random()
            < 0.5
        ):
            a_value, b_value = (
                b_value,
                a_value,
            )

        return a_value, b_value

    def _encode_sample(
        self,
        a_value: int,
        b_value: int,
    ) -> dict[str, list[int]]:
        product_value = (
            a_value * b_value
        )

        a_token_ids = self.tokenizer.encode(
            str(a_value)
        )

        b_token_ids = self.tokenizer.encode(
            str(b_value)
        )

        product_token_ids = (
            self.tokenizer.encode(
                str(product_value)
            )
        )

        a_token_ids.reverse()
        b_token_ids.reverse()
        product_token_ids.reverse()

        product_token_ids.append(
            self.eos_id
        )

        self._check_sequence_length(
            sequence_name="a",
            integer_value=a_value,
            token_ids=a_token_ids,
            maximum_sequence_length=(
                self.operand_seq_len
            ),
        )

        self._check_sequence_length(
            sequence_name="b",
            integer_value=b_value,
            token_ids=b_token_ids,
            maximum_sequence_length=(
                self.operand_seq_len
            ),
        )

        self._check_sequence_length(
            sequence_name="product",
            integer_value=product_value,
            token_ids=product_token_ids,
            maximum_sequence_length=(
                self.product_seq_len
            ),
        )

        return {
            "a": self._pad_token_ids_to_length(
                token_ids=a_token_ids,
                required_sequence_length=(
                    self.operand_seq_len
                ),
            ),
            "b": self._pad_token_ids_to_length(
                token_ids=b_token_ids,
                required_sequence_length=(
                    self.operand_seq_len
                ),
            ),
            "product": (
                self._pad_token_ids_to_length(
                    token_ids=product_token_ids,
                    required_sequence_length=(
                        self.product_seq_len
                    ),
                )
            ),
        }

    def _convert_encoded_samples_to_batch(
        self,
        encoded_samples: list[
            dict[str, list[int]]
        ],
    ) -> dict[str, torch.Tensor]:
        return {
            "a": torch.tensor(
                [
                    encoded_sample["a"]
                    for encoded_sample
                    in encoded_samples
                ],
                dtype=torch.long,
            ),
            "b": torch.tensor(
                [
                    encoded_sample["b"]
                    for encoded_sample
                    in encoded_samples
                ],
                dtype=torch.long,
            ),
            "product": torch.tensor(
                [
                    encoded_sample["product"]
                    for encoded_sample
                    in encoded_samples
                ],
                dtype=torch.long,
            ),
        }

    def _create_tensor_dataset(
        self,
        tensor_batch: dict[str, torch.Tensor],
    ) -> MultiplicationTensorDataset:
        return MultiplicationTensorDataset(
            a_token_ids=tensor_batch["a"],
            b_token_ids=tensor_batch["b"],
            product_token_ids=(
                tensor_batch["product"]
            ),
        )

    @staticmethod
    def _concatenate_tensor_batches(
        tensor_batches: list[
            dict[str, torch.Tensor]
        ],
    ) -> dict[str, torch.Tensor]:
        return {
            "a": torch.cat(
                [
                    tensor_batch["a"]
                    for tensor_batch
                    in tensor_batches
                ]
            ),
            "b": torch.cat(
                [
                    tensor_batch["b"]
                    for tensor_batch
                    in tensor_batches
                ]
            ),
            "product": torch.cat(
                [
                    tensor_batch["product"]
                    for tensor_batch
                    in tensor_batches
                ]
            ),
        }

    def _effective_max_product_value(
        self,
        max_product_value: int | float,
    ) -> int:
        return min(
            max(
                1,
                int(max_product_value),
            ),
            self.maximum_representable_product_value,
        )

    def _projected_max_product_value(
        self,
        lookahead_steps: int,
    ) -> float:
        projected_max_product_value = (
            self.current_max_product_value
        )

        for _ in range(lookahead_steps):
            projected_max_product_value = (
                self._grow_max_product_value(
                    projected_max_product_value
                )
            )

            if (
                projected_max_product_value
                >= self.maximum_representable_product_value
            ):
                break

        return projected_max_product_value

    def _grow_max_product_value(
        self,
        current_max_product_value: float,
    ) -> float:
        return min(
            float(
                self.maximum_representable_product_value
            ),
            current_max_product_value
            * (
                1.0
                + self.product_growth_rate
            ),
        )

    def _advance_curriculum(
        self,
    ) -> None:
        self.current_max_product_value = (
            self._grow_max_product_value(
                self.current_max_product_value
            )
        )

    def _configuration_state(
        self,
    ) -> dict:
        return {
            "batch_size": self.batch_size,
            "operand_seq_len": (
                self.operand_seq_len
            ),
            "product_seq_len": (
                self.product_seq_len
            ),
            "initial_max_product_value": (
                self.initial_max_product_value
            ),
            "product_growth_rate": (
                self.product_growth_rate
            ),
            "rehearsal_fraction": (
                self.rehearsal_fraction
            ),
            "rehearsal_max_product_value": (
                self.rehearsal_max_product_value
            ),
            "zero_sample_probability": (
                self.zero_sample_probability
            ),
            "seed": self.seed,
            "tokenizer_vocabulary": tuple(
                self.tokenizer.vocab_list
            ),
            "pad_id": self.pad_id,
            "eos_id": self.eos_id,
        }

    @staticmethod
    def _check_sequence_length(
        sequence_name: str,
        integer_value: int,
        token_ids: list[int],
        maximum_sequence_length: int,
    ) -> None:
        if (
            len(token_ids)
            > maximum_sequence_length
        ):
            raise ValueError(
                f"{sequence_name}={integer_value} "
                f"requires {len(token_ids)} tokens, "
                f"but its maximum sequence length is "
                f"{maximum_sequence_length}."
            )

    def _pad_token_ids_to_length(
        self,
        token_ids: list[int],
        required_sequence_length: int,
    ) -> list[int]:
        padding_length = (
            required_sequence_length
            - len(token_ids)
        )

        return (
            token_ids
            + [self.pad_id]
            * padding_length
        )