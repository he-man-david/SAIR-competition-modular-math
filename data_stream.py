import math
import random

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from tokenizer import DigitTokenizer


class MultiplicationModuloTensorDataset(Dataset):
    def __init__(
        self,
        a_token_ids: torch.Tensor,
        b_token_ids: torch.Tensor,
        modulus_token_ids: torch.Tensor,
        product_token_ids: torch.Tensor,
        modular_result_token_ids: torch.Tensor,
    ):
        self.a_token_ids = a_token_ids
        self.b_token_ids = b_token_ids
        self.modulus_token_ids = modulus_token_ids
        self.product_token_ids = product_token_ids
        self.modular_result_token_ids = modular_result_token_ids

    def __len__(self) -> int:
        return self.a_token_ids.shape[0]

    def __getitem__(
        self,
        sample_index: int,
    ) -> dict[str, torch.Tensor]:
        return {
            "a": self.a_token_ids[sample_index],
            "b": self.b_token_ids[sample_index],
            "modulus": self.modulus_token_ids[sample_index],
            "product": self.product_token_ids[sample_index],
            "modular_result": self.modular_result_token_ids[
                sample_index
            ],
        }


class MultiplicationModuloDataStream:
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
            raise ValueError(
                "batch_size must be greater than 0."
            )

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

        self.operand_seq_len = max_seq_len
        self.modulus_seq_len = max_seq_len
        self.modular_result_seq_len = max_seq_len
        self.product_seq_len = 2 * max_seq_len + 1

        self.initial_max = float(initial_max)
        self.current_max = float(initial_max)
        self.growth_rate = float(growth_rate)

        self.beta_alpha = float(beta_alpha)
        self.beta_beta = float(beta_beta)

        self.rehearsal_fraction = float(
            rehearsal_fraction
        )

        self.rehearsal_max = rehearsal_max

        self.max_operand_multiple = (
            max_operand_multiple
        )

        self.pad_id = tokenizer.char_to_int[
            "<pad>"
        ]

        self.eos_id = tokenizer.char_to_int[
            "<eos>"
        ]

        self.max_operand_value = (
            10**self.operand_seq_len
        ) - 1

        self.max_modulus_value = (
            10 ** (self.modulus_seq_len - 1)
        )

        self.seed = seed
        self.random_number_generator = random.Random(
            seed
        )

        self.step = 0

    def next_batch(
        self,
    ) -> dict[str, torch.Tensor]:
        rehearsal_sample_count = round(
            self.batch_size
            * self.rehearsal_fraction
        )

        curriculum_sample_count = (
            self.batch_size
            - rehearsal_sample_count
        )

        encoded_samples: list[
            dict[str, list[int]]
        ] = []

        curriculum_generation_maximum = (
            self._get_generation_maximum(
                self.current_max
            )
        )

        for _ in range(
            curriculum_sample_count
        ):
            (
                a_value,
                b_value,
                modulus_value,
            ) = (
                self._sample_operands_and_modulus(
                    random_number_generator=(
                        self.random_number_generator
                    ),
                    maximum_modular_result=(
                        curriculum_generation_maximum
                    ),
                )
            )

            encoded_samples.append(
                self._encode_sample(
                    a_value=a_value,
                    b_value=b_value,
                    modulus_value=modulus_value,
                )
            )

        rehearsal_generation_maximum = (
            self._get_generation_maximum(
                self.rehearsal_max
            )
        )

        for _ in range(
            rehearsal_sample_count
        ):
            (
                a_value,
                b_value,
                modulus_value,
            ) = (
                self._sample_operands_and_modulus(
                    random_number_generator=(
                        self.random_number_generator
                    ),
                    maximum_modular_result=(
                        rehearsal_generation_maximum
                    ),
                )
            )

            encoded_samples.append(
                self._encode_sample(
                    a_value=a_value,
                    b_value=b_value,
                    modulus_value=modulus_value,
                )
            )

        self.random_number_generator.shuffle(
            encoded_samples
        )

        batch = self._convert_encoded_samples_to_batch(
            encoded_samples
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

        validation_random_number_generator = (
            random.Random(seed)
        )

        validation_maximum = (
            self.current_max
            * (
                1.0 + self.growth_rate
            ) ** lookahead_steps
        )

        rehearsal_sample_count = round(
            validation_batch_size
            * self.rehearsal_fraction
        )

        curriculum_sample_count = (
            validation_batch_size
            - rehearsal_sample_count
        )

        encoded_validation_samples: list[
            dict[str, list[int]]
        ] = []

        rehearsal_generation_maximum = (
            self._get_generation_maximum(
                self.rehearsal_max
            )
        )

        for _ in range(num_steps):
            curriculum_generation_maximum = (
                self._get_generation_maximum(
                    validation_maximum
                )
            )

            encoded_validation_step_samples: list[
                dict[str, list[int]]
            ] = []

            for _ in range(
                curriculum_sample_count
            ):
                (
                    a_value,
                    b_value,
                    modulus_value,
                ) = (
                    self._sample_operands_and_modulus(
                        random_number_generator=(
                            validation_random_number_generator
                        ),
                        maximum_modular_result=(
                            curriculum_generation_maximum
                        ),
                    )
                )

                encoded_validation_step_samples.append(
                    self._encode_sample(
                        a_value=a_value,
                        b_value=b_value,
                        modulus_value=modulus_value,
                    )
                )

            for _ in range(
                rehearsal_sample_count
            ):
                (
                    a_value,
                    b_value,
                    modulus_value,
                ) = (
                    self._sample_operands_and_modulus(
                        random_number_generator=(
                            validation_random_number_generator
                        ),
                        maximum_modular_result=(
                            rehearsal_generation_maximum
                        ),
                    )
                )

                encoded_validation_step_samples.append(
                    self._encode_sample(
                        a_value=a_value,
                        b_value=b_value,
                        modulus_value=modulus_value,
                    )
                )

            validation_random_number_generator.shuffle(
                encoded_validation_step_samples
            )

            encoded_validation_samples.extend(
                encoded_validation_step_samples
            )

            validation_maximum *= (
                1.0 + self.growth_rate
            )

        validation_dataset = (
            self._create_tensor_dataset(
                encoded_validation_samples
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

        evaluation_random_number_generator = (
            random.Random(seed)
        )

        fixed_generation_maximum = (
            self._get_generation_maximum(
                max_value
            )
        )

        encoded_evaluation_samples: list[
            dict[str, list[int]]
        ] = []

        for _ in range(num_samples):
            (
                a_value,
                b_value,
                modulus_value,
            ) = (
                self._sample_operands_and_modulus(
                    random_number_generator=(
                        evaluation_random_number_generator
                    ),
                    maximum_modular_result=(
                        fixed_generation_maximum
                    ),
                )
            )

            encoded_evaluation_samples.append(
                self._encode_sample(
                    a_value=a_value,
                    b_value=b_value,
                    modulus_value=modulus_value,
                )
            )

        evaluation_dataset = (
            self._create_tensor_dataset(
                encoded_evaluation_samples
            )
        )

        return DataLoader(
            evaluation_dataset,
            batch_size=loader_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
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

        original_data_stream_state = (
            self.state_dict()
        )

        encoded_training_samples: list[
            dict[str, list[int]]
        ] = []

        for _ in tqdm(
            range(num_steps),
            desc=(
                f"Generating {num_steps:,} "
                f"training batches"
            ),
        ):
            training_batch = self.next_batch()

            for sample_index in range(
                self.batch_size
            ):
                encoded_training_samples.append(
                    {
                        "a": training_batch[
                            "a"
                        ][sample_index].tolist(),
                        "b": training_batch[
                            "b"
                        ][sample_index].tolist(),
                        "modulus": training_batch[
                            "modulus"
                        ][sample_index].tolist(),
                        "product": training_batch[
                            "product"
                        ][sample_index].tolist(),
                        "modular_result": (
                            training_batch[
                                "modular_result"
                            ][sample_index].tolist()
                        ),
                    }
                )

        self.load_state_dict(
            original_data_stream_state
        )

        training_dataset = (
            self._create_tensor_dataset(
                encoded_training_samples
            )
        )

        return DataLoader(
            training_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=pin_memory,
        )

    def _get_generation_maximum(
        self,
        maximum_value: int | float,
    ) -> int:
        return min(
            max(
                2,
                int(maximum_value),
            ),
            self.max_modulus_value,
        )

    def _sample_operands_and_modulus(
        self,
        random_number_generator: random.Random,
        maximum_modular_result: int,
    ) -> tuple[int, int, int]:
        modular_result_value = int(
            random_number_generator.betavariate(
                self.beta_alpha,
                self.beta_beta,
            )
            * maximum_modular_result
        )

        modulus_value = (
            random_number_generator.randint(
                max(
                    2,
                    modular_result_value + 1,
                ),
                maximum_modular_result,
            )
        )

        while True:
            base_a_value = (
                random_number_generator.randint(
                    1,
                    modulus_value - 1,
                )
            )

            if (
                math.gcd(
                    base_a_value,
                    modulus_value,
                )
                == 1
            ):
                break

        inverse_a_value = pow(
            base_a_value,
            -1,
            modulus_value,
        )

        base_b_value = (
            modular_result_value
            * inverse_a_value
        ) % modulus_value

        maximum_a_modulus_multiple = min(
            self.max_operand_multiple,
            (
                self.max_operand_value
                - base_a_value
            )
            // modulus_value,
        )

        maximum_b_modulus_multiple = min(
            self.max_operand_multiple,
            (
                self.max_operand_value
                - base_b_value
            )
            // modulus_value,
        )

        a_modulus_multiple = (
            random_number_generator.randint(
                0,
                maximum_a_modulus_multiple,
            )
        )

        b_modulus_multiple = (
            random_number_generator.randint(
                0,
                maximum_b_modulus_multiple,
            )
        )

        a_value = (
            base_a_value
            + a_modulus_multiple
            * modulus_value
        )

        b_value = (
            base_b_value
            + b_modulus_multiple
            * modulus_value
        )

        if (
            random_number_generator.random()
            < 0.5
        ):
            a_value, b_value = (
                b_value,
                a_value,
            )

        return (
            a_value,
            b_value,
            modulus_value,
        )

    def _encode_sample(
        self,
        a_value: int,
        b_value: int,
        modulus_value: int,
    ) -> dict[str, list[int]]:
        product_value = (
            a_value * b_value
        )

        modular_result_value = (
            product_value % modulus_value
        )

        a_token_ids = self.tokenizer.encode(
            str(a_value)
        )

        b_token_ids = self.tokenizer.encode(
            str(b_value)
        )

        modulus_token_ids = (
            self.tokenizer.encode(
                str(modulus_value)
            )
        )

        product_token_ids = (
            self.tokenizer.encode(
                str(product_value)
            )
        )

        modular_result_token_ids = (
            self.tokenizer.encode(
                str(modular_result_value)
            )
        )

        a_token_ids.reverse()
        b_token_ids.reverse()
        modulus_token_ids.reverse()
        product_token_ids.reverse()
        modular_result_token_ids.reverse()

        product_token_ids.append(
            self.eos_id
        )

        modular_result_token_ids.append(
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
            sequence_name="modulus",
            integer_value=modulus_value,
            token_ids=modulus_token_ids,
            maximum_sequence_length=(
                self.modulus_seq_len
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

        self._check_sequence_length(
            sequence_name="modular_result",
            integer_value=(
                modular_result_value
            ),
            token_ids=(
                modular_result_token_ids
            ),
            maximum_sequence_length=(
                self.modular_result_seq_len
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
            "modulus": (
                self._pad_token_ids_to_length(
                    token_ids=modulus_token_ids,
                    required_sequence_length=(
                        self.modulus_seq_len
                    ),
                )
            ),
            "product": (
                self._pad_token_ids_to_length(
                    token_ids=product_token_ids,
                    required_sequence_length=(
                        self.product_seq_len
                    ),
                )
            ),
            "modular_result": (
                self._pad_token_ids_to_length(
                    token_ids=(
                        modular_result_token_ids
                    ),
                    required_sequence_length=(
                        self.modular_result_seq_len
                    ),
                )
            ),
        }

    def _check_sequence_length(
        self,
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
            "modulus": torch.tensor(
                [
                    encoded_sample[
                        "modulus"
                    ]
                    for encoded_sample
                    in encoded_samples
                ],
                dtype=torch.long,
            ),
            "product": torch.tensor(
                [
                    encoded_sample[
                        "product"
                    ]
                    for encoded_sample
                    in encoded_samples
                ],
                dtype=torch.long,
            ),
            "modular_result": torch.tensor(
                [
                    encoded_sample[
                        "modular_result"
                    ]
                    for encoded_sample
                    in encoded_samples
                ],
                dtype=torch.long,
            ),
        }

    def _create_tensor_dataset(
        self,
        encoded_samples: list[
            dict[str, list[int]]
        ],
    ) -> MultiplicationModuloTensorDataset:
        tensor_batch = (
            self._convert_encoded_samples_to_batch(
                encoded_samples
            )
        )

        return MultiplicationModuloTensorDataset(
            a_token_ids=tensor_batch["a"],
            b_token_ids=tensor_batch["b"],
            modulus_token_ids=(
                tensor_batch["modulus"]
            ),
            product_token_ids=(
                tensor_batch["product"]
            ),
            modular_result_token_ids=(
                tensor_batch[
                    "modular_result"
                ]
            ),
        )

    def state_dict(self) -> dict:
        return {
            "step": self.step,
            "current_max": self.current_max,
            "random_state": (
                self.random_number_generator.getstate()
            ),
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
            required_keys - state.keys()
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

        self.random_number_generator.setstate(
            state["random_state"]
        )

    def reset(self) -> None:
        self.step = 0
        self.current_max = self.initial_max

        self.random_number_generator = (
            random.Random(
                self.seed
            )
        )