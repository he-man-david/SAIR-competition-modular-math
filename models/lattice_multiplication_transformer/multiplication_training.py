import os
from pathlib import Path
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from IPython.display import display
from matplotlib.ticker import MaxNLocator
from tqdm.auto import tqdm

from data_streams.multiplication_data_stream import MultiplicationDataStream
from tokenizer import DigitTokenizer


DEFAULT_CHECKPOINT_DIR = Path(
    "/teamspace/studios/this_studio/"
    "SAIR-competition-modular-math/checkpoints"
)


class TrainingHistory(TypedDict):
    global_step: int
    train_step: list[int]
    train_loss: list[float]
    train_exact_accuracy: list[float]
    train_token_accuracy: list[float]
    validation_step: list[int]
    validation_loss: list[float]
    validation_exact_accuracy: list[float]
    validation_token_accuracy: list[float]


def create_training_history() -> TrainingHistory:
    return {
        "global_step": 0,
        "train_step": [],
        "train_loss": [],
        "train_exact_accuracy": [],
        "train_token_accuracy": [],
        "validation_step": [],
        "validation_loss": [],
        "validation_exact_accuracy": [],
        "validation_token_accuracy": [],
    }


def truncate_predictions_after_eos(
    predictions: torch.Tensor,
    eos_id: int,
    pad_id: int,
) -> torch.Tensor:
    sequence_positions = torch.arange(
        predictions.shape[1],
        device=predictions.device,
    ).unsqueeze(0)

    eos_positions = predictions == eos_id
    sequence_has_eos = eos_positions.any(dim=1)
    first_eos_position = eos_positions.int().argmax(dim=1)

    positions_after_eos = (
        sequence_positions
        > first_eos_position.unsqueeze(1)
    ) & sequence_has_eos.unsqueeze(1)

    return predictions.masked_fill(
        positions_after_eos,
        pad_id,
    )


def compute_product_accuracies(
    product_predictions: torch.Tensor,
    product_targets: torch.Tensor,
    eos_id: int,
    pad_id: int,
) -> tuple[int, int, int, int]:
    product_predictions = truncate_predictions_after_eos(
        predictions=product_predictions,
        eos_id=eos_id,
        pad_id=pad_id,
    )

    valid_target_positions = product_targets != pad_id

    correct_product_tokens = (
        product_predictions == product_targets
    ) & valid_target_positions

    correct_product_sequences = (
        (product_predictions == product_targets)
        | ~valid_target_positions
    ).all(dim=1)

    return (
        correct_product_sequences.sum().item(),
        product_targets.shape[0],
        correct_product_tokens.sum().item(),
        valid_target_positions.sum().item(),
    )


@torch.no_grad()
def validate_model(
    model: nn.Module,
    data_stream: MultiplicationDataStream,
    loss_function: nn.Module,
    tokenizer: DigitTokenizer,
    device: torch.device,
    validation_steps: int,
    validation_lookahead_steps: int,
    use_bf16: bool,
) -> dict[str, float]:
    model_was_training = model.training
    model.eval()

    pad_id = tokenizer.char_to_int["<pad>"]
    eos_id = tokenizer.char_to_int["<eos>"]

    validation_loader = (
        data_stream.create_validation_chunk_loader(
            num_steps=validation_steps,
            lookahead_steps=validation_lookahead_steps,
            batch_size=data_stream.batch_size,
            seed=10_000,
        )
    )

    validation_loss_weighted_sum = 0.0
    validation_target_token_count = 0

    exact_correct_count = 0
    sequence_count = 0

    token_correct_count = 0
    token_count = 0

    try:
        for validation_batch in validation_loader:
            a = validation_batch["a"].to(
                device,
                non_blocking=True,
            )

            b = validation_batch["b"].to(
                device,
                non_blocking=True,
            )

            product_targets = validation_batch[
                "product"
            ].to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                product_logits = model(
                    a,
                    b,
                )

                validation_loss = loss_function(
                    product_logits.reshape(
                        -1,
                        tokenizer.vocab_size,
                    ),
                    product_targets.reshape(-1),
                )

            (
                batch_exact_correct_count,
                batch_sequence_count,
                batch_token_correct_count,
                batch_token_count,
            ) = compute_product_accuracies(
                product_predictions=(
                    product_logits.argmax(dim=-1)
                ),
                product_targets=product_targets,
                eos_id=eos_id,
                pad_id=pad_id,
            )

            validation_loss_weighted_sum += (
                validation_loss.item()
                * batch_token_count
            )

            validation_target_token_count += (
                batch_token_count
            )

            exact_correct_count += (
                batch_exact_correct_count
            )

            sequence_count += (
                batch_sequence_count
            )

            token_correct_count += (
                batch_token_correct_count
            )

            token_count += (
                batch_token_count
            )
    finally:
        model.train(
            model_was_training
        )

    return {
        "loss": (
            validation_loss_weighted_sum
            / validation_target_token_count
        ),
        "exact_accuracy": (
            exact_correct_count
            / sequence_count
        ),
        "token_accuracy": (
            token_correct_count
            / token_count
        ),
    }


def plot_training_history(
    training_history: TrainingHistory,
    display_handle: Any | None = None,
) -> Any | None:
    if not training_history["train_step"]:
        return display_handle

    figure, (
        loss_axis,
        accuracy_axis,
    ) = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        sharex=True,
    )

    loss_axis.plot(
        training_history["train_step"],
        training_history["train_loss"],
        label="Training loss",
        linewidth=2,
        linestyle="-",
    )

    if training_history["validation_loss"]:
        loss_axis.plot(
            training_history["validation_step"],
            training_history["validation_loss"],
            label="Validation loss",
            linewidth=2,
            linestyle="--",
            marker="o",
            markersize=4,
        )

    accuracy_axis.plot(
        training_history["train_step"],
        training_history[
            "train_exact_accuracy"
        ],
        label=(
            "Training exact accuracy "
            "(logged batch)"
        ),
        linewidth=2,
        linestyle="-",
    )

    accuracy_axis.plot(
        training_history["train_step"],
        training_history[
            "train_token_accuracy"
        ],
        label=(
            "Training token accuracy "
            "(logged batch)"
        ),
        linewidth=2,
        linestyle="-",
    )

    if training_history[
        "validation_exact_accuracy"
    ]:
        accuracy_axis.plot(
            training_history["validation_step"],
            training_history[
                "validation_exact_accuracy"
            ],
            label="Validation exact accuracy",
            linewidth=2,
            linestyle="--",
            marker="o",
            markersize=4,
        )

        accuracy_axis.plot(
            training_history["validation_step"],
            training_history[
                "validation_token_accuracy"
            ],
            label="Validation token accuracy",
            linewidth=2,
            linestyle="--",
            marker="o",
            markersize=4,
        )

    loss_axis.set_title(
        "Lattice Multiplication Training"
    )

    loss_axis.set_ylabel(
        "Cross-entropy loss"
    )

    accuracy_axis.set_xlabel(
        "Training step"
    )

    accuracy_axis.set_ylabel(
        "Accuracy"
    )

    accuracy_axis.set_ylim(
        0.0,
        1.0,
    )

    accuracy_axis.xaxis.set_major_locator(
        MaxNLocator(
            integer=True
        )
    )

    loss_axis.legend(
        loc="upper right"
    )

    accuracy_axis.legend(
        loc="lower right"
    )

    loss_axis.grid(
        alpha=0.3
    )

    accuracy_axis.grid(
        alpha=0.3
    )

    figure.tight_layout()

    if display_handle is None:
        display_handle = display(
            figure,
            display_id=True,
        )
    else:
        display_handle.update(
            figure
        )

    plt.close(
        figure
    )

    return display_handle


def train_model(
    model: nn.Module,
    data_stream: MultiplicationDataStream,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    tokenizer: DigitTokenizer,
    device: torch.device,
    training_history: TrainingHistory,
    steps: int,
    log_interval: int,
    validation_interval: int,
    validation_steps: int,
    validation_lookahead_steps: int,
    plot_interval: int,
    checkpoint_name: str,
    checkpoint_interval: int,
    gradient_clip_norm: float,
    use_bf16: bool,
    checkpoint_dir: str | os.PathLike = (
        DEFAULT_CHECKPOINT_DIR
    ),
) -> None:
    _validate_training_configuration(
        steps=steps,
        log_interval=log_interval,
        validation_interval=(
            validation_interval
        ),
        validation_steps=validation_steps,
        validation_lookahead_steps=(
            validation_lookahead_steps
        ),
        plot_interval=plot_interval,
        checkpoint_interval=(
            checkpoint_interval
        ),
        gradient_clip_norm=(
            gradient_clip_norm
        ),
    )

    pad_id = tokenizer.char_to_int[
        "<pad>"
    ]

    eos_id = tokenizer.char_to_int[
        "<eos>"
    ]

    (
        checkpoint_file_stem,
        checkpoint_file_suffix,
    ) = _split_checkpoint_name(
        checkpoint_name
    )

    running_training_loss_sum = (
        torch.zeros(
            (),
            device=device,
            dtype=torch.float32,
        )
    )

    running_training_step_count = 0

    display_handle = None

    last_plotted_history_size = (
        0,
        0,
    )

    progress_bar = tqdm(
        range(steps),
        desc="Training multiplication",
    )

    for training_iteration in progress_bar:
        model.train()

        training_batch = (
            data_stream.next_batch()
        )

        a = training_batch["a"].to(
            device,
            non_blocking=True,
        )

        b = training_batch["b"].to(
            device,
            non_blocking=True,
        )

        product_targets = training_batch[
            "product"
        ].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            product_logits = model(
                a,
                b,
            )

            training_loss = loss_function(
                product_logits.reshape(
                    -1,
                    tokenizer.vocab_size,
                ),
                product_targets.reshape(-1),
            )

        training_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_norm,
        )

        optimizer.step()

        training_history[
            "global_step"
        ] += 1

        global_step = training_history[
            "global_step"
        ]

        running_training_loss_sum += (
            training_loss.detach().float()
        )

        running_training_step_count += 1

        should_log = (
            global_step % log_interval == 0
            or training_iteration == steps - 1
        )

        if should_log:
            (
                logged_loss,
                logged_exact_accuracy,
                logged_token_accuracy,
            ) = _append_training_log(
                training_history=(
                    training_history
                ),
                running_training_loss_sum=(
                    running_training_loss_sum
                ),
                running_training_step_count=(
                    running_training_step_count
                ),
                product_logits=(
                    product_logits.detach()
                ),
                product_targets=(
                    product_targets
                ),
                eos_id=eos_id,
                pad_id=pad_id,
            )

            running_training_loss_sum.zero_()
            running_training_step_count = 0

            progress_bar.set_postfix(
                loss=f"{logged_loss:.4f}",
                exact=(
                    f"{logged_exact_accuracy:.4f}"
                ),
                token=(
                    f"{logged_token_accuracy:.4f}"
                ),
                max_product=(
                    f"{data_stream.current_max_product_value:.2f}"
                ),
            )

        if (
            global_step
            % validation_interval
            == 0
        ):
            validation_metrics = (
                validate_model(
                    model=model,
                    data_stream=data_stream,
                    loss_function=loss_function,
                    tokenizer=tokenizer,
                    device=device,
                    validation_steps=(
                        validation_steps
                    ),
                    validation_lookahead_steps=(
                        validation_lookahead_steps
                    ),
                    use_bf16=use_bf16,
                )
            )

            training_history[
                "validation_step"
            ].append(
                global_step
            )

            training_history[
                "validation_loss"
            ].append(
                validation_metrics["loss"]
            )

            training_history[
                "validation_exact_accuracy"
            ].append(
                validation_metrics[
                    "exact_accuracy"
                ]
            )

            training_history[
                "validation_token_accuracy"
            ].append(
                validation_metrics[
                    "token_accuracy"
                ]
            )

        if (
            global_step
            % checkpoint_interval
            == 0
        ):
            checkpoint_file_name = (
                f"{global_step}_"
                f"{checkpoint_file_stem}"
                f"{checkpoint_file_suffix}"
            )

            save_checkpoint(
                file_name=checkpoint_file_name,
                model=model,
                optimizer=optimizer,
                data_stream=data_stream,
                training_history=(
                    training_history
                ),
                checkpoint_dir=checkpoint_dir,
                training_configuration={
                    "log_interval": (
                        log_interval
                    ),
                    "validation_interval": (
                        validation_interval
                    ),
                    "validation_steps": (
                        validation_steps
                    ),
                    "validation_lookahead_steps": (
                        validation_lookahead_steps
                    ),
                    "plot_interval": (
                        plot_interval
                    ),
                    "checkpoint_name": (
                        checkpoint_name
                    ),
                    "checkpoint_interval": (
                        checkpoint_interval
                    ),
                    "gradient_clip_norm": (
                        gradient_clip_norm
                    ),
                    "use_bf16": (
                        use_bf16
                    ),
                },
            )

        if (
            global_step
            % plot_interval
            == 0
        ):
            current_history_size = (
                len(
                    training_history[
                        "train_step"
                    ]
                ),
                len(
                    training_history[
                        "validation_step"
                    ]
                ),
            )

            if (
                current_history_size
                != last_plotted_history_size
            ):
                display_handle = (
                    plot_training_history(
                        training_history=(
                            training_history
                        ),
                        display_handle=(
                            display_handle
                        ),
                    )
                )

                last_plotted_history_size = (
                    current_history_size
                )

    final_history_size = (
        len(
            training_history[
                "train_step"
            ]
        ),
        len(
            training_history[
                "validation_step"
            ]
        ),
    )

    if (
        final_history_size
        != last_plotted_history_size
    ):
        plot_training_history(
            training_history=training_history,
            display_handle=display_handle,
        )


def save_checkpoint(
    file_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_stream: MultiplicationDataStream,
    training_history: TrainingHistory,
    checkpoint_dir: str | os.PathLike = (
        DEFAULT_CHECKPOINT_DIR
    ),
    training_configuration: dict | None = None,
) -> None:
    checkpoint_directory = Path(
        checkpoint_dir
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_directory
        / file_name
    )

    torch.save(
        {
            "checkpoint_version": 1,
            "runtime_metadata": (
                _runtime_metadata(
                    model=model,
                    optimizer=optimizer,
                    data_stream=data_stream,
                )
            ),
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "data_stream_state_dict": (
                data_stream.state_dict()
            ),
            "training_history": (
                training_history
            ),
            "training_configuration": (
                training_configuration
            ),
            "torch_cpu_rng_state": (
                torch.get_rng_state()
            ),
            "torch_cuda_rng_states": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
        checkpoint_path,
    )

    print(
        f"Checkpoint saved at step "
        f"{training_history['global_step']:,}: "
        f"{checkpoint_path}"
    )


def load_checkpoint(
    file_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_stream: MultiplicationDataStream,
    training_history: TrainingHistory,
    device: torch.device,
    checkpoint_dir: str | os.PathLike = (
        DEFAULT_CHECKPOINT_DIR
    ),
) -> None:
    checkpoint_path = (
        Path(checkpoint_dir)
        / file_name
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    required_checkpoint_keys = {
        "checkpoint_version",
        "runtime_metadata",
        "model_state_dict",
        "optimizer_state_dict",
        "data_stream_state_dict",
        "training_history",
        "training_configuration",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
    }

    missing_checkpoint_keys = (
        required_checkpoint_keys
        - checkpoint.keys()
    )

    if missing_checkpoint_keys:
        raise KeyError(
            "Missing checkpoint keys: "
            f"{sorted(missing_checkpoint_keys)}"
        )

    if (
        checkpoint["checkpoint_version"]
        != 1
    ):
        raise ValueError(
            "Unsupported checkpoint version: "
            f"{checkpoint['checkpoint_version']}."
        )

    saved_runtime_metadata = checkpoint[
        "runtime_metadata"
    ]

    current_runtime_metadata = (
        _runtime_metadata(
            model=model,
            optimizer=optimizer,
            data_stream=data_stream,
        )
    )

    _validate_runtime_metadata(
        saved_runtime_metadata=(
            saved_runtime_metadata
        ),
        current_runtime_metadata=(
            current_runtime_metadata
        ),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    data_stream.load_state_dict(
        checkpoint[
            "data_stream_state_dict"
        ]
    )

    training_history.clear()

    training_history.update(
        checkpoint["training_history"]
    )

    torch.set_rng_state(
        checkpoint[
            "torch_cpu_rng_state"
        ].cpu()
    )

    saved_cuda_rng_states = checkpoint[
        "torch_cuda_rng_states"
    ]

    if (
        torch.cuda.is_available()
        and saved_cuda_rng_states is not None
    ):
        torch.cuda.set_rng_state_all(
            [
                rng_state.cpu()
                for rng_state
                in saved_cuda_rng_states
            ]
        )

    print(
        f"Checkpoint loaded from step "
        f"{training_history['global_step']:,}: "
        f"{checkpoint_path}"
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_stream: MultiplicationDataStream,
    tokenizer: DigitTokenizer,
    device: torch.device,
    max_product_value: int,
    number_of_samples: int,
    use_bf16: bool,
) -> None:
    model_was_training = model.training
    model.eval()

    pad_id = tokenizer.char_to_int[
        "<pad>"
    ]

    eos_id = tokenizer.char_to_int[
        "<eos>"
    ]

    evaluation_loader = (
        data_stream.create_fixed_loader(
            num_samples=number_of_samples,
            max_product_value=(
                max_product_value
            ),
            batch_size=(
                data_stream.batch_size
            ),
            seed=42,
        )
    )

    exact_correct_count = 0
    sequence_count = 0

    wrong_predictions: list[
        tuple[
            str,
            str,
            str,
            str,
        ]
    ] = []

    try:
        for evaluation_batch in tqdm(
            evaluation_loader,
            desc="Evaluating multiplication",
        ):
            a_token_ids = evaluation_batch[
                "a"
            ]

            b_token_ids = evaluation_batch[
                "b"
            ]

            product_target_token_ids = (
                evaluation_batch[
                    "product"
                ]
            )

            a = a_token_ids.to(
                device,
                non_blocking=True,
            )

            b = b_token_ids.to(
                device,
                non_blocking=True,
            )

            product_targets = (
                product_target_token_ids.to(
                    device,
                    non_blocking=True,
                )
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                product_logits = model(
                    a,
                    b,
                )

            product_predictions = (
                truncate_predictions_after_eos(
                    predictions=(
                        product_logits.argmax(
                            dim=-1
                        )
                    ),
                    eos_id=eos_id,
                    pad_id=pad_id,
                )
            )

            correct_sequences = (
                (
                    product_predictions
                    == product_targets
                )
                | (
                    product_targets
                    == pad_id
                )
            ).all(
                dim=1
            )

            exact_correct_count += (
                correct_sequences.sum().item()
            )

            sequence_count += (
                product_targets.shape[0]
            )

            if len(wrong_predictions) < 10:
                product_predictions_cpu = (
                    product_predictions.cpu()
                )

                correct_sequences_cpu = (
                    correct_sequences.cpu()
                )

                for sample_index in range(
                    correct_sequences_cpu.shape[0]
                ):
                    if correct_sequences_cpu[
                        sample_index
                    ].item():
                        continue

                    wrong_predictions.append(
                        (
                            _decode_reversed_number(
                                a_token_ids[
                                    sample_index
                                ],
                                tokenizer,
                            ),
                            _decode_reversed_number(
                                b_token_ids[
                                    sample_index
                                ],
                                tokenizer,
                            ),
                            _decode_reversed_number(
                                product_target_token_ids[
                                    sample_index
                                ],
                                tokenizer,
                            ),
                            _decode_reversed_number(
                                product_predictions_cpu[
                                    sample_index
                                ],
                                tokenizer,
                            ),
                        )
                    )

                    if (
                        len(wrong_predictions)
                        == 10
                    ):
                        break
    finally:
        model.train(
            model_was_training
        )

    accuracy = (
        exact_correct_count
        / sequence_count
    )

    print(
        f"Accuracy: {accuracy:.6f} "
        f"({exact_correct_count:,}/"
        f"{sequence_count:,})"
    )

    if not wrong_predictions:
        print(
            "\nNo wrong predictions."
        )
        return

    print(
        "\nWrong predictions:"
    )

    for (
        a_text,
        b_text,
        expected_product_text,
        predicted_product_text,
    ) in wrong_predictions:
        print(
            f"\n{a_text} × {b_text} "
            f"= {expected_product_text}"
        )

        print(
            "Prediction:",
            predicted_product_text,
        )


def _append_training_log(
    training_history: TrainingHistory,
    running_training_loss_sum: torch.Tensor,
    running_training_step_count: int,
    product_logits: torch.Tensor,
    product_targets: torch.Tensor,
    eos_id: int,
    pad_id: int,
) -> tuple[float, float, float]:
    average_training_loss = (
        running_training_loss_sum
        / running_training_step_count
    ).item()

    (
        exact_correct_count,
        sequence_count,
        token_correct_count,
        token_count,
    ) = compute_product_accuracies(
        product_predictions=(
            product_logits.argmax(dim=-1)
        ),
        product_targets=product_targets,
        eos_id=eos_id,
        pad_id=pad_id,
    )

    exact_accuracy = (
        exact_correct_count
        / sequence_count
    )

    token_accuracy = (
        token_correct_count
        / token_count
    )

    training_history[
        "train_step"
    ].append(
        training_history["global_step"]
    )

    training_history[
        "train_loss"
    ].append(
        average_training_loss
    )

    training_history[
        "train_exact_accuracy"
    ].append(
        exact_accuracy
    )

    training_history[
        "train_token_accuracy"
    ].append(
        token_accuracy
    )

    return (
        average_training_loss,
        exact_accuracy,
        token_accuracy,
    )


def _decode_reversed_number(
    token_ids: torch.Tensor,
    tokenizer: DigitTokenizer,
) -> str:
    return tokenizer.decode(
        token_ids.tolist()
    )[::-1]


def _split_checkpoint_name(
    checkpoint_name: str,
) -> tuple[str, str]:
    checkpoint_name_path = Path(
        checkpoint_name
    )

    checkpoint_file_stem = (
        checkpoint_name_path.stem
    )

    checkpoint_file_suffix = (
        checkpoint_name_path.suffix
        or ".pt"
    )

    if not checkpoint_file_stem:
        raise ValueError(
            "checkpoint_name must contain "
            "a file name."
        )

    return (
        checkpoint_file_stem,
        checkpoint_file_suffix,
    )


def _runtime_metadata(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_stream: MultiplicationDataStream,
) -> dict:
    model_configuration_names = (
        "vocab_size",
        "pad_id",
        "max_seq_len",
        "d_model",
        "transformer_dim",
        "rows_per_step",
        "product_seq_len",
    )

    model_configuration = {
        configuration_name: getattr(
            model,
            configuration_name,
        )
        for configuration_name
        in model_configuration_names
        if hasattr(
            model,
            configuration_name,
        )
    }

    return {
        "model_class": (
            type(model).__name__
        ),
        "model_configuration": (
            model_configuration
        ),
        "optimizer_class": (
            type(optimizer).__name__
        ),
        "data_stream_class": (
            type(data_stream).__name__
        ),
        "torch_version": (
            torch.__version__
        ),
    }


def _validate_runtime_metadata(
    saved_runtime_metadata: dict,
    current_runtime_metadata: dict,
) -> None:
    compatibility_keys = (
        "model_class",
        "model_configuration",
        "optimizer_class",
        "data_stream_class",
    )

    mismatches = {
        key: {
            "saved": (
                saved_runtime_metadata.get(
                    key
                )
            ),
            "current": (
                current_runtime_metadata.get(
                    key
                )
            ),
        }
        for key in compatibility_keys
        if (
            saved_runtime_metadata.get(key)
            != current_runtime_metadata.get(key)
        )
    }

    if mismatches:
        raise ValueError(
            "The checkpoint does not match "
            "the current training objects: "
            f"{mismatches}"
        )


def _validate_training_configuration(
    steps: int,
    log_interval: int,
    validation_interval: int,
    validation_steps: int,
    validation_lookahead_steps: int,
    plot_interval: int,
    checkpoint_interval: int,
    gradient_clip_norm: float,
) -> None:
    positive_integer_values = {
        "steps": steps,
        "log_interval": log_interval,
        "validation_interval": (
            validation_interval
        ),
        "validation_steps": (
            validation_steps
        ),
        "plot_interval": plot_interval,
        "checkpoint_interval": (
            checkpoint_interval
        ),
    }

    for (
        value_name,
        value,
    ) in positive_integer_values.items():
        if value <= 0:
            raise ValueError(
                f"{value_name} must be "
                "greater than 0."
            )

    if validation_lookahead_steps < 0:
        raise ValueError(
            "validation_lookahead_steps "
            "must be non-negative."
        )

    if gradient_clip_norm <= 0:
        raise ValueError(
            "gradient_clip_norm must be "
            "greater than 0."
        )