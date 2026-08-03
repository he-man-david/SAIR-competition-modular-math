import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from IPython.display import display
from tqdm.auto import tqdm


def create_training_history() -> dict:
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
    sequence_length = predictions.shape[1]

    sequence_positions = torch.arange(
        sequence_length,
        device=predictions.device,
    ).unsqueeze(0)

    eos_positions = predictions == eos_id

    sequence_has_eos = eos_positions.any(
        dim=1
    )

    first_eos_position = eos_positions.int().argmax(
        dim=1
    )

    positions_after_eos = (
        sequence_positions
        > first_eos_position.unsqueeze(1)
    )

    positions_after_eos = (
        positions_after_eos
        & sequence_has_eos.unsqueeze(1)
    )

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
    product_predictions = (
        truncate_predictions_after_eos(
            predictions=product_predictions,
            eos_id=eos_id,
            pad_id=pad_id,
        )
    )

    valid_target_positions = (
        product_targets != pad_id
    )

    correct_product_tokens = (
        (product_predictions == product_targets)
        & valid_target_positions
    )

    token_correct_count = (
        correct_product_tokens.sum().item()
    )

    token_count = (
        valid_target_positions.sum().item()
    )

    correct_or_target_padding = (
        (product_predictions == product_targets)
        | ~valid_target_positions
    )

    correct_product_sequences = (
        correct_or_target_padding.all(
            dim=1
        )
    )

    exact_correct_count = (
        correct_product_sequences.sum().item()
    )

    sequence_count = product_targets.shape[0]

    return (
        exact_correct_count,
        sequence_count,
        token_correct_count,
        token_count,
    )


@torch.no_grad()
def validate_model(
    model: nn.Module,
    data_stream,
    loss_function: nn.Module,
    tokenizer,
    device: torch.device,
    batch_size: int,
    validation_steps: int,
    use_bf16: bool,
) -> dict[str, float]:
    model_was_training = model.training
    model.eval()

    pad_id = tokenizer.char_to_int["<pad>"]
    eos_id = tokenizer.char_to_int["<eos>"]

    validation_loader = (
        data_stream.create_validation_chunk_loader(
            num_steps=validation_steps,
            lookahead_steps=0,
            batch_size=batch_size,
            seed=10_000,
        )
    )

    validation_loss_sum = 0.0
    validation_batch_count = 0

    exact_correct_count = 0
    sequence_count = 0

    token_correct_count = 0
    token_count = 0

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

        product_predictions = (
            product_logits.argmax(
                dim=-1
            )
        )

        (
            batch_exact_correct_count,
            batch_sequence_count,
            batch_token_correct_count,
            batch_token_count,
        ) = compute_product_accuracies(
            product_predictions=product_predictions,
            product_targets=product_targets,
            eos_id=eos_id,
            pad_id=pad_id,
        )

        validation_loss_sum += (
            validation_loss.item()
        )

        validation_batch_count += 1

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

    model.train(
        model_was_training
    )

    return {
        "loss": (
            validation_loss_sum
            / validation_batch_count
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
    training_history: dict,
    figure,
    loss_axis,
    accuracy_axis,
    display_handle,
) -> None:
    loss_axis.clear()
    accuracy_axis.clear()

    loss_axis.plot(
        training_history["train_step"],
        training_history["train_loss"],
        label="Training loss",
        linewidth=2,
    )

    if training_history["validation_loss"]:
        loss_axis.plot(
            training_history["validation_step"],
            training_history["validation_loss"],
            label="Validation loss",
            linewidth=2,
            marker="o",
        )

    accuracy_axis.plot(
        training_history["train_step"],
        training_history[
            "train_exact_accuracy"
        ],
        label="Training exact accuracy",
        linewidth=2,
    )

    accuracy_axis.plot(
        training_history["train_step"],
        training_history[
            "train_token_accuracy"
        ],
        label="Training token accuracy",
        linewidth=2,
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
            marker="o",
        )

        accuracy_axis.plot(
            training_history["validation_step"],
            training_history[
                "validation_token_accuracy"
            ],
            label="Validation token accuracy",
            linewidth=2,
            marker="o",
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

    display_handle.update(
        figure
    )


def train_model(
    model: nn.Module,
    data_stream,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    tokenizer,
    device: torch.device,
    training_history: dict,
    steps: int,
    batch_size: int,
    validation_interval: int,
    validation_steps: int,
    plot_interval: int,
    gradient_clip_norm: float,
    use_bf16: bool,
) -> None:
    pad_id = tokenizer.char_to_int["<pad>"]
    eos_id = tokenizer.char_to_int["<eos>"]

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        sharex=True,
    )

    loss_axis = axes[0]
    accuracy_axis = axes[1]

    display_handle = display(
        figure,
        display_id=True,
    )

    progress_bar = tqdm(
        range(steps),
        desc="Training multiplication",
    )

    for _ in progress_bar:
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

        training_history["global_step"] += 1

        with torch.no_grad():
            product_predictions = (
                product_logits.argmax(
                    dim=-1
                )
            )

            (
                exact_correct_count,
                sequence_count,
                token_correct_count,
                token_count,
            ) = compute_product_accuracies(
                product_predictions=(
                    product_predictions
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
            training_loss.item()
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

        if (
            training_history["global_step"]
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
                    batch_size=batch_size,
                    validation_steps=(
                        validation_steps
                    ),
                    use_bf16=use_bf16,
                )
            )

            training_history[
                "validation_step"
            ].append(
                training_history["global_step"]
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
            training_history["global_step"]
            % plot_interval
            == 0
        ):
            plot_training_history(
                training_history=training_history,
                figure=figure,
                loss_axis=loss_axis,
                accuracy_axis=accuracy_axis,
                display_handle=display_handle,
            )

        progress_bar.set_postfix(
            loss=f"{training_loss.item():.4f}",
            exact=f"{exact_accuracy:.4f}",
            token=f"{token_accuracy:.4f}",
            current_max=(
                f"{data_stream.current_max:.2f}"
            ),
        )

    plot_training_history(
        training_history=training_history,
        figure=figure,
        loss_axis=loss_axis,
        accuracy_axis=accuracy_axis,
        display_handle=display_handle,
    )


def save_checkpoint(
    file_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_stream,
    training_history: dict,
    checkpoint_folder: str = "checkpoint",
) -> None:
    os.makedirs(
        checkpoint_folder,
        exist_ok=True,
    )

    checkpoint_path = os.path.join(
        checkpoint_folder,
        file_name,
    )

    torch.save(
        {
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
    data_stream,
    training_history: dict,
    device: torch.device,
    checkpoint_folder: str = "checkpoint",
) -> None:
    checkpoint_path = os.path.join(
        checkpoint_folder,
        file_name,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    data_stream.load_state_dict(
        checkpoint["data_stream_state_dict"]
    )

    training_history.clear()

    training_history.update(
        checkpoint["training_history"]
    )

    print(
        f"Checkpoint loaded from step "
        f"{training_history['global_step']:,}: "
        f"{checkpoint_path}"
    )


def _decode_reversed_number(
    token_ids: torch.Tensor,
    tokenizer,
    eos_id: int,
    pad_id: int,
) -> str:
    reversed_digit_characters = []

    for token_id in token_ids.tolist():
        if token_id == eos_id:
            break

        if token_id == pad_id:
            continue

        reversed_digit_characters.append(
            tokenizer.int_to_char[token_id]
        )

    return "".join(
        reversed(
            reversed_digit_characters
        )
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_stream,
    tokenizer,
    device: torch.device,
    max_value: int,
    number_of_samples: int,
    batch_size: int,
    use_bf16: bool,
) -> None:
    model_was_training = model.training
    model.eval()

    pad_id = tokenizer.char_to_int["<pad>"]
    eos_id = tokenizer.char_to_int["<eos>"]

    evaluation_loader = (
        data_stream.create_fixed_loader(
            num_samples=number_of_samples,
            max_value=max_value,
            batch_size=batch_size,
            seed=42,
        )
    )

    exact_correct_count = 0
    sequence_count = 0

    wrong_predictions = []

    for evaluation_batch in tqdm(
        evaluation_loader,
        desc="Evaluating multiplication",
    ):
        a_token_ids = evaluation_batch["a"]
        b_token_ids = evaluation_batch["b"]

        product_target_token_ids = (
            evaluation_batch["product"]
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
            product_logits.argmax(
                dim=-1
            )
        )

        product_predictions = (
            truncate_predictions_after_eos(
                predictions=product_predictions,
                eos_id=eos_id,
                pad_id=pad_id,
            )
        )

        correct_or_target_padding = (
            (product_predictions == product_targets)
            | (product_targets == pad_id)
        )

        correct_sequences = (
            correct_or_target_padding.all(
                dim=1
            )
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

                a_text = _decode_reversed_number(
                    token_ids=(
                        a_token_ids[sample_index]
                    ),
                    tokenizer=tokenizer,
                    eos_id=eos_id,
                    pad_id=pad_id,
                )

                b_text = _decode_reversed_number(
                    token_ids=(
                        b_token_ids[sample_index]
                    ),
                    tokenizer=tokenizer,
                    eos_id=eos_id,
                    pad_id=pad_id,
                )

                expected_product_text = (
                    _decode_reversed_number(
                        token_ids=(
                            product_target_token_ids[
                                sample_index
                            ]
                        ),
                        tokenizer=tokenizer,
                        eos_id=eos_id,
                        pad_id=pad_id,
                    )
                )

                predicted_product_text = (
                    _decode_reversed_number(
                        token_ids=(
                            product_predictions_cpu[
                                sample_index
                            ]
                        ),
                        tokenizer=tokenizer,
                        eos_id=eos_id,
                        pad_id=pad_id,
                    )
                )

                wrong_predictions.append(
                    (
                        a_text,
                        b_text,
                        expected_product_text,
                        predicted_product_text,
                    )
                )

                if len(wrong_predictions) == 10:
                    break

    accuracy = (
        exact_correct_count
        / sequence_count
    )

    print(
        f"Accuracy: {accuracy:.6f} "
        f"({exact_correct_count:,}/"
        f"{sequence_count:,})"
    )

    if wrong_predictions:
        print()
        print("Wrong predictions:")

        for (
            a_text,
            b_text,
            expected_product_text,
            predicted_product_text,
        ) in wrong_predictions:
            print()
            print(
                f"{a_text} × {b_text} "
                f"= {expected_product_text}"
            )

            print(
                "Prediction:",
                predicted_product_text,
            )
    else:
        print()
        print(
            "No wrong predictions."
        )

    model.train(
        model_was_training
    )