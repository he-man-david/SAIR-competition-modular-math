import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from IPython.display import display
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class Trainer:
    def __init__(
        self,
        train_loss_history: list[float] | None = None,
        val_loss_history: list[float] | None = None,
        train_accuracy_history: list[float] | None = None,
        val_accuracy_history: list[float] | None = None,
        metric_step_history: list[int] | None = None,
        checkpoint_dir: str | os.PathLike = (
            "/teamspace/studios/this_studio/"
            "SAIR-competition-modular-math/checkpoints"
        ),
        device: str | torch.device | None = None,
    ):
        self.train_loss_history = (
            train_loss_history if train_loss_history is not None else []
        )
        self.val_loss_history = (
            val_loss_history if val_loss_history is not None else []
        )
        self.train_accuracy_history = (
            train_accuracy_history
            if train_accuracy_history is not None
            else []
        )
        self.val_accuracy_history = (
            val_accuracy_history
            if val_accuracy_history is not None
            else []
        )
        self.metric_step_history = (
            metric_step_history if metric_step_history is not None else []
        )

        self.live_plot_display = None
        self.global_step = 0
        self.checkpoint_dir = Path(checkpoint_dir)

        self.device = (
            torch.device(device)
            if device is not None
            else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

    def add_loss_and_accuracy(
        self,
        step: int,
        train_loss: float,
        val_loss: float,
        train_accuracy: float,
        val_accuracy: float,
    ):
        self.metric_step_history.append(int(step))
        self.train_loss_history.append(float(train_loss))
        self.val_loss_history.append(float(val_loss))
        self.train_accuracy_history.append(float(train_accuracy))
        self.val_accuracy_history.append(float(val_accuracy))

    def plot_live_loss(self, target_step: int) -> None:
        if not self.metric_step_history:
            return
    
        fig, loss_axis = plt.subplots(figsize=(14, 8))
        accuracy_axis = loss_axis.twinx()
    
        loss_axis.plot(
            self.metric_step_history,
            self.train_loss_history,
            label="Training loss",
            linewidth=2,
            marker=".",
            markersize=4,
        )
    
        loss_axis.plot(
            self.metric_step_history,
            self.val_loss_history,
            label="Validation loss",
            linewidth=2,
            marker=".",
            markersize=4,
        )
    
        accuracy_axis.plot(
            self.metric_step_history,
            self.train_accuracy_history,
            label="Training accuracy",
            linewidth=2,
            linestyle="--",
        )
    
        accuracy_axis.plot(
            self.metric_step_history,
            self.val_accuracy_history,
            label="Validation accuracy",
            linewidth=2,
            linestyle="--",
        )
    
        loss_axis.set_title("Training and Validation Metrics")
        loss_axis.set_xlabel("Training step")
        loss_axis.set_ylabel("Cross-entropy loss")
        accuracy_axis.set_ylabel("Exact-match accuracy")
    
        loss_axis.set_xlim(
            left=0,
            right=max(target_step, self.metric_step_history[-1]),
        )
    
        loss_axis.xaxis.set_major_locator(
            MaxNLocator(nbins=20, integer=True)
        )
    
        loss_lines, loss_labels = loss_axis.get_legend_handles_labels()
        accuracy_lines, accuracy_labels = (
            accuracy_axis.get_legend_handles_labels()
        )
    
        loss_axis.legend(
            loss_lines + accuracy_lines,
            loss_labels + accuracy_labels,
            loc="best",
        )
    
        fig.tight_layout()
    
        if self.live_plot_display is None:
            self.live_plot_display = display(
                fig,
                display_id=True,
            )
        else:
            self.live_plot_display.update(fig)
    
        plt.close(fig)

    @staticmethod
    def plot_test_loss(
        test_loss_history: list[float],
        test_accuracy_history: list[float],
    ):
        batch_numbers = list(
            range(10, len(test_loss_history) + 1, 10)
        )
        plotted_losses = test_loss_history[9::10]
        plotted_accuracies = test_accuracy_history[9::10]

        if not plotted_losses:
            return

        clear_output(wait=True)

        fig, ax1 = plt.subplots(figsize=(14, 8))

        ax1.plot(
            batch_numbers,
            plotted_losses,
            label="Test Loss",
            linewidth=2,
            marker=".",
        )

        ax1.set_xlabel("Batch", fontsize=12)
        ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
        ax1.set_ylim(
            0.0,
            max(max(plotted_losses), 1.0),
        )
        ax1.xaxis.set_major_locator(
            MaxNLocator(nbins=20, integer=True)
        )

        ax2 = ax1.twinx()

        ax2.plot(
            batch_numbers,
            plotted_accuracies,
            label="Test Accuracy",
            color="orange",
            linestyle="--",
            linewidth=2,
            marker=".",
        )

        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_ylim(0, 1)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
        )

        ax1.set_title(
            "Test Loss and Accuracy",
            fontsize=14,
            fontweight="bold",
        )

        ax1.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        display(fig)
        plt.close(fig)

    def evaluate(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        loss_fct,
        vocab_size: int,
        max_seq_len: int,
        pad_id: int,
        description: str = "Validation",
    ) -> tuple[float, float]:
        if len(data_loader) == 0:
            raise ValueError("data_loader must contain at least one batch.")

        total_loss = 0.0
        total_accuracy = 0.0

        model.eval()

        with torch.no_grad():
            for batch in tqdm(
                data_loader,
                desc=description,
                leave=False,
            ):
                batch = batch.to(self.device)
                a, b, tgt = batch.unbind(dim=1)

                logits = model(a, b, tgt)

                loss = loss_fct(
                    logits.reshape(-1, vocab_size),
                    tgt.reshape(-1),
                )

                predictions = model.predict(
                    a,
                    b,
                    max_new_tokens=max_seq_len,
                )

                accuracy = self.compute_accuracy_from_predictions(
                    predictions,
                    tgt,
                    pad_id,
                )

                total_loss += loss.item()
                total_accuracy += accuracy.item()

        average_loss = total_loss / len(data_loader)
        average_accuracy = total_accuracy / len(data_loader)

        return average_loss, average_accuracy

    def test(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        loss_fct,
        vocab_size: int,
        max_seq_len: int,
        pad_id: int,
    ) -> tuple[float, float]:
        if len(test_loader) == 0:
            raise ValueError(
                "test_loader must contain at least one batch."
            )

        test_loss_history: list[float] = []
        test_accuracy_history: list[float] = []

        model.eval()

        with torch.no_grad():
            for batch_index, batch in enumerate(
                tqdm(test_loader, desc="Testing"),
                start=1,
            ):
                batch = batch.to(self.device)
                a, b, tgt = batch.unbind(dim=1)

                logits = model(a, b, tgt)

                test_loss = loss_fct(
                    logits.reshape(-1, vocab_size),
                    tgt.reshape(-1),
                )

                predictions = model.predict(
                    a,
                    b,
                    max_new_tokens=max_seq_len,
                )

                test_accuracy = (
                    self.compute_accuracy_from_predictions(
                        predictions,
                        tgt,
                        pad_id,
                    )
                )

                test_loss_history.append(test_loss.item())
                test_accuracy_history.append(
                    test_accuracy.item()
                )

                if batch_index % 10 == 0:
                    self.plot_test_loss(
                        test_loss_history,
                        test_accuracy_history,
                    )

        average_test_loss = (
            sum(test_loss_history) / len(test_loss_history)
        )
        average_test_accuracy = (
            sum(test_accuracy_history)
            / len(test_accuracy_history)
        )

        print(
            f"Test Loss: {average_test_loss:.4f} | "
            f"Test Accuracy: {average_test_accuracy:.4f}"
        )

        return average_test_loss, average_test_accuracy

    @staticmethod
    def compute_accuracy(
        logits: torch.Tensor,
        targets: torch.Tensor,
        pad_id: int,
    ) -> torch.Tensor:
        predictions = logits.argmax(dim=-1)

        corrects = (
            (predictions == targets)
            | (targets == pad_id)
        )

        exact_matches = corrects.all(dim=-1)

        return exact_matches.float().mean()

    @staticmethod
    def compute_accuracy_from_predictions(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pad_id: int,
    ) -> torch.Tensor:
        if predictions.shape != targets.shape:
            raise ValueError(
                "predictions and targets must have the same shape. "
                f"Received {predictions.shape} and {targets.shape}."
            )

        corrects = (
            (predictions == targets)
            | (targets == pad_id)
        )

        exact_matches = corrects.all(dim=-1)

        return exact_matches.float().mean()

    def save_checkpoint(
        self,
        checkpoint_name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler=None,
        data_stream=None,
    ):
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not checkpoint_name.endswith(".pt"):
            checkpoint_name += ".pt"

        checkpoint_path = (
            self.checkpoint_dir / checkpoint_name
        )

        checkpoint_state = {
            "model_state_dict": model.state_dict(),
            "global_step": self.global_step,
            "metric_step_history": self.metric_step_history,
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
            "train_accuracy_history": (
                self.train_accuracy_history
            ),
            "val_accuracy_history": (
                self.val_accuracy_history
            ),
        }

        if optimizer is not None:
            checkpoint_state["optimizer_state_dict"] = (
                optimizer.state_dict()
            )

        if scheduler is not None:
            checkpoint_state["scheduler_state_dict"] = (
                scheduler.state_dict()
            )

        if data_stream is not None:
            checkpoint_state["data_stream_state_dict"] = (
                data_stream.state_dict()
            )

        torch.save(
            checkpoint_state,
            checkpoint_path,
        )

        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(
        self,
        checkpoint_name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler=None,
        data_stream=None,
    ):
        if not checkpoint_name.endswith(".pt"):
            checkpoint_name += ".pt"

        checkpoint_path = (
            self.checkpoint_dir / checkpoint_name
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if (
            optimizer is not None
            and "optimizer_state_dict" in checkpoint
        ):
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if (
            scheduler is not None
            and "scheduler_state_dict" in checkpoint
        ):
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        if (
            data_stream is not None
            and "data_stream_state_dict" in checkpoint
        ):
            data_stream.load_state_dict(
                checkpoint["data_stream_state_dict"]
            )

        self.global_step = checkpoint.get(
            "global_step",
            0,
        )

        self.metric_step_history = checkpoint.get(
            "metric_step_history",
            [],
        )

        self.train_loss_history = checkpoint.get(
            "train_loss_history",
            [],
        )

        self.val_loss_history = checkpoint.get(
            "val_loss_history",
            [],
        )

        self.train_accuracy_history = checkpoint.get(
            "train_accuracy_history",
            [],
        )

        self.val_accuracy_history = checkpoint.get(
            "val_accuracy_history",
            [],
        )

        print(
            f"Checkpoint loaded: {checkpoint_path} | "
            f"Global step: {self.global_step:,}"
        )