import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from IPython.display import clear_output, display


class Trainer:
    def __init__(
        self,
        train_loss_history: list[float] | None = None,
        val_loss_history: list[float] | None = None,
        train_accuracy_history: list[float] | None = None,
        val_accuracy_history: list[float] | None = None,
        checkpoint_dir: str | os.PathLike = "/teamspace/studios/this_studio/SAIR-competition-modular-math/checkpoints",
        device: str | torch.device | None = None,
    ):
        self.train_loss_history = train_loss_history if train_loss_history is not None else []
        self.val_loss_history = val_loss_history if val_loss_history is not None else []
        self.train_accuracy_history = train_accuracy_history if train_accuracy_history is not None else []
        self.val_accuracy_history = val_accuracy_history if val_accuracy_history is not None else []
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    @staticmethod
    def dataset_tensor_to_loader(
        dataset_tensor: torch.Tensor,
        train_percent: float = 0.8,
        val_percent: float = 0.1,
        batch_size: int = 64,
        rand_seed: int = 42,
    ):
        total_rows = dataset_tensor.size(0)

        if total_rows < 3:
            raise ValueError("dataset_tensor must contain at least 3 rows to create train, val, and test splits.")

        if train_percent < 0 or val_percent < 0:
            raise ValueError("train_percent and val_percent must be non-negative.")

        if train_percent + val_percent >= 1:
            raise ValueError("train_percent + val_percent must be less than 1 so test split is non-empty.")

        train_size = int(train_percent * total_rows)
        val_size = int(val_percent * total_rows)
        test_size = total_rows - train_size - val_size

        if train_size <= 0:
            raise ValueError("train split is empty. Increase train_percent or use a larger dataset.")

        if val_size <= 0:
            raise ValueError("validation split is empty. Increase val_percent or use a larger dataset.")

        if test_size <= 0:
            raise ValueError("test split is empty. Make train_percent + val_percent smaller.")

        train_subset, val_subset, test_subset = random_split(
            dataset_tensor,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(rand_seed),
        )

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    
    def add_loss_and_accuracy(self, train_loss: float, val_loss: float, train_accuracy: float, val_accuracy: float):
        self.train_loss_history.append(float(train_loss))
        self.val_loss_history.append(float(val_loss))
        self.train_accuracy_history.append(float(train_accuracy))
        self.val_accuracy_history.append(float(val_accuracy))

    
    def plot_live_loss(self, current_epoch: int, total_target_epochs: int):
        if len(self.train_loss_history) == 0 or len(self.val_loss_history) == 0:
            raise ValueError("No loss history to plot yet.")
    
        if len(self.train_loss_history) != len(self.val_loss_history):
            raise ValueError("Loss histories must have the same length.")
    
        if len(self.train_accuracy_history) != len(self.val_accuracy_history):
            raise ValueError("Accuracy histories must have the same length.")
    
        clear_output(wait=True)
    
        fig, ax1 = plt.subplots(figsize=(14, 8))
    
        epochs_range = list(range(1, len(self.train_loss_history) + 1))
    
        # Loss axis
        ax1.plot(
            epochs_range,
            self.train_loss_history,
            label="Training Loss",
            linewidth=2,
            marker=".",
            markersize=4
        )
    
        ax1.plot(
            epochs_range,
            self.val_loss_history,
            label="Validation Loss",
            linewidth=2,
            marker=".",
            markersize=4
        )
    
        ax1.set_xlabel("Epochs", fontsize=12)
        ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
    
        # Accuracy axis
        ax2 = ax1.twinx()
    
        ax2.plot(
            epochs_range,
            self.train_accuracy_history,
            label="Training Accuracy",
            linestyle="--",
            linewidth=2,
            marker="."
        )
    
        ax2.plot(
            epochs_range,
            self.val_accuracy_history,
            label="Validation Accuracy",
            linestyle="--",
            linewidth=2,
            marker="."
        )
    
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_ylim(0, 1)
    
        ax1.set_title(
            "Live Training Loss and Accuracy",
            fontsize=14,
            fontweight="bold"
        )
    
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            fontsize=11
        )
    
        ax1.grid(True, linestyle="--", alpha=0.6)
    
        plt.tight_layout()
        display(fig)
        plt.close(fig)
    
        print(
            f"Latest Stats -> Epoch {current_epoch:02d}/{total_target_epochs:02d} | "
            f"Train Loss: {self.train_loss_history[-1]:.4f} | "
            f"Val Loss: {self.val_loss_history[-1]:.4f} | "
            f"Train Acc: {self.train_accuracy_history[-1]:.4f} | "
            f"Val Acc: {self.val_accuracy_history[-1]:.4f}"
        )

    
    def save_checkpoint(self, checkpoint_name: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if not checkpoint_name.endswith(".pt"):
            checkpoint_name += ".pt"

        checkpoint_path = self.checkpoint_dir / checkpoint_name

        checkpoint_state = {
            'model_state_dict': model.state_dict(),
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            'epochs_completed': len(self.train_loss_history),
        }

        if optimizer is not None:
            checkpoint_state['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint_state, checkpoint_path)
        print(f"Checkpoint manually saved as: {checkpoint_path}")

    
    def load_checkpoint(self, checkpoint_name: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None):
        if not checkpoint_name.endswith(".pt"):
            checkpoint_name += ".pt"

        checkpoint_path = self.checkpoint_dir / checkpoint_name

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint file found at {checkpoint_path}")

        print(f"📂 Loading checkpoint: {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.train_loss_history = checkpoint.get('train_loss_history', [])
        self.val_loss_history = checkpoint.get('val_loss_history', [])

        print(f"Successfully restored state from {checkpoint_name} (Epochs completed: {checkpoint.get('epochs_completed', len(self.train_loss_history))})")
        return checkpoint

