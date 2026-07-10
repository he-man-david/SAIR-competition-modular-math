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
        checkpoint_dir: str | os.PathLike = "/teamspace/studios/this_studio/SAIR-competition-modular-math/checkpoints",
        device: str | torch.device | None = None,
    ):
        self.train_loss_history = train_loss_history if train_loss_history is not None else []
        self.val_loss_history = val_loss_history if val_loss_history is not None else []
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

    def add_loss(self, train_loss: float, val_loss: float):
        self.train_loss_history.append(float(train_loss))
        self.val_loss_history.append(float(val_loss))

    def plot_live_loss(self, current_epoch: int, total_target_epochs: int):
        if len(self.train_loss_history) == 0 or len(self.val_loss_history) == 0:
            raise ValueError("No loss history to plot yet. Add losses first with trainer.add_loss(train_loss, val_loss).")

        if len(self.train_loss_history) != len(self.val_loss_history):
            raise ValueError("train_loss_history and val_loss_history must have the same length.")

        clear_output(wait=True)

        fig, ax = plt.subplots(figsize=(14, 8))
        epochs_range = list(range(1, len(self.train_loss_history) + 1))

        ax.plot(
            epochs_range,
            self.train_loss_history,
            label='Training Loss',
            color='royalblue',
            linewidth=2,
            marker='.',
            markersize=4
        )
        ax.plot(
            epochs_range,
            self.val_loss_history,
            label='Validation Loss',
            color='darkorange',
            linewidth=2,
            marker='.',
            markersize=4
        )

        ax.set_title('Live Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Cross-Entropy Loss', fontsize=12)

        num_epochs = len(epochs_range)
        if num_epochs <= 20:
            step = 1
        elif num_epochs <= 50:
            step = 5
        elif num_epochs <= 200:
            step = 10
        else:
            step = 25

        custom_ticks = [1] + list(range(step, num_epochs + 1, step))
        if num_epochs not in custom_ticks:
            custom_ticks.append(num_epochs)

        ax.set_xticks(custom_ticks)

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=11)

        plt.tight_layout()
        display(fig)
        plt.close(fig)

        print(f"Latest Stats -> Epoch {current_epoch:02d}/{total_target_epochs:02d} | "
              f"Train Loss: {self.train_loss_history[-1]:.4f} | Val Loss: {self.val_loss_history[-1]:.4f}")

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