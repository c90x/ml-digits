"""
10: Final CNN model
"""

import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn
from ml_digits import (
    RESOLUTION,
    RNG,
    DigitDataset,
    DigitTransformedDataset2,
    device,
    experiments_path,
    new_dataset_items,
    runs_path,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

name = "10_final_model"

run_path = runs_path / name
experiment_path = experiments_path / name


class FinalDigitModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # 1x28x28 -> 32x28x28
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32x28x28 -> 32x14x14
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 32x14x14 -> 64x14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64x14x14 -> 64x7x7
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 64x7x7 -> 128x7x7
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            #
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def train(
    model: FinalDigitModel,
    *,
    dataset_train: DigitTransformedDataset2,
    dataset_validate: DigitTransformedDataset2,
    batch_size: int = 1024,
    max_epochs: int = 400,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    patience: int = 25,
    max_train_seconds: int = 6 * 60 * 60,
):
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    writer = SummaryWriter(run_path / "final")

    loader_train = DataLoader[DigitTransformedDataset2](
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    loader_val = DataLoader[DigitTransformedDataset2](
        dataset_validate,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
    )

    model.to(device)

    started_at = time.monotonic()
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    stopped_reason = "max_epochs"

    for epoch in range(1, max_epochs + 1):
        if time.monotonic() - started_at >= max_train_seconds:
            stopped_reason = "time_limit"
            break

        dataset_train.reroll()
        dataset_validate.reroll()

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in loader_train:
            inputs = inputs.to(device)
            targets = targets.to(device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()

        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in loader_val:
                inputs = inputs.to(device)
                targets = targets.to(device, dtype=torch.long)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_running_loss += loss.item() * inputs.size(0)
                val_correct += outputs.argmax(dim=1).eq(targets).sum().item()
                val_total += targets.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, global_step=epoch)
        writer.add_scalar("Loss/val", val_loss, global_step=epoch)
        writer.add_scalar("Accuracy/train", train_acc, global_step=epoch)
        writer.add_scalar("Accuracy/val", val_acc, global_step=epoch)
        writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], global_step=epoch)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        improved = val_acc > best_val_acc or (
            val_acc == best_val_acc and val_loss < best_val_loss
        )

        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                best_state,
                experiment_path / "model_best.pt",
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:>{len(str(max_epochs))}}/{max_epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"best_val_acc={best_val_acc:.4f}",
        )

        if epochs_without_improvement >= patience:
            stopped_reason = "early_stopping"
            break

    model.load_state_dict(best_state)
    writer.close()

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc,
        "stopped_reason": stopped_reason,
        "train_seconds": time.monotonic() - started_at,
    }


def evaluate(
    model: nn.Module,
    dataset: DigitDataset,
    *,
    batch_size: int = 1024,
) -> dict[str, float]:
    model.to(device)
    model.eval()

    loader = DataLoader[DigitDataset](
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device, dtype=torch.long)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total += targets.size(0)

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


def main():
    rng = np.random.default_rng(RNG)

    experiment_path.mkdir(exist_ok=True)
    run_path.mkdir(exist_ok=True)

    dataset_items = new_dataset_items(random_state=RNG)

    dataset_train = DigitTransformedDataset2(
        items=dataset_items.train_list,
        r=RESOLUTION,
        rng=rng,
    )
    dataset_validate = DigitTransformedDataset2(
        items=dataset_items.validate_list,
        r=RESOLUTION,
        rng=rng,
    )
    dataset_test = DigitDataset(
        items=dataset_items.test_list,
        r=RESOLUTION,
    )

    config = {
        "model": "FinalDigitModel",
        "batch_size": 1024,
        "max_epochs": 400,
        "lr": 3e-3,
        "weight_decay": 1e-4,
        "patience": 25,
        "max_train_seconds": 6 * 60 * 60,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "train_dataset": "DigitTransformedDataset",
        "validate_dataset": "DigitTransformedDataset",
        "test_dataset": "DigitDataset",
    }

    model = FinalDigitModel()

    train_result = train(
        model,
        dataset_train=dataset_train,
        dataset_validate=dataset_validate,
        batch_size=config["batch_size"],
        max_epochs=config["max_epochs"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        patience=config["patience"],
        max_train_seconds=config["max_train_seconds"],
    )

    eval_result = evaluate(
        model,
        dataset_test,
        batch_size=config["batch_size"],
    )

    results = {
        "config": config,
        "train": train_result,
        "eval": eval_result,
    }

    torch.save(
        model.state_dict(),
        experiment_path / "model_final.pt",
    )

    with open(experiment_path / "results.json", "w") as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    main()
