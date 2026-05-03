"""
04: Logistic Regression Model with different optimizers
"""

import torch
import torch.nn as nn
import json
from ml_digits import (
    RESOLUTION,
    RNG,
    DigitDataset,
    device,
    experiments_path,
    new_dataset_items,
    runs_path,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

name = "04_logistic_regression_optimizers"

run_path = runs_path / name
experiment_path = experiments_path / name


class LogisticRegressionModel(nn.Module):
    def __init__(self, *, r: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),  # (1, R, R) -> RxR
            nn.Linear(r * r, 10),  # RxR -> 10 digit scores
        )

    def forward(self, x):
        return self.net(x)


def make_optimizer(
    optimizer_name: str,
    parameters,
    *,
    lr: float,
):
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr)

    if optimizer_name == "sgd_momentum":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9)

    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr)

    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr)

    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=lr)

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train(
    model: LogisticRegressionModel,
    *,
    dataset_train: DigitDataset,
    dataset_validate: DigitDataset,
    batch_size: int = 1024,
    epochs: int,
    lr: float,
    optimizer_name: str,
):
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    writer = SummaryWriter(run_path / optimizer_name)

    loader_train = DataLoader[DigitDataset](
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    loader_val = DataLoader[DigitDataset](
        dataset_validate,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(
        optimizer_name,
        model.parameters(),
        lr=lr,
    )

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in loader_train:
            inputs = inputs.to(device)
            targets = targets.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
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
                preds = outputs.argmax(dim=1)
                val_correct += preds.eq(targets).sum().item()
                val_total += targets.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        writer.add_scalar("Loss/train", train_loss, global_step=epoch)
        writer.add_scalar("Loss/val", val_loss, global_step=epoch)
        writer.add_scalar("Accuracy/train", train_acc, global_step=epoch)
        writer.add_scalar("Accuracy/val", val_acc, global_step=epoch)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"optimizer={optimizer_name:<12}  "
            f"lr={lr:.0e}  "
            f"Epoch {epoch:>{len(str(epochs))}}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}",
        )

    writer.close()

    return history


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
    experiment_path.mkdir(exist_ok=True)

    dataset_items = new_dataset_items(random_state=RNG)
    dataset_train = DigitDataset(items=dataset_items.train_list, r=RESOLUTION)
    dataset_validate = DigitDataset(items=dataset_items.validate_list, r=RESOLUTION)
    dataset_test = DigitDataset(items=dataset_items.test_list, r=RESOLUTION)

    lr = 1e-3

    optimizers = [
        "sgd",
        "sgd_momentum",
        "adam",
        "adamw",
        "rmsprop",
    ]

    results = {}

    for optimizer_name in optimizers:
        model = LogisticRegressionModel(r=RESOLUTION)

        train_result = train(
            model,
            dataset_train=dataset_train,
            dataset_validate=dataset_validate,
            epochs=50,
            lr=lr,
            optimizer_name=optimizer_name,
        )

        eval_result = evaluate(model, dataset_test)

        results[optimizer_name] = {
            "optimizer": optimizer_name,
            "lr": lr,
            "train": train_result,
            "eval": eval_result,
        }

        torch.save(
            model.state_dict(),
            experiment_path
            / f"model_epoch50-optimizer_{optimizer_name}-lr_{lr:.0e}.pt",
        )

    with open(experiment_path / "results.json", "w") as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    main()
