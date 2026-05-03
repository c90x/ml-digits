"""
05: Convolutional Models with different convolution configurations
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

name = "05_convolution_configs_adamw"

run_path = runs_path / name
experiment_path = experiments_path / name


class ConvModel(nn.Module):
    def __init__(
        self,
        *,
        r: int,
        conv_channels: list[int],
        kernel_size: int = 3,
        use_maxpool: bool = False,
        hidden_layers: list[int],
        dropout: float = 0.0,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        in_channels = 1

        for out_channels in conv_channels:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())

            if use_maxpool:
                layers.append(nn.MaxPool2d(kernel_size=2))

            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, r, r)
            feature_size = self.features(dummy).numel()

        classifier_layers: list[nn.Module] = []
        in_features = feature_size

        for hidden_size in hidden_layers:
            classifier_layers.append(nn.Linear(in_features, hidden_size))
            classifier_layers.append(nn.ReLU())

            if dropout > 0:
                classifier_layers.append(nn.Dropout(dropout))

            in_features = hidden_size

        classifier_layers.append(nn.Linear(in_features, 10))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def train(
    model: ConvModel,
    *,
    dataset_train: DigitDataset,
    dataset_validate: DigitDataset,
    config_name: str,
    batch_size: int = 1024,
    epochs: int,
    lr: float = 1e-3,
    patience: int = 5,
):
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    writer = SummaryWriter(run_path / config_name)

    loader_train = DataLoader[DigitDataset](
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    loader_val = DataLoader[DigitDataset](
        dataset_validate,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.to(device)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

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
            f"config={config_name:<18}  "
            f"Epoch {epoch:>{len(str(epochs))}}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping for {config_name} at epoch {epoch}. "
                f"Best epoch: {best_epoch}, best_val_loss={best_val_loss:.4f}",
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    writer.close()

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
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
    experiment_path.mkdir(exist_ok=True)

    dataset_items = new_dataset_items(random_state=RNG)
    dataset_train = DigitDataset(items=dataset_items.train_list, r=RESOLUTION)
    dataset_validate = DigitDataset(items=dataset_items.validate_list, r=RESOLUTION)
    dataset_test = DigitDataset(items=dataset_items.test_list, r=RESOLUTION)

    lr = 1e-3

    configs = [
        {
            "name": "conv8",
            "conv_channels": [8],
            "kernel_size": 3,
            "use_maxpool": False,
            "hidden_layers": [],
            "dropout": 0.0,
        },
        {
            "name": "conv16",
            "conv_channels": [16],
            "kernel_size": 3,
            "use_maxpool": False,
            "hidden_layers": [],
            "dropout": 0.0,
        },
        {
            "name": "conv16_pool",
            "conv_channels": [16],
            "kernel_size": 3,
            "use_maxpool": True,
            "hidden_layers": [],
            "dropout": 0.0,
        },
        {
            "name": "conv8_16",
            "conv_channels": [8, 16],
            "kernel_size": 3,
            "use_maxpool": False,
            "hidden_layers": [],
            "dropout": 0.0,
        },
        {
            "name": "conv8_16_pool",
            "conv_channels": [8, 16],
            "kernel_size": 3,
            "use_maxpool": True,
            "hidden_layers": [],
            "dropout": 0.0,
        },
        {
            "name": "conv16_32",
            "conv_channels": [16, 32],
            "kernel_size": 3,
            "use_maxpool": False,
            "hidden_layers": [],
            "dropout": 0.0,
        },
        {
            "name": "conv16_32_pool",
            "conv_channels": [16, 32],
            "kernel_size": 3,
            "use_maxpool": True,
            "hidden_layers": [],
            "dropout": 0.0,
        },
        {
            "name": "conv16_fc64",
            "conv_channels": [16],
            "kernel_size": 3,
            "use_maxpool": True,
            "hidden_layers": [64],
            "dropout": 0.0,
        },
        {
            "name": "conv16_32_fc128",
            "conv_channels": [16, 32],
            "kernel_size": 3,
            "use_maxpool": True,
            "hidden_layers": [128],
            "dropout": 0.0,
        },
        {
            "name": "conv16_32_fc128_drop",
            "conv_channels": [16, 32],
            "kernel_size": 3,
            "use_maxpool": True,
            "hidden_layers": [128],
            "dropout": 0.2,
        },
        {
            "name": "conv32_64_fc128",
            "conv_channels": [32, 64],
            "kernel_size": 3,
            "use_maxpool": True,
            "hidden_layers": [128],
            "dropout": 0.0,
        },
    ]

    results = {}

    for config in configs:
        config_name = config["name"]

        model = ConvModel(
            r=RESOLUTION,
            conv_channels=config["conv_channels"],
            kernel_size=config["kernel_size"],
            use_maxpool=config["use_maxpool"],
            hidden_layers=config["hidden_layers"],
            dropout=config["dropout"],
        )

        train_result = train(
            model,
            dataset_train=dataset_train,
            dataset_validate=dataset_validate,
            config_name=config_name,
            epochs=90,
            lr=lr,
            patience=5,
        )

        eval_result = evaluate(model, dataset_test)

        results[config_name] = {
            "config": config,
            "optimizer": "adamw",
            "lr": lr,
            "train": train_result,
            "eval": eval_result,
        }

        torch.save(
            model.state_dict(),
            experiment_path / f"model_{config_name}.pt",
        )

    with open(experiment_path / "results.json", "w") as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    main()
