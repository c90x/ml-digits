"""
03: Model with different layer ReLU
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

name = "03_layers_relu"

run_path = runs_path / name
experiment_path = experiments_path / name


class LayeredModel(nn.Module):
    def __init__(self, *, r: int, hidden_sizes: list[int]):
        super().__init__()

        layers: list[nn.Module] = [nn.Flatten()]
        dims = [r * r] + hidden_sizes + [10]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train(
    model: LayeredModel,
    *,
    dataset_train: DigitDataset,
    dataset_validate: DigitDataset,
    batch_size: int = 1024,
    epochs: int,
    layer_name: str,
):
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    writer = SummaryWriter(run_path / layer_name)

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
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

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
            f"{layer_name}  "
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

    layer_configs = {
        "l128": [128],
        "l6": [6],
        "l6_6": [6, 6],
        "l6_6_6": [6, 6, 6],
        "l6_6_6_6": [6, 6, 6, 6],
        "l6_6_6_6_6": [6, 6, 6, 6, 6],
        "l8_6_3_6": [8, 6, 3, 6],
    }

    results = {}

    for layer_name, layer_sizes in layer_configs.items():
        model = LayeredModel(r=RESOLUTION, hidden_sizes=layer_sizes)

        train_result = train(
            model,
            dataset_train=dataset_train,
            dataset_validate=dataset_validate,
            epochs=50,
            layer_name=layer_name,
        )

        eval_result = evaluate(model, dataset_test)

        results[layer_name] = {"train": train_result, "eval": eval_result}

        torch.save(
            model.state_dict(),
            experiment_path / f"model_epoch50-{layer_name}.pt",
        )

    with open(experiment_path / "results.json", "w") as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    main()
