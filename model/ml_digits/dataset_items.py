from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import dataset_path
from .env import RATIO_TEST, RATIO_VAL


@dataclass(kw_only=True)
class DatasetItems:
    random_state: int | None

    all: dict[int, list[Path]]
    test: dict[int, list[Path]]
    train: dict[int, list[Path]]
    validate: dict[int, list[Path]]
    train_validate: dict[int, list[Path]]

    all_list: list[tuple[Path, int]]
    test_list: list[tuple[Path, int]]
    train_list: list[tuple[Path, int]]
    validate_list: list[tuple[Path, int]]
    train_validate_list: list[tuple[Path, int]]


def new_dataset_items(*, random_state: int | None = None) -> DatasetItems:
    all = dataset_items_get_all()
    train_validate, test = dataset_items_split(
        all, split=RATIO_TEST, random_state=random_state
    )
    train, validate = dataset_items_split(
        train_validate, split=RATIO_VAL, random_state=random_state
    )

    return DatasetItems(
        random_state=random_state,
        all=all,
        test=test,
        train=train,
        validate=validate,
        train_validate=train_validate,
        all_list=dataset_items_to_list(all),
        test_list=dataset_items_to_list(test),
        train_list=dataset_items_to_list(train),
        validate_list=dataset_items_to_list(validate),
        train_validate_list=dataset_items_to_list(train_validate),
    )


def dataset_items_get_all() -> dict[int, list[Path]]:
    global dataset_path
    return {
        digit: list(dataset_path.glob(f"HWD-V1-Standard/{digit}/*.png"))
        for digit in range(10)
    }


def dataset_items_split(
    dataset: dict[int, list[Path]], split: float, random_state: int | None = None
) -> tuple[dict[int, list[Path]], dict[int, list[Path]]]:
    """
    Return two split dicts from DATASET_ALL
    """
    train, test = {}, {}

    rng = np.random.default_rng(random_state)

    for digit, items in dataset.items():
        shuffled_items = list(items)
        rng.shuffle(shuffled_items)

        split_idx = int(len(shuffled_items) * split)
        train[digit] = shuffled_items[:split_idx]
        test[digit] = shuffled_items[split_idx:]

    return train, test


def dataset_items_to_list(dataset_dict):
    return [(item, digit) for digit, items in dataset_dict.items() for item in items]
