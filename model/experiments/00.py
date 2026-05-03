"""
00: Dataset details
"""

from ml_digits import new_dataset_items, RNG


def main():
    dataset_items = new_dataset_items(random_state=RNG)

    print("dataset_items.all_list", len(dataset_items.all_list))
    print("dataset_items.train_validate_list", len(dataset_items.train_validate_list))
    print("dataset_items.train_list", len(dataset_items.train_list))
    print("dataset_items.validate_list", len(dataset_items.validate_list))
    print("dataset_items.test_list", len(dataset_items.test_list))


if __name__ == "__main__":
    main()
