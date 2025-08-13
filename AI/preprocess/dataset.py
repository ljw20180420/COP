import os
import datasets
from common_ai.utils import split_train_valid_test


def get_dataset(
    DNA_data: os.PathLike,
    test_ratio: float,
    validation_ratio: float,
    seed: int,
) -> datasets.Dataset:
    """Parameters of dataset.

    Args:
        DNA_data: The csv file of DNA data.
        test_ratio: Proportion for test samples.
        validation_ratio: Proportion for validation samples.
        seed: random seed.
    """
    ds = datasets.load_dataset(
        path="csv",
        data_files=os.fspath(DNA_data),
    )
    ds = split_train_valid_test(
        ds=ds, validation_ratio=validation_ratio, test_ratio=test_ratio, seed=seed
    )

    return ds
