import os

import datasets
from common_ai.dataset import MyDatasetAbstract
from common_ai.utils import SeqTokenizer, split_train_valid_test


def get_dataset(
    data_file: os.PathLike,
    name: str,
    test_ratio: float,
    validation_ratio: float,
    seed: int,
) -> datasets.Dataset:
    """Parameters of dataset.

    Args:
        data_file: The csv file of DNA data.
        name: name of the file.
        test_ratio: Proportion for test samples.
        validation_ratio: Proportion for validation samples.
        seed: random seed.
    """
    ds = datasets.load_dataset(
        path="csv",
        data_files=os.fspath(data_file),
    )
    ds = split_train_valid_test(
        ds=ds, validation_ratio=validation_ratio, test_ratio=test_ratio, seed=seed
    )

    return ds
