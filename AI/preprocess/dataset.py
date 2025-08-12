import os
import datasets


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
    ds = train_validation_test_split(
        ds=ds, validation_ratio=validation_ratio, test_ratio=test_ratio, seed=seed
    )

    return ds


def train_validation_test_split(
    ds: datasets.Dataset, validation_ratio: float, test_ratio: float, seed: int
) -> datasets.Dataset:
    ds = ds["train"].train_test_split(
        test_size=test_ratio + validation_ratio, seed=seed
    )
    ds2 = ds.pop("test").train_test_split(
        test_size=test_ratio / (test_ratio + validation_ratio),
        seed=seed,
    )
    ds["validation"] = ds2.pop("train")
    ds["test"] = ds2.pop("test")
    return ds
