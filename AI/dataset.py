import os

import datasets
import jsonargparse
import optuna
from common_ai.dataset import MyDatasetAbstract


class MyDataset(MyDatasetAbstract):
    def __init__(
        self,
        data_dir: os.PathLike,
        name: str,
    ) -> None:
        """MyDataset arguments.

        Args:
            data_dir: The directory containing csv files.
            name: name of the dataset.
        """
        super().__init__(name=name)
        self.data_dir = data_dir

    def __call__(self) -> datasets.Dataset:
        ds = datasets.load_dataset(path="csv", name=self.name, data_dir=self.data_dir)

        return ds

    @classmethod
    def hpo(cls, trial: optuna.Trial, cfg: jsonargparse.Namespace) -> None:
        pass
