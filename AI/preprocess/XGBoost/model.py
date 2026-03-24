import os

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from common_ai.generator import MyGenerator
from common_ai.optimizer import MyOptimizer
from common_ai.profiler import MyProfiler
from common_ai.train import MyTrain
from tqdm import tqdm

from ..data_collator import DataCollator
from ..model import MLBase


class XGBoost(MLBase):
    def __init__(
        self,
        protein_feature: os.PathLike,
        protein_length: int,
        dna_length: int,
        subsample: float,
        colsample_bynode: float,
        eta: float,
        max_depth: int,
        num_boost_round: int,
    ) -> None:
        """XGBoost arguments.

        Args:
            protein_feature: file contains info for mouse C2H2 zinc fingers.
            protein_length: maximally allowed protein length.
            dna_length: maximally allowed DNA length.
            subsample: subsample ratio of the training instances.
            colsample_bynode: subsample ratio of columns for each node (split).
            eta: Shrink of step size after each round.
            max_depth: maximum depth of a tree.
            num_boost_round: Number of trees generated in single epochs.
        """
        self.subsample = subsample
        self.colsample_bynode = colsample_bynode
        self.eta = eta
        self.max_depth = max_depth
        self.num_boost_round = num_boost_round

        self.data_collator = DataCollator(protein_feature, protein_length, dna_length)

        self.booster = None

    def eval_output(
        self, examples: list[dict], batch: dict, my_generator: MyGenerator
    ) -> pd.DataFrame:
        X_value = self._get_feature(
            input=batch["input"],
            label=None,
        )
        batch_size = X_value.shape[0]
        probas = self.booster.predict(
            data=xgb.DMatrix(
                data=X_value,
                feature_types=["c"] * X_value.shape[-1],
                enable_categorical=True,
            )
        )
        df = pd.DataFrame(
            {
                "sample_idx": np.arange(batch_size),
                "proba": probas,
                "DNA": [example["DNA"] for example in examples],
                "protein": [example["protein"] for example in examples],
            }
        )

        return df

    def state_dict(self) -> dict:
        return {"booster": torch.frombuffer(self.booster.save_raw(), dtype=torch.uint8)}

    def load_state_dict(self, state_dict: dict) -> None:
        self.booster = xgb.Booster(
            model_file=bytearray(state_dict["booster"].numpy().tobytes())
        )

    def _train_booster(self, my_generator: MyGenerator) -> dict:
        evals_result = {}
        self.booster = xgb.train(
            params={
                "subsample": self.subsample,
                "colsample_bynode": self.colsample_bynode,
                "eta": self.eta,
                "max_depth": self.max_depth,
                "booster": "gbtree",
                "objective": "binary:logistic",
                "seed": my_generator.seed,
                "device": self.device,
            },
            dtrain=self.Xy_train,
            num_boost_round=self.num_boost_round,
            evals=[(self.Xy_train, "train"), (self.Xy_eval, "eval")],
            evals_result=evals_result,
            xgb_model=self.booster,
        )

        return evals_result

    def my_train_epoch(
        self,
        my_train: MyTrain,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
        my_generator: MyGenerator,
        my_optimizer: MyOptimizer,
        my_profiler: MyProfiler,
        metrcis: dict,
    ) -> tuple:
        if not hasattr(self, "Xy_train"):
            X_train, y_train = [], []
            for examples in tqdm(train_dataloader):
                batch = self.data_collator(
                    examples, output_label=True, my_generator=my_generator
                )
                X_value, y_value = self._get_feature(
                    input=batch["input"], label=batch["label"]
                )
                X_train.append(X_value)
                y_train.append(y_value)

            X_train = np.concatenate(X_train)
            y_train = np.concatenate(y_train)

            self.Xy_train = xgb.QuantileDMatrix(
                data=X_train,
                label=y_train,
                feature_types=["c"] * X_train.shape[-1],
                enable_categorical=True,
            )

        if not hasattr(self, "Xy_eval"):
            X_eval, y_eval = [], []
            for examples in tqdm(eval_dataloader):
                batch = self.data_collator(
                    examples, output_label=True, my_generator=my_generator
                )
                X_value, y_value = self._get_feature(
                    input=batch["input"], label=batch["label"]
                )
                X_eval.append(X_value)
                y_eval.append(y_value)

            X_eval = np.concatenate(X_eval)
            y_eval = np.concatenate(y_eval)

            # Use QuantileDMatrix for evaluation and test is not recommanded because it needs train data as ref, which defeats the purpose of saving memory. See https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.QuantileDMatrix and https://www.kaggle.com/code/cdeotte/xgboost-using-original-data-cv-0-976?scriptVersionId=257750413&cellId=24
            self.Xy_eval = xgb.DMatrix(
                data=X_eval,
                label=y_eval,
                feature_types=["c"] * X_eval.shape[-1],
                enable_categorical=True,
            )

        evals_result = self._train_booster(my_generator)

        return (
            np.mean(evals_result["train"]["logloss"]).item() * self.Xy_train.num_row(),
            self.Xy_train.num_row(),
            float("nan"),
        )

    def my_eval_epoch(
        self,
        my_train: MyTrain,
        eval_dataloader: torch.utils.data.DataLoader,
        my_generator: MyGenerator,
        metrics: dict,
    ) -> tuple:
        eval_loss = (
            float(self.booster.eval(self.Xy_eval).split(":")[1])
            * self.Xy_eval.num_row()
        )
        for examples in tqdm(eval_dataloader):
            batch = self.data_collator(
                examples, output_label=True, my_generator=my_generator
            )
            df = self.eval_output(examples, batch, my_generator)
            for metric_name, metric_fun in metrics.items():
                metric_fun.step(
                    df=df,
                    examples=examples,
                    batch=batch,
                )

        metric_loss_dict = {}
        for metric_name, metric_fun in metrics.items():
            metric_loss_dict[metric_name] = metric_fun.epoch()

        return eval_loss, self.Xy_eval.num_row(), metric_loss_dict


class RandomForest(XGBoost):
    # https://xgboost.readthedocs.io/en/stable/tutorials/rf.html

    def __init__(
        self,
        protein_feature: os.PathLike,
        protein_length: int,
        dna_length: int,
        subsample: float,
        colsample_bynode: float,
        num_parallel_tree: int,
        max_depth: int,
    ):
        """RandomForest arguments.

        Args:
            protein_feature: file contains info for mouse C2H2 zinc fingers.
            protein_length: maximally allowed protein length.
            dna_length: maximally allowed DNA length.
            subsample: subsample ratio of the training instances.
            colsample_bynode: subsample ratio of columns for each node (split).
            num_parallel_tree: the size of the forest being trained.
            max_depth: maximum depth of a tree.
        """
        self.subsample = subsample
        self.colsample_bynode = colsample_bynode
        self.num_parallel_tree = num_parallel_tree
        self.max_depth = max_depth

        self.data_collator = DataCollator(protein_feature, protein_length, dna_length)

        self.booster = None

    def _train_booster(self, my_generator: MyGenerator) -> dict:
        evals_result = {}
        self.booster = xgb.train(
            params={
                "subsample": self.subsample,
                "colsample_bynode": self.colsample_bynode,
                "num_parallel_tree": self.num_parallel_tree,
                "eta": 1,
                "max_depth": self.max_depth,
                "booster": "gbtree",
                "objective": "binary:logistic",
                "seed": my_generator.seed,
                "device": self.device,
            },
            dtrain=self.Xy_train,
            num_boost_round=1,
            evals=[(self.Xy_train, "train")],
            evals_result=evals_result,
            xgb_model=self.booster,
        )

        return evals_result


class DecisionTree(RandomForest):
    def __init__(
        self,
        protein_feature: os.PathLike,
        protein_length: int,
        dna_length: int,
        max_depth: int,
    ):
        """DecisionTree arguments.

        Args:
            protein_feature: file contains info for mouse C2H2 zinc fingers.
            protein_length: maximally allowed protein length.
            dna_length: maximally allowed DNA length.
            max_depth: maximum depth of a tree.
        """
        super().__init__(
            protein_feature,
            protein_length,
            dna_length,
            subsample=1.0,
            colsample_bynode=1.0,
            num_parallel_tree=1,
            max_depth=max_depth,
        )
