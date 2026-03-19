import os

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from common_ai.generator import MyGenerator
from common_ai.metric import MyMetricAbstract
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
        num_boost_round: int,
    ) -> None:
        """XGBoost arguments.

        Args:
            protein_feature: file contains info for mouse C2H2 zinc fingers.
            protein_length: maximally allowed protein length.
            dna_length: maximally allowed DNA length.
            subsample: subsample ratio of the training instances.
            colsample_bynode: the subsample ratio of columns for each node (split).
            eta: Shrink of step size after each round.
            num_boost_round: Number of trees generated in single epochs.
        """
        self.subsample = subsample
        self.colsample_bynode = colsample_bynode
        self.eta = eta
        self.num_boost_round = num_boost_round

        self.data_collator = DataCollator(protein_feature, protein_length, dna_length)

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
            ),
            iteration_range=(0, self.best_iteration + 1),
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
        return {
            "booster": torch.frombuffer(self.booster.save_raw(), dtype=torch.uint8),
            "best_iteration": self.best_iteration,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.booster = xgb.Booster(
            model_file=bytearray(state_dict["booster"].numpy().tobytes())
        )
        self.best_iteration = self.state_dict["best_iteration"]

    def _metric(
        self,
        predt: np.ndarray,
        dtrain: xgb.DMatrix,
        metrics: dict[str, MyMetricAbstract],
    ) -> tuple[str, dict[str, float]]:
        metric_loss_dict = {}
        for metric_name, metric_fun in metrics.items():
            metric_fun.step(
                df=pd.DataFrame({"proba": predt}),
                examples={},
                batch={"label": {"bind": torch.from_numpy(dtrain.get_label())}},
            )
            metric_loss_dict[metric_name] = metric_fun.epoch()

        return "custom_metric", metric_loss_dict

    def _train_booster(self, my_generator: MyGenerator, metrics: dict) -> dict:
        evals_result = {}
        self.booster = xgb.train(
            params={
                "booster": "gbtree",
                "subsample": self.subsample,
                "colsample_bynode": self.colsample_bynode,
                "eta": self.eta,
                "device": self.device,
                "objective": "binary:logistic",
                "seed": my_generator.seed,
            },
            dtrain=self.Xy_train,
            num_boost_round=self.num_boost_round,
            evals=[(self.Xy_train, "train"), (self.Xy_eval, "eval")],
            evals_result=evals_result,
            custom_metric=lambda predt, dmatrix, metrics=metrics: self._metric(
                predt, dmatrix, metrics
            ),
        )

        self.best_iteration = np.argmin(
            [cm["AccuracyMetric"] for cm in evals_result["eval"]["custom_metric"]]
        ).item()

        return evals_result

    def my_train_epoch(
        self,
        my_train: MyTrain,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
        my_generator: MyGenerator,
        my_optimizer: MyOptimizer,
        my_profiler: MyProfiler,
        metrics: dict,
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

        self.evals_result = self._train_booster(my_generator, metrics)

        return (
            self.evals_result["train"]["logloss"][self.best_iteration].item()
            * self.Xy_train.num_row(),
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
            self.evals_result["eval"]["logloss"][self.best_iteration].item()
            * self.Xy_eval.num_row()
        )

        metric_loss_dict = self.evals_result["eval"]["custom_metric"][
            self.best_iteration
        ]

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
    ):
        """RandomForest arguments.

        Args:
            protein_feature: file contains info for mouse C2H2 zinc fingers.
            protein_length: maximally allowed protein length.
            dna_length: maximally allowed DNA length.
            subsample: subsample ratio of the training instances.
            colsample_bynode: the subsample ratio of columns for each node (split).
            num_parallel_tree: the size of the forest being trained.
        """
        self.subsample = subsample
        self.colsample_bynode = colsample_bynode
        self.num_parallel_tree = num_parallel_tree

        self.data_collator = DataCollator(protein_feature, protein_length, dna_length)

    def _train_booster(self, my_generator: MyGenerator, metrics: dict) -> dict:
        evals_result = {}
        self.booster = xgb.train(
            params={
                "booster": "gbtree",
                "subsample": self.subsample,
                "colsample_bynode": self.colsample_bynode,
                "num_parallel_tree": self.num_parallel_tree,
                "eta": 1,
                "device": self.device,
                "objective": "binary:logistic",
                "seed": my_generator.seed,
            },
            dtrain=self.Xy_train,
            num_boost_round=1,
            evals=[(self.Xy_train, "train")],
            evals_result=evals_result,
            custom_metric=lambda predt, dmatrix, metrics=metrics: self._metric(
                predt, dmatrix, metrics
            ),
        )

        self.best_iteration = 0

        return evals_result


class DecisionTree(RandomForest):
    def __init__(
        self,
        protein_feature: os.PathLike,
        protein_length: int,
        dna_length: int,
        subsample: float,
        colsample_bynode: float,
    ):
        """DecisionTree arguments.

        Args:
            protein_feature: file contains info for mouse C2H2 zinc fingers.
            protein_length: maximally allowed protein length.
            dna_length: maximally allowed DNA length.
            subsample: subsample ratio of the training instances.
            colsample_bynode: the subsample ratio of columns for each node (split).
        """
        super().__init__(
            protein_feature,
            protein_length,
            dna_length,
            subsample,
            colsample_bynode,
            num_parallel_tree=1,
        )
