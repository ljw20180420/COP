import os
from typing import Optional

import jsonargparse
import numpy as np
import optuna
import pandas as pd
import torch
import xgboost as xgb
from common_ai.generator import MyGenerator
from common_ai.initializer import MyInitializer
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
        eta: float,
        num_boost_round: int,
    ) -> None:
        """XGBoost arguments.

        Args:
            protein_feature: file contains info for mouse C2H2 zinc fingers.
            protein_length: maximally allowed protein length.
            dna_length: maximally allowed DNA length.
            eta: Shrink of step size after each round.
            num_boost_round: Number of trees generated in single epochs.
        """
        super().__init__()
        self.eta = eta
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

    def my_train_epoch(
        self,
        my_train: MyTrain,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
        my_generator: MyGenerator,
        my_optimizer: MyOptimizer,
        my_profiler: MyProfiler,
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

        evals_result = {}
        self.booster = xgb.train(
            params={
                "device": self.device,
                "eta": self.eta,
                "objective": "binary:logistic",
                "seed": my_generator.seed,
            },
            dtrain=self.Xy_train,
            num_boost_round=self.num_boost_round,
            evals=[(self.Xy_train, "train")],
            evals_result=evals_result,
            xgb_model=self.booster,
        )

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
