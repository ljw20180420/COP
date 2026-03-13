import os
from typing import Optional

import jsonargparse
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
from common_ai.generator import MyGenerator
from common_ai.initializer import MyInitializer
from common_ai.optimizer import MyOptimizer
from common_ai.profiler import MyProfiler
from common_ai.train import MyTrain
from tqdm import tqdm

from ..data_collator import DataCollator
from ..model import MLBase


class LightGBM(MLBase):
    def __init__(
        self,
        protein_feature: os.PathLike,
        protein_length: int,
        dna_length: int,
        eta: float,
        num_boost_round: int,
    ) -> None:
        """LigtGBM arguments.

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
        probas = self.booster.predict(data=X_value)
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
            "booster": torch.frombuffer(
                bytearray(self.booster.model_to_string().encode()),
                dtype=torch.uint8,
            )
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.booster = lgb.Booster(
            model_str=state_dict["booster"].numpy().tobytes().decode()
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
        if not hasattr(self, "train_data"):
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

            self.train_data = lgb.Dataset(
                data=X_train,
                label=y_train,
                categorical_feature=list(range(X_train.shape[-1])),
                free_raw_data=False,
            )

        if not hasattr(self, "eval_data"):
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

            self.eval_data = lgb.Dataset(
                data=X_eval,
                label=y_eval,
                reference=self.train_data,
                categorical_feature=list(range(X_eval.shape[-1])),
                free_raw_data=False,
            )

        eval_result = {}
        self.booster = lgb.train(
            params={
                "device": self.device,
                "eta": self.eta,
                "objective": "binary",
                "seed": my_generator.seed,
            },
            train_set=self.train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=[self.train_data, self.eval_data],
            valid_names=["train", "eval"],
            init_model=self.booster,
            keep_training_booster=True,
            callbacks=[lgb.record_evaluation(eval_result)],
        )

        return (
            np.mean(eval_result["train"]["binary_logloss"]).item()
            * self.train_data.num_data(),
            self.train_data.num_data(),
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
            self.booster.eval(data=self.eval_data, name="eval")[0][2].item()
            * self.eval_data.num_data()
        )
        for examples in eval_dataloader:
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

        return eval_loss, self.eval_data.num_data(), metric_loss_dict
