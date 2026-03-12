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
from common_ai.model import MyModelAbstract
from common_ai.optimizer import MyOptimizer
from common_ai.profiler import MyProfiler
from common_ai.train import MyTrain
from tqdm import tqdm

from ..data_collator import DataCollator


class LightGBM(MyModelAbstract):
    def __init__(
        self,
        protein_feature: os.PathLike,
        protein_length: int,
        dna_length: int,
        eta: float,
        max_depth: int,
        subsample: float,
        reg_lambda: float,
        num_boost_round: int,
    ) -> None:
        """LigtGBM arguments.

        Args:
            protein_feature: file contains info for mouse C2H2 zinc fingers.
            protein_length: maximally allowed protein length.
            dna_length: maximally allowed DNA length.
            eta: Shrink of step size after each round.
            max_depth: maximum depth of a tree.
            subsample: subsample ratio of the training instances.
            reg_lambda: L2 regularization term on weights.
            num_boost_round: Number of trees generated in single epochs.
        """
        super().__init__()

        self.eta = eta
        self.max_depth = max_depth
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.num_boost_round = num_boost_round

        self.data_collator = DataCollator(protein_feature, protein_length, dna_length)

        self.booster = None

    def my_initialize_model(
        self, my_initializer: MyInitializer, my_generator: MyGenerator
    ) -> None:
        pass

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
            )

        eval_result = {}
        self.booster = lgb.train(
            params={
                "device": self.device,
                "eta": self.eta,
                "max_depth": self.max_depth,
                "subsample": self.subsample,
                "reg_lambda": self.reg_lambda,
                "objective": "binary",
                "seed": my_generator.seed,
            },
            train_set=self.train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=[self.train_data],
            valid_names=["train"],
            init_model=self.booster,
            callbacks=[lgb.record_evaluation(eval_result)],
        )

        return (
            eval_result["train"]["logloss"][0] * self.train_data.num_data(),
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
                categorical_feature=list(range(X_eval.shape[-1])),
            )

        eval_loss = (
            self.booster.eval(data=self.eval_data, name="eval")[2]
            * self.eval_data.num_data()
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

        return eval_loss, self.eval_data.num_data(), metric_loss_dict

    def _get_feature(
        self,
        input: dict,
        label: Optional[dict],
    ) -> tuple[np.ndarray]:
        X_value = np.concatenate(
            (
                input["dna_id"].cpu().numpy(),
                input["protein_id"].cpu().numpy(),
                input["second_id"].cpu().numpy(),
            ),
            axis=1,
        )

        if label is not None:
            y_value = label["bind"].cpu().numpy()
            return X_value, y_value

        return X_value

    @classmethod
    def hpo(cls, trial: optuna.Trial, cfg: jsonargparse.Namespace) -> None:
        pass
