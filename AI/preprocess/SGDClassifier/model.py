import os
import pickle
from typing import Literal, Optional

import jsonargparse
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
from scipy import special
from sklearn import linear_model, preprocessing
from tqdm import tqdm

from ..data_collator import DataCollator


class SGDClassifier(MyModelAbstract):
    def __init__(
        self,
        protein_feature: os.PathLike,
        protein_length: int,
        dna_length: int,
        loss: Literal[
            "hinge",
            "log_loss",
            "modified_huber",
            "squared_hinge",
            "perceptron",
            "squared_error",
            "huber",
            "epsilon_insensitive",
            "squared_epsilon_insensitive",
        ],
        penalty: Optional[Literal["l2", "l1", "elasticnet"]],
        alpha: float,
        l1_ratio: float,
    ) -> None:
        """SGDClassifier arguments.

        Args:
            protein_feature: file contains info for mouse C2H2 zinc fingers.
            protein_length: maximally allowed protein length.
            dna_length: maximally allowed DNA length.
            loss: the loos function to be used.
            penalty: regularization type among l2, l1, l2/l1 (elasticnet), None.
            alpha: constant that multiplies the penalty term, controlling regularization strength.
            l1_ratio: ratio of l1 regularization, only relevant for elasticnet.
        """
        super().__init__()

        self.data_collator = DataCollator(protein_feature, protein_length, dna_length)

        self.sgd_classifier = linear_model.SGDClassifier(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            n_jobs=-1,
        )

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
        probas = special.expit(
            self.sgd_classifier.decision_function(X=X_value),
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
            "sgd_classifier": torch.frombuffer(
                bytearray(pickle.dumps(self.sgd_classifier)), dtype=torch.uint8
            )
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.sgd_classifier = pickle.loads(
            state_dict["sgd_classifier"].numpy().tobytes()
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
        train_loss, train_loss_num = 0.0, 0.0
        for examples in tqdm(train_dataloader):
            batch = self.data_collator(
                examples, output_label=True, my_generator=my_generator
            )
            X_value, y_value = self._get_feature(
                input=batch["input"], label=batch["label"]
            )

            self.sgd_classifier.partial_fit(X=X_value, y=y_value)

            score = self.sgd_classifier.decision_function(X=X_value)
            train_loss += -(
                (np.ma.log(special.expit(score)).filled(-1000) * y_value).sum().item()
            )
            train_loss += -(
                (np.ma.log(special.expit(-score)).filled(-1000) * (1 - y_value))
                .sum()
                .item()
            )
            train_loss_num += X_value.shape[0]

        return train_loss, train_loss_num, float("nan")

    def my_eval_epoch(
        self,
        my_train: MyTrain,
        eval_dataloader: torch.utils.data.DataLoader,
        my_generator: MyGenerator,
        metrics: dict,
    ) -> tuple:
        eval_loss, eval_loss_num = 0.0, 0.0
        for examples in tqdm(eval_dataloader):
            batch = self.data_collator(
                examples, output_label=True, my_generator=my_generator
            )
            X_value, y_value = self._get_feature(
                input=batch["input"], label=batch["label"]
            )

            score = self.sgd_classifier.decision_function(X=X_value)
            eval_loss += -(
                (np.ma.log(special.expit(score)).filled(-1000) * y_value).sum().item()
            )
            eval_loss += -(
                (np.ma.log(special.expit(-score)).filled(-1000) * (1 - y_value))
                .sum()
                .item()
            )
            eval_loss_num += X_value.shape[0]
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

        return eval_loss, eval_loss_num, metric_loss_dict

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
