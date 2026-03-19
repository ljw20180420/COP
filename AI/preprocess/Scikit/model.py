import os
import pickle
from abc import abstractmethod
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
from common_ai.generator import MyGenerator
from common_ai.optimizer import MyOptimizer
from common_ai.profiler import MyProfiler
from common_ai.train import MyTrain
from scipy import special
from sklearn import linear_model, naive_bayes
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from ..data_collator import DataCollator
from ..model import MLBase


class SKBase(MLBase):
    def __init__(self):
        self.dna_onehot_encoder = OneHotEncoder().fit([[i] for i in range(7)])
        self.protein_onehot_encoder = OneHotEncoder().fit([[i] for i in range(26)])
        self.second_onehot_encoder = OneHotEncoder().fit([[i] for i in range(12)])

    def eval_output(
        self, examples: list[dict], batch: dict, my_generator: MyGenerator
    ) -> pd.DataFrame:
        X_value = self._get_feature(
            input=batch["input"],
            label=None,
        )
        batch_size = X_value.shape[0]
        probas = self.predict_proba(X_value)
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
            "classifier": torch.frombuffer(
                bytearray(pickle.dumps(self.classifier)), dtype=torch.uint8
            )
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.classifier = pickle.loads(state_dict["classifier"].numpy().tobytes())

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
        train_loss, train_loss_num = 0.0, 0.0
        for examples in tqdm(train_dataloader):
            batch = self.data_collator(
                examples, output_label=True, my_generator=my_generator
            )
            X_value, y_value = self._get_feature(
                input=batch["input"], label=batch["label"]
            )

            self.classifier.partial_fit(X=X_value, y=y_value, classes=[0, 1])

            log_probas = self.predict_log_proba(X_value)
            train_loss += -(
                log_probas[np.arange(len(y_value)), y_value.astype(int)].sum().item()
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

            log_probas = self.predict_log_proba(X_value)
            eval_loss += -(
                log_probas[np.arange(len(y_value)), y_value.astype(int)].sum().item()
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
        print(metric_loss_dict)

        return eval_loss, eval_loss_num, metric_loss_dict

    def _get_feature(
        self,
        input: dict,
        label: Optional[dict],
    ) -> tuple[np.ndarray]:
        dna_ids = input["dna_id"].cpu().numpy()
        protein_ids = input["protein_id"].cpu().numpy()
        second_ids = input["second_id"].cpu().numpy()
        X_value = np.concatenate(
            [
                self.dna_onehot_encoder.transform(dna_ids[:, [c]]).toarray()
                for c in range(dna_ids.shape[1])
            ]
            + [
                self.protein_onehot_encoder.transform(protein_ids[:, [c]]).toarray()
                for c in range(protein_ids.shape[1])
            ]
            + [
                self.second_onehot_encoder.transform(second_ids[:, [c]]).toarray()
                for c in range(second_ids.shape[1])
            ],
            axis=1,
        )

        if label is not None:
            y_value = label["bind"].cpu().numpy()
            return X_value, y_value

        return X_value

    @abstractmethod
    def predict_proba(self, X_value: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_log_proba(self, X_value: np.ndarray) -> np.ndarray:
        pass


class GaussianNB(SKBase):
    def __init__(
        self,
        protein_feature: os.PathLike,
        protein_length: int,
        dna_length: int,
    ) -> None:
        """GaussianNB arguments.

        Args:
            protein_feature: file contains info for mouse C2H2 zinc fingers.
            protein_length: maximally allowed protein length.
            dna_length: maximally allowed DNA length.
        """
        super().__init__()

        self.data_collator = DataCollator(protein_feature, protein_length, dna_length)

        self.classifier = naive_bayes.GaussianNB()

    def predict_proba(self, X_value: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(X_value)[:, 1]

    def predict_log_proba(self, X_value: np.ndarray) -> np.ndarray:
        return self.classifier.predict_log_proba(X_value)


class SKLinearBase(SKBase):
    def predict_proba(self, X_value: np.ndarray) -> np.ndarray:
        return special.expit(self.classifier.decision_function(X=X_value))

    def predict_log_proba(self, X_value: np.ndarray) -> np.ndarray:
        score = self.classifier.decision_function(X=X_value)
        return np.stack(
            [
                np.ma.log(special.expit(-score)).filled(-1000),
                np.ma.log(special.expit(score)).filled(-1000),
            ],
            axis=1,
        )


class SGDClassifier(SKLinearBase):
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
        random_state: int,
    ) -> None:
        """SGDClassifier arguments.

        Args:
            protein_feature: file contains info for mouse C2H2 zinc fingers.
            protein_length: maximally allowed protein length.
            dna_length: maximally allowed DNA length.
            loss: the loss function to be used.
            penalty: regularization type among l2, l1, l2/l1 (elasticnet), None.
            alpha: constant that multiplies the penalty term, controlling regularization strength.
            l1_ratio: ratio of l1 regularization, only relevant for elasticnet.
            random_state: use for shuffling data.
        """
        super().__init__()

        self.data_collator = DataCollator(protein_feature, protein_length, dna_length)

        self.classifier = linear_model.SGDClassifier(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            n_jobs=-1,
            random_state=random_state,
        )


class Perceptron(SKLinearBase):
    def __init__(
        self,
        protein_feature: os.PathLike,
        protein_length: int,
        dna_length: int,
        penalty: Optional[Literal["l2", "l1", "elasticnet"]],
        alpha: float,
        l1_ratio: float,
        random_state: int,
    ) -> None:
        """Perceptron arguments.

        Args:
            protein_feature: file contains info for mouse C2H2 zinc fingers.
            protein_length: maximally allowed protein length.
            dna_length: maximally allowed DNA length.
            penalty: regularization type among l2, l1, l2/l1 (elasticnet), None.
            alpha: constant that multiplies the penalty term, controlling regularization strength.
            l1_ratio: ratio of l1 regularization, only relevant for elasticnet.
            random_state: use for shuffling data.
        """
        super().__init__()

        self.data_collator = DataCollator(protein_feature, protein_length, dna_length)

        self.classifier = linear_model.Perceptron(
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            n_jobs=-1,
            random_state=random_state,
        )


class PassiveAggressiveClassifier(SKLinearBase):
    def __init__(
        self,
        protein_feature: os.PathLike,
        protein_length: int,
        dna_length: int,
        loss: Literal["hinge", "squared_hinge"],
        random_state: int,
    ) -> None:
        """PassiveAggressiveClassifier arguments.

        Args:
            protein_feature: file contains info for mouse C2H2 zinc fingers.
            protein_length: maximally allowed protein length.
            dna_length: maximally allowed DNA length.
            loss: the loss function to be used.
            random_state: use for shuffling data.
        """
        super().__init__()

        self.data_collator = DataCollator(protein_feature, protein_length, dna_length)

        self.classifier = linear_model.PassiveAggressiveClassifier(
            loss=loss,
            n_jobs=-1,
            random_state=random_state,
        )
