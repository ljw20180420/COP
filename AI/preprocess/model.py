import pickle
from abc import abstractmethod
from typing import Optional

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
from tqdm import tqdm


class MLBase(MyModelAbstract):
    def my_initialize_model(
        self, my_initializer: MyInitializer, my_generator: MyGenerator
    ) -> None:
        pass

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


class SKBase(MLBase):
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
    ) -> tuple:
        train_loss, train_loss_num = 0.0, 0.0
        for examples in tqdm(train_dataloader):
            batch = self.data_collator(
                examples, output_label=True, my_generator=my_generator
            )
            X_value, y_value = self._get_feature(
                input=batch["input"], label=batch["label"]
            )

            self.classifier.partial_fit(X=X_value, y=y_value)

            log_probas = self.predict_log_proba(X_value)
            train_loss += -(log_probas[np.arange(len(y_value)), y_value].sum().item())
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

            score = self.classifier.decision_function(X=X_value)
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

    @abstractmethod
    def predict_proba(self, X_value: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_log_proba(self, X_value: np.ndarray) -> np.ndarray:
        pass


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
