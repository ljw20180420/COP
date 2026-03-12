import os
from typing import Literal, Optional

from sklearn import linear_model

from ..data_collator import DataCollator
from ..model import SKLinearBase


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
