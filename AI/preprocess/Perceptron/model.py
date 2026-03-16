import os
from typing import Literal, Optional

from sklearn import linear_model

from ..data_collator import DataCollator
from ..model import SKLinearBase


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
        self.data_collator = DataCollator(protein_feature, protein_length, dna_length)

        self.classifier = linear_model.Perceptron(
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            n_jobs=-1,
            random_state=random_state,
        )
