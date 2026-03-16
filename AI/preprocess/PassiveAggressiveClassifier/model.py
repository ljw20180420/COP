import os
from typing import Literal

from sklearn import linear_model

from ..data_collator import DataCollator
from ..model import SKLinearBase


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
        self.data_collator = DataCollator(protein_feature, protein_length, dna_length)

        self.classifier = linear_model.PassiveAggressiveClassifier(
            loss=loss,
            n_jobs=-1,
            random_state=random_state,
        )
