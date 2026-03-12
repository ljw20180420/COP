import os

import numpy as np
from sklearn import naive_bayes

from ..data_collator import DataCollator
from ..model import SKBase


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
