import numpy as np
import pandas as pd
import torch
from torch import nn


class MyBCELoss:
    def __init__(
        self,
        pos_weight: float,
    ) -> None:
        """MyBCELoss arguments.

        Args:
            pos_weight: Weight for positive samples (https://www.tensorflow.org/tutorials/structured_data/imbalanced_data).
        """
        self.bce_loss = nn.BCELoss(reduction=None)
        self.pos_weight = pos_weight

    def __call__(
        self,
        df: pd.DataFrame,
        binds: list,
    ) -> tuple:
        probas = torch.from_numpy(df["proba"].to_numpy())
        binds = torch.tensor(binds)
        loss = self.bce_loss(
            probas,
            binds,
        )
        loss[binds > 0.0] = loss[binds > 0.0] * self.pos_weight
        loss = loss.cpu().numpy()
        loss_num = np.arange(len(loss))

        return loss, loss_num
