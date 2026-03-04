import os
from typing import Optional

import jsonargparse
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from common_ai.generator import MyGenerator
from common_ai.initializer import MyInitializer
from common_ai.model import MyModelAbstract
from common_ai.protein_bert import ProteinBert
from common_ai.utils import ElasticNet
from einops.layers.torch import EinMix, Rearrange

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from torch.nn import BCEWithLogitsLoss

from .data_collator import DataCollator
from .encoder import DNAEncoder, SecondEncoder


class COPFormer(MyModelAbstract, nn.Module):
    def __init__(
        self,
        protein_feature: os.PathLike,
        protein_length: int,
        DNA_length: int,
        minimal_unbind_summit_distance: int,
        select_worst_loss_ratio: float,
        dim_emb: int,
        heads: int,
        dim_head: int,
        depth: int,
        dim_ffn: int,
        dropout: float,
        pos_weight: float,
        reg_l1: float,
        reg_l2: float,
        use_hyena: bool,
        hyena_order: int,
        hyena_filter_order: int,
    ) -> None:
        """COPFormer arguments.

        Args:
            protein_feature: file contains info for mouse C2H2 zinc fingers.
            protein_length: maximally allowed protein length.
            DNA_length: maximally allowed DNA length.
            minimal_unbind_summit_distance: if no protein peak found within this distance of a locus, then this locus is considered not occupied by this protein.
            select_worst_loss_ratio: the ratio among negative samples to choose the worst (highest) loss according to the most recent loss records.
            dim_emb: embedding dimension size.
            heads: number of attention heads.
            dim_head: dimension size per head.
            depth: number of attention layer.
            dim_ffn: dimension size of hidden layer in feedforward network.
            dropout: dropout rate.
            pos_weight: weight of positive sample.
            reg_l1: l1 regularization coefficient.
            reg_l2: l2 regularization coefficient.
            use_hyena: whether to use hyena in encoders.
            hyena_order: the order of hyena.
            hyena_filter_order: the order of hyena filter.
        """
        super().__init__()

        self.data_collator = DataCollator(
            protein_feature,
            protein_length,
            DNA_length,
            minimal_unbind_summit_distance,
            select_worst_loss_ratio,
        )

        self.second_encoder = SecondEncoder(
            12,
            protein_length,
            dim_emb,
            dim_head,
            heads,
            depth,
            dim_ffn,
            dropout,
            use_hyena,
            hyena_order,
            hyena_filter_order,
        )

        self.dna_encoder = DNAEncoder(
            7,
            DNA_length,
            dim_emb,
            dim_head,
            heads,
            depth,
            dim_ffn,
            dropout,
            use_hyena,
            hyena_order,
            hyena_filter_order,
        )

        self.classifier = nn.Sequential(
            EinMix(
                "b d -> b d_0",
                weight_shape="d d_0",
                bias_shape="d_0",
                d=dim_emb,
                d_0=dim_emb,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            EinMix(
                "b d -> b o",
                weight_shape="d o",
                bias_shape="o",
                d=dim_emb,
                o=1,
            ),
            Rearrange("b 1 -> b"),
        )

        self.protein_bert_head = nn.Sequential(
            EinMix(
                "b s d_b -> b s d",
                weight_shape="d_b d",
                bias_shape="d",
                d_b=128,
                d=dim_emb,
            ),
            nn.Dropout(dropout),
            nn.RMSNorm(dim_emb),
        )

        self.bce_with_logits_loss = BCEWithLogitsLoss(
            reduction=None, pos_weight=torch.tensor(pos_weight)
        )

        self.elastic_net = ElasticNet(reg_l1, reg_l2)

        self.protein_bert = ProteinBert(
            num_tokens=26,
            dim_token=128,
            dim_global=512,
            depth=6,
            narrow_conv_kernel=9,
            wide_conv_kernel=9,
            wide_conv_dilation=5,
            attn_heads=4,
            attn_dim_head=64,
        )

    def forward(
        self,
        input: dict,
        label: Optional[dict],
        my_generator: Optional[MyGenerator],
    ) -> dict:
        # encode protein
        protein_embs = self.protein_bert(input["protein_id"])
        protein_embs = self.protein_bert_head(protein_embs)

        # encode secondary structure
        second_embs, second_mask = self.second_encoder(input["second_id"])

        # encode DNA
        dna_embs = self.dna_encoder(
            input["dna_id"], protein_embs, second_embs, second_mask
        )

        # classify
        logit = self.classifier(dna_embs)

        if label is not None:
            loss, loss_num = self.loss_fun(logit, label)
            return {
                "logit": logit,
                "loss": loss,
                "loss_num": loss_num,
            }
        return {"logit": logit}

    def loss_fun(self, logit: torch.Tensor, label: dict) -> tuple[float, float]:
        loss_num = logit.shape[0]
        losses = self.bce_with_logits_loss(input=logit, target=label["bind"])
        for rn, actual_protein, loss in zip(
            label["rn"], label["actual_protein"], losses
        ):
            self.data_collator.recent_losses[(rn, actual_protein)] = loss
        # elastic_net only regularize weights (not bias) for linear and convolution layers
        loss = losses.sum() + loss_num * self.elastic_net(self)

        return loss, loss_num

    def my_initialize_model(
        self, my_initializer: MyInitializer, my_generator: MyGenerator
    ) -> None:
        my_initializer(self, my_generator)
        self.protein_bert.load_pretrain_weights(
            "AI/preprocess/epoch_92400_sample_23500000.pkl"
        )

    def eval_output(self, examples: list[dict], batch: dict) -> pd.DataFrame:
        batch_size = len(examples)
        result = self(input=batch["input"], label=None, my_generator=None)
        probas = F.sigmoid(result["logit"]).cpu().numpy()
        df = pd.DataFrame(
            {
                "sample_idx": np.arange(batch_size),
                "proba": probas,
                "protein": batch["label"]["actual_protein"],
                "DNA": [example["DNA"] for example in examples],
            }
        )

        return df

    def state_dict(self) -> dict:
        return {
            "pytorch_state_dict": super().state_dict(),
            "recent_loss": self.data_collator.recent_losses,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict["pytorch_state_dict"])
        self.data_collator.recent_losses = state_dict["recent_loss"]

    @classmethod
    def hpo(cls, trial: optuna.Trial, cfg: jsonargparse.Namespace) -> None:
        pass
