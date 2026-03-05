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

from ..data_collator import DataCollator
from .encoder import DNAEncoder, SecondEncoder


class COP(MyModelAbstract, nn.Module):
    def __init__(
        self,
        protein_feature: os.PathLike,
        protein_length: int,
        DNA_length: int,
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
        """COP arguments.

        Args:
            protein_feature: file contains info for mouse C2H2 zinc fingers.
            protein_length: maximally allowed protein length.
            DNA_length: maximally allowed DNA length.
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

        self.data_collator = DataCollator(protein_feature, protein_length, DNA_length)

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
            reduction="sum", pos_weight=torch.tensor(pos_weight)
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
            loss, loss_num = self.loss_fun(logit, label["bind"])
            return {
                "logit": logit,
                "loss": loss,
                "loss_num": loss_num,
            }
        return {"logit": logit}

    def loss_fun(self, logit: torch.Tensor, bind: torch.Tensor) -> tuple[float, float]:
        loss_num = logit.shape[0]
        # elastic_net only regularize weights (not bias) for linear and convolution layers
        loss = self.bce_with_logits_loss(
            input=logit, target=bind
        ) + loss_num * self.elastic_net(self)

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
                "DNA": [
                    self.data_collator.DNAs[example["DNAidx"]] for example in examples
                ],
                "protein": [example["protein"] for example in examples],
            }
        )

        return df

    @classmethod
    def hpo(cls, trial: optuna.Trial, cfg: jsonargparse.Namespace) -> None:
        cfg.model.init_args.protein_length = trial.suggest_int(
            "COP/COP/protein_length", 100, 300
        )
        cfg.model.init_args.DNA_length = trial.suggest_int(
            "COP/COP/DNA_length", 50, 150
        )
        cfg.model.init_args.dim_emb = trial.suggest_categorical(
            "COP/COP/dim_emb",
            choices=[32, 64, 128],
        )
        cfg.model.init_args.heads = trial.suggest_int("COP/COP/heads", 1, 3)
        cfg.model.init_args.dim_head = trial.suggest_categorical(
            "COP/COP/dim_head",
            choices=[16, 32, 64],
        )
        cfg.model.init_args.depth = trial.suggest_int("COP/COP/depth", 2, 6)
        cfg.model.init_args.dim_ffn = trial.suggest_categorical(
            "COP/COP/dim_ffn",
            choices=[64, 128, 256],
        )
        cfg.model.init_args.dropout = trial.suggest_float("COP/COP/dropout", 0.01, 0.1)
        cfg.model.init_args.reg_l1 = trial.suggest_float(
            "COP/COP/reg_l1", 0.000000001, 0.0000001
        )
        cfg.model.init_args.reg_l2 = trial.suggest_float(
            "COP/COP/reg_l2", 0.000000001, 0.0000001
        )
