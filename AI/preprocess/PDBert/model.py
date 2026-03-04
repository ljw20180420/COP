import os
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from common_ai.generator import MyGenerator
from common_ai.initializer import MyInitializer
from common_ai.protein_bert import ProteinBert, ProteinBertLayerCross
from einops.layers.torch import Rearrange
from torch import nn

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum

from ..COPFormer.data_collator import DataCollator


class PDBertModel(nn.Module):
    model_type = "PDBert"

    def __init__(
        self,
        protein_num_tokens: int,
        DNA_num_tokens: int,
        dim_token: int,
        dim_global: int,
        protein_depth: int,
        DNA_depth: int,
        cross_depth: int,
        narrow_conv_kernel: int,
        wide_conv_kernel: int,
        wide_conv_dilation: int,
        attn_heads: int,
        attn_dim_head: int,
        pos_weight: float,
        dropout: float,
        protein_data: os.PathLike,
        protein_length: int,
        DNA_length: int,
        minimal_unbind_summit_distance: int,
        select_worst_loss_ratio: float,
        protein_bert_pretrained_weights: os.PathLike,
    ) -> None:
        """ProteinBert arguments.

        Args:
            protein_num_tokens: Number of protein tokens.
            DNA_num_tokens: Number of DNA tokens.
            dim_token: Token-wise embedding dimension.
            dim_global: Global embedding dimension.
            protein_depth: Number of protein bert layers.
            DNA_depth: Number of DNA bert layers.
            cross_depth: Number of cross bert layers.
            narrow_conv_kernel: Kernal size of narrow convolution.
            wide_conv_kernel: Kernel size of wide convolution.
            wide_conv_dilation: Dilation of wide convolution.
            attn_heads: Number of cross attension heads.
            attn_dim_head: Dimension of each cross attension head.
            pos_weight: Weight for positive samples (https://www.tensorflow.org/tutorials/structured_data/imbalanced_data).
            dropout: Dropout probability of classifier head.
            protein_data: File contains proteins used for training.
            protein_length: Protein bert use pad instead of mask. So always pad protein by p to protein_length. If the protein is longer than protein length, it is sheared.
            DNA_length: As protein length.
            minimal_unbind_summit_distance: Minimal distance between summit such that the protein is considered not bind to the target peak.
            select_worst_loss_ratio: Select this worst (maximal) loss negative samples according to this ratio. The left negative samples are selected randomly.
            protein_bert_pretrained_weights: Pretrained weights of protein bert mode.
        """
        super().__init__()
        self.protein_bert_pretrained_weights = protein_bert_pretrained_weights

        self.data_collator = DataCollator(
            protein_data=protein_data,
            protein_length=protein_length,
            DNA_length=DNA_length,
            minimal_unbind_summit_distance=minimal_unbind_summit_distance,
            select_worst_loss_ratio=select_worst_loss_ratio,
        )

        self.protein_bert = ProteinBert(
            num_tokens=protein_num_tokens,
            dim_token=dim_token,
            dim_global=dim_global,
            depth=protein_depth,
            narrow_conv_kernel=narrow_conv_kernel,
            wide_conv_kernel=wide_conv_kernel,
            wide_conv_dilation=wide_conv_dilation,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
        )

        self.DNA_bert = ProteinBert(
            num_tokens=DNA_num_tokens,
            dim_token=dim_token,
            dim_global=dim_global,
            depth=DNA_depth,
            narrow_conv_kernel=narrow_conv_kernel,
            wide_conv_kernel=wide_conv_kernel,
            wide_conv_dilation=wide_conv_dilation,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
        )

        self.layer_crosses = nn.ModuleList(
            [
                ProteinBertLayerCross(
                    dim_token=dim_token,
                    dim_global=dim_global,
                    narrow_conv_kernel=narrow_conv_kernel,
                    wide_conv_dilation=wide_conv_dilation,
                    wide_conv_kernel=wide_conv_kernel,
                    attn_heads=attn_heads,
                    attn_dim_head=attn_dim_head,
                )
                for _ in range(cross_depth)
            ]
        )

        # huggingface的分类头
        self.classifier = nn.Sequential(
            nn.Linear(2 * dim_global, dim_global),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_global, 1),
            Rearrange("b () () -> b"),
        )

        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(
            reduction=None, pos_weight=torch.tensor(pos_weight)
        )

    def forward(
        self, input: dict, label: Optional[dict], my_generator: Optional[MyGenerator]
    ) -> dict:
        protein_id = input["protein_id"].to(self.device)
        DNA_id = input["DNA_id"].to(self.device)

        protein_tokens, protein_annotation = self.protein_bert(protein_id)
        DNA_tokens, DNA_annotation = self.DNA_bert(DNA_id)
        for layer_cross in self.layer_crosses:
            protein_tokens, protein_annotation, DNA_tokens, DNA_annotation = (
                layer_cross(
                    protein_tokens, protein_annotation, DNA_tokens, DNA_annotation
                )
            )
        logit = self.classifier(
            torch.cat(
                (
                    protein_annotation,
                    DNA_annotation,
                ),
                dim=2,
            )
        )

        if label is not None:
            losses, loss_num = self.loss_fun(logit, label["bind"])
            self.data_collator.recent_losses.update(
                {
                    (rn, actual_index): loss.item()
                    for rn, actual_index, loss in zip(
                        label["rn"], label["actual_index"], losses
                    )
                }
            )
            return {
                "logit": logit,
                "loss": losses.sum(),
                "loss_num": loss_num,
            }
        return {"logit": logit}

    def loss_fun(self, logit: torch.Tensor, bind: torch.Tensor) -> float:
        batch_size = logit.shape[0]
        losses = self.bce_with_logits_loss(input=logit, target=bind)
        loss_num = batch_size

        return losses, loss_num

    def my_initialize_model(
        self, my_initializer: MyInitializer, my_generator: MyGenerator
    ) -> None:
        my_initializer(self, my_generator)
        self.protein_bert.load_pretrain_weights(self.protein_bert_pretrained_weights)

    def eval_output(self, examples: list[dict], batch: dict) -> pd.DataFrame:
        batch_size = len(examples)
        result = self(input=batch["input"], label=None, my_generator=None)
        probas = F.sigmoid(result["logit"]).cpu().numpy()
        df = pd.DataFrame(
            {
                "sample_idx": np.arange(batch_size),
                "proba": probas,
                "protein": [example["protein"] for example in examples],
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
