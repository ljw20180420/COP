import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from typing import Optional, Callable

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops.layers.torch import Rearrange

from common_ai.utils import MyGenerator
from common_ai.protein_bert import ProteinBert, ProteinBertLayerCross
from .data_collator import DataCollator


class PDBertConfig(PretrainedConfig):
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
        **kwargs,
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
        self.protein_num_tokens = protein_num_tokens
        self.DNA_num_tokens = DNA_num_tokens
        self.dim_token = dim_token
        self.dim_global = dim_global
        self.protein_depth = protein_depth
        self.DNA_depth = DNA_depth
        self.cross_depth = cross_depth
        self.narrow_conv_kernel = narrow_conv_kernel
        self.wide_conv_kernel = wide_conv_kernel
        self.wide_conv_dilation = wide_conv_dilation
        self.attn_heads = attn_heads
        self.attn_dim_head = attn_dim_head
        self.pos_weight = pos_weight
        self.dropout = dropout
        self.protein_data = protein_data
        self.protein_length = protein_length
        self.DNA_length = DNA_length
        self.minimal_unbind_summit_distance = minimal_unbind_summit_distance
        self.select_worst_loss_ratio = select_worst_loss_ratio
        self.protein_bert_pretrained_weights = protein_bert_pretrained_weights
        super().__init__(**kwargs)


class PDBertModel(PreTrainedModel):
    config_class = PDBertConfig

    def __init__(self, config: PDBertConfig) -> None:
        super().__init__(config)
        self.data_collator = DataCollator(
            protein_data=config.protein_data,
            protein_length=config.protein_length,
            DNA_length=config.DNA_length,
            minimal_unbind_summit_distance=config.minimal_unbind_summit_distance,
            select_worst_loss_ratio=config.select_worst_loss_ratio,
        )

        self.protein_bert = ProteinBert(
            num_tokens=config.protein_num_tokens,
            dim_token=config.dim_token,
            dim_global=config.dim_global,
            depth=config.protein_depth,
            narrow_conv_kernel=config.narrow_conv_kernel,
            wide_conv_kernel=config.wide_conv_kernel,
            wide_conv_dilation=config.wide_conv_dilation,
            attn_heads=config.attn_heads,
            attn_dim_head=config.attn_dim_head,
        )

        self.DNA_bert = ProteinBert(
            num_tokens=config.DNA_num_tokens,
            dim_token=config.dim_token,
            dim_global=config.dim_global,
            depth=config.DNA_depth,
            narrow_conv_kernel=config.narrow_conv_kernel,
            wide_conv_kernel=config.wide_conv_kernel,
            wide_conv_dilation=config.wide_conv_dilation,
            attn_heads=config.attn_heads,
            attn_dim_head=config.attn_dim_head,
        )

        self.layer_crosses = nn.ModuleList(
            [
                ProteinBertLayerCross(
                    dim_token=config.dim_token,
                    dim_global=config.dim_global,
                    narrow_conv_kernel=config.narrow_conv_kernel,
                    wide_conv_dilation=config.wide_conv_dilation,
                    wide_conv_kernel=config.wide_conv_kernel,
                    attn_heads=config.attn_heads,
                    attn_dim_head=config.attn_dim_head,
                )
                for _ in range(config.cross_depth)
            ]
        )

        # huggingface的分类头
        self.classifier = nn.Sequential(
            nn.Linear(2 * config.dim_global, config.dim_global),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_global, 1),
            Rearrange("b () () -> b"),
        )

        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(
            reduction=None, pos_weight=torch.tensor(config.pos_weight)
        )

    def forward(
        self, input: dict, label: Optional[dict], my_generator: MyGenerator
    ) -> dict:
        protein_id = input["protein_id"]
        DNA_id = input["DNA_id"]

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

    def my_initialize_model(
        self, my_train_initialize_model: Callable, initializer: Callable
    ):
        my_train_initialize_model(self, initializer)
        self.protein_bert.load_pretrain_weights(
            self.config.protein_bert_pretrained_weights
        )
