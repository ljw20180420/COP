import os
import numpy as np
import pandas as pd
import torch
import pickle
from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from typing import Optional

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops.layers.torch import Rearrange, EinMix
from einops import rearrange, repeat, einsum

from .data_collator import DataCollator
from ..utils import Residual


class ProteinBertCrossAttention(nn.Module):
    def __init__(
        self, dim_token: int, dim_global: int, heads: int, dim_head: int
    ) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_global = dim_global
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(
            nn.Linear(dim_global, heads * dim_head, bias=False),
            nn.Tanh(),
        )
        self.to_k = nn.Sequential(
            nn.Linear(dim_token, heads * dim_head, bias=False),
            nn.Tanh(),
        )
        self.to_v = nn.Sequential(
            nn.Linear(dim_token, dim_global, bias=False),
            nn.GELU(),
            Rearrange("b s (h d) -> b s h d", h=heads, d=dim_global // heads),
        )

    def forward(self, tokens: torch.Tensor, annotation: torch.Tensor) -> torch.Tensor:
        q = self.to_q(annotation)
        k = self.to_k(tokens)
        v = self.to_v(tokens)

        sim = (
            einsum(
                rearrange(
                    q, "b () (h d_h) -> b h d_h", h=self.heads, d_h=self.dim_head
                ),
                rearrange(
                    k, "b s (h d_h) -> b s h d_h", h=self.heads, d_h=self.dim_head
                ),
                "b h d_h, b s h d_h -> b s h",
            )
            * self.scale
        )
        attn = sim.softmax(dim=1)
        out = einsum(attn, v, "b s h, b s h d -> b h d")
        out = rearrange(out, "b h d -> b () (h d)", h=self.heads)
        return out


class ProteinBertLayer(nn.Module):
    def __init__(
        self,
        dim_token: int,
        dim_global: int,
        narrow_conv_kernel: int,
        wide_conv_kernel: int,
        wide_conv_dilation: int,
        attn_heads: int,
        attn_dim_head: int,
    ) -> None:
        super().__init__()

        self.narrow_conv = nn.Sequential(
            Rearrange("b s d -> b d s", d=dim_token),
            nn.Conv1d(
                dim_token,
                dim_token,
                narrow_conv_kernel,
                padding=narrow_conv_kernel // 2,
            ),
            Rearrange("b d s -> b s d", d=dim_token),
            nn.GELU(),
        )

        self.wide_conv = nn.Sequential(
            Rearrange("b s d -> b d s", d=dim_token),
            nn.Conv1d(
                dim_token,
                dim_token,
                wide_conv_kernel,
                dilation=wide_conv_dilation,
                padding=wide_conv_kernel // 2 * wide_conv_dilation,
            ),
            Rearrange("b d s -> b s d", d=dim_token),
            nn.GELU(),
        )

        self.extract_global_info = nn.Sequential(
            nn.Linear(dim_global, dim_token),
            nn.GELU(),
        )

        self.local_feedforward = nn.Sequential(
            nn.LayerNorm(dim_token, eps=1e-3),
            Residual(
                nn.Sequential(
                    nn.Linear(dim_token, dim_token),
                    nn.GELU(),
                )
            ),
            nn.LayerNorm(dim_token, eps=1e-3),
        )

        self.global_attend_local = ProteinBertCrossAttention(
            dim_token=dim_token,
            dim_global=dim_global,
            heads=attn_heads,
            dim_head=attn_dim_head,
        )

        self.global_dense = nn.Sequential(
            nn.Linear(dim_global, dim_global),
            nn.GELU(),
        )

        self.global_feedforward = nn.Sequential(
            nn.LayerNorm(dim_global, eps=1e-3),
            Residual(
                nn.Sequential(
                    nn.Linear(dim_global, dim_global),
                    nn.GELU(),
                )
            ),
            nn.LayerNorm(dim_global, eps=1e-3),
        )

    def forward(
        self, tokens: torch.Tensor, annotation: torch.Tensor
    ) -> tuple[torch.Tensor]:
        # process local (protein sequence)
        narrow_out = self.narrow_conv(tokens)
        wide_out = self.wide_conv(tokens)
        global_info = self.extract_global_info(annotation)

        tokens = tokens + narrow_out + wide_out + global_info
        tokens = self.local_feedforward(tokens)

        # process global (annotations)
        annotation = (
            annotation
            + self.global_dense(annotation)
            + self.global_attend_local(tokens, annotation)
        )
        annotation = self.global_feedforward(annotation)

        return tokens, annotation


class ProteinBertLayerCross(nn.Module):
    def __init__(
        self,
        dim_token: int,
        dim_global: int,
        narrow_conv_kernel: int,
        wide_conv_kernel: int,
        wide_conv_dilation: int,
        attn_heads: int,
        attn_dim_head: int,
    ) -> None:
        assert (
            dim_token % 2 == 0 and dim_global % 2 == 0
        ), "token and global dimension cannot be odd"

        self.ptpg = ProteinBertLayer(
            dim_token,
            dim_global,
            narrow_conv_kernel,
            wide_conv_kernel,
            wide_conv_dilation,
            attn_heads,
            attn_dim_head,
        )

        self.ptdg = ProteinBertLayer(
            dim_token,
            dim_global,
            narrow_conv_kernel,
            wide_conv_kernel,
            wide_conv_dilation,
            attn_heads,
            attn_dim_head,
        )

        self.dtpg = ProteinBertLayer(
            dim_token,
            dim_global,
            narrow_conv_kernel,
            wide_conv_kernel,
            wide_conv_dilation,
            attn_heads,
            attn_dim_head,
        )

        self.dtdg = ProteinBertLayer(
            dim_token,
            dim_global,
            narrow_conv_kernel,
            wide_conv_kernel,
            wide_conv_dilation,
            attn_heads,
            attn_dim_head,
        )

        self.protein_local_feedforward = nn.Sequential(
            nn.LayerNorm(2 * dim_token, eps=1e-3),
            Residual(
                nn.Sequential(
                    nn.Linear(2 * dim_token, dim_token),
                    nn.GELU(),
                )
            ),
            nn.LayerNorm(dim_token, eps=1e-3),
        )

        self.protein_global_feedforward = nn.Sequential(
            nn.LayerNorm(2 * dim_global, eps=1e-3),
            Residual(
                nn.Sequential(
                    nn.Linear(2 * dim_global, dim_global),
                    nn.GELU(),
                )
            ),
            nn.LayerNorm(dim_global, eps=1e-3),
        )

        self.DNA_local_feedforward = nn.Sequential(
            nn.LayerNorm(2 * dim_token, eps=1e-3),
            Residual(
                nn.Sequential(
                    nn.Linear(2 * dim_token, dim_token),
                    nn.GELU(),
                )
            ),
            nn.LayerNorm(dim_token, eps=1e-3),
        )

        self.DNA_global_feedforward = nn.Sequential(
            nn.LayerNorm(2 * dim_global, eps=1e-3),
            Residual(
                nn.Sequential(
                    nn.Linear(2 * dim_global, dim_global),
                    nn.GELU(),
                )
            ),
            nn.LayerNorm(dim_global, eps=1e-3),
        )

    def forward(
        self,
        protein_tokens: torch.Tensor,
        protein_annotation: torch.Tensor,
        DNA_tokens: torch.Tensor,
        DNA_annotation: torch.Tensor,
    ):
        protein_tokens_ptpg, protein_annotation_ptpg = self.ptpg(
            protein_tokens, protein_annotation
        )
        protein_tokens_ptdg, DNA_annotation_ptdg = self.ptdg(
            protein_tokens, DNA_annotation
        )
        DNA_tokens_dtpg, protein_annotation_dtpg = self.dtpg(
            DNA_tokens, protein_annotation
        )
        DNA_tokens_dtdg, DNA_annotation_dtdg = self.dtdg(DNA_tokens, DNA_annotation)
        protein_tokens = self.protein_local_feedforward(
            torch.cat(
                (
                    protein_tokens_ptpg,
                    protein_tokens_ptdg,
                )
            ),
            dim=2,
        )
        protein_annotation = self.protein_global_feedforward(
            torch.cat(
                (
                    protein_annotation_ptpg,
                    protein_annotation_dtpg,
                )
            ),
            dim=2,
        )
        DNA_tokens = self.DNA_local_feedforward(
            torch.cat(
                (
                    DNA_tokens_dtpg,
                    DNA_tokens_dtdg,
                )
            ),
            dim=2,
        )
        DNA_annotation = self.DNA_global_feedforward(
            torch.cat(
                (
                    DNA_annotation_ptdg,
                    DNA_annotation_dtdg,
                )
            ),
            dim=2,
        )
        return protein_tokens, protein_annotation, DNA_tokens, DNA_annotation


class ProteinBertConfig(PretrainedConfig):
    model_type = "ProteinBert"

    def __init__(
        self,
        num_tokens: int,
        dim_token: int,
        dim_global: int,
        depth: int,
        narrow_conv_kernel: int,
        wide_conv_kernel: int,
        wide_conv_dilation: int,
        attn_heads: int,
        attn_dim_head: int,
        **kwargs,
    ) -> None:
        """ProteinBert arguments.

        Args:
            num_tokens: Number of tokens.
            dim_token: Token-wise embedding dimension.
            dim_global: Global embedding dimension.
            depth: Number of bert layers.
            narrow_conv_kernel: Kernal size of narrow convolution.
            wide_conv_kernel: Kernel size of wide convolution.
            wide_conv_dilation: Dilation of wide convolution.
            attn_heads: Number of cross attension heads.
            attn_dim_head: Dimension of each cross attension head.
        """
        self.num_tokens = num_tokens
        self.dim_token = dim_token
        self.dim_global = dim_global
        self.depth = depth
        self.narrow_conv_kernel = narrow_conv_kernel
        self.wide_conv_kernel = wide_conv_kernel
        self.wide_conv_dilation = wide_conv_dilation
        self.attn_heads = attn_heads
        self.attn_dim_head = attn_dim_head
        super().__init__(**kwargs)


class ProteinBertModel(PreTrainedModel):
    config_class = ProteinBertConfig

    def __init__(self, config: ProteinBertConfig) -> None:
        super().__init__(config)
        self.token_emb = nn.Embedding(config.num_tokens, config.dim_token)
        self.global_bias = nn.Parameter(torch.zeros(config.dim_global))
        self.active_global = nn.GELU()

        self.layers = nn.ModuleList(
            [
                ProteinBertLayer(
                    dim_token=config.dim_token,
                    dim_global=config.dim_global,
                    narrow_conv_kernel=config.narrow_conv_kernel,
                    wide_conv_dilation=config.wide_conv_dilation,
                    wide_conv_kernel=config.wide_conv_kernel,
                    attn_heads=config.attn_heads,
                    attn_dim_head=config.attn_dim_head,
                )
                for _ in range(config.depth)
            ]
        )

    def forward(self, id: torch.Tensor) -> tuple[torch.Tensor]:
        batch_size = id.shape[0]

        tokens = self.token_emb(id)

        annotation = repeat(
            self.active_global(self.global_bias),
            "d -> b () d",
            b=batch_size,
            d=self.config.dim_global,
        )

        for layer in self.layers:
            tokens, annotation = layer(tokens, annotation)

        return tokens, annotation

    def load_pretrain_weights(self, weights_file: os.PathLike):
        with open(weights_file, "rb") as fd:
            _, model_weights, _ = pickle.load(fd)
        self.global_bias.data = torch.from_numpy(model_weights[1])
        self.token_emb.weight.data = torch.from_numpy(model_weights[2])

        for i, layer in enumerate(self.layers):
            # torch Linear weight is (out_feature, in_feature)
            # tensorflow Linear weight is (in_feature, out_feature)
            # EinMix weight is (in_feature, out_feature), the same as tensorflow
            # EinMix bias is (1, ..., 1, out_feature)
            layer.extract_global_info[0].weight.data = rearrange(
                torch.from_numpy(model_weights[i * 23 + 3]), "in out -> out in"
            )
            layer.extract_global_info[0].bias.data = torch.from_numpy(
                model_weights[i * 23 + 4]
            )
            # torch Conv weight is (out_channel, in_channel, kernel_dim1, kernel_dim2, ...)
            # tensorflow Conv weight is (kernel_dim1, kernel_dim2, ..., in_channel, out_channel)
            layer.narrow_conv[1].weight.data = torch.from_numpy(
                model_weights[i * 23 + 5]
            ).permute(2, 1, 0)
            layer.narrow_conv[1].bias.data = torch.from_numpy(model_weights[i * 23 + 6])
            layer.wide_conv[1].weight.data = torch.from_numpy(
                model_weights[i * 23 + 7]
            ).permute(2, 1, 0)
            layer.wide_conv[1].bias.data = torch.from_numpy(model_weights[i * 23 + 8])
            layer.local_feedforward[0].weight.data = torch.from_numpy(
                model_weights[i * 23 + 9]
            )
            layer.local_feedforward[0].bias.data = torch.from_numpy(
                model_weights[i * 23 + 10]
            )
            layer.local_feedforward[1].module[0].weight.data = rearrange(
                torch.from_numpy(model_weights[i * 23 + 11]), "in out -> out in"
            )
            layer.local_feedforward[1].module[0].bias.data = torch.from_numpy(
                model_weights[i * 23 + 12]
            )
            layer.local_feedforward[2].weight.data = torch.from_numpy(
                model_weights[i * 23 + 13]
            )
            layer.local_feedforward[2].bias.data = torch.from_numpy(
                model_weights[i * 23 + 14]
            )
            layer.global_dense[0].weight.data = rearrange(
                torch.from_numpy(model_weights[i * 23 + 15]), "in out -> out in"
            )
            layer.global_dense[0].bias.data = torch.from_numpy(
                model_weights[i * 23 + 16]
            )
            layer.global_attend_local.to_q[0].weight.data = rearrange(
                torch.from_numpy(model_weights[i * 23 + 17]),
                "n d hd -> (n hd) d",
            )
            layer.global_attend_local.to_k[0].weight.data = rearrange(
                torch.from_numpy(model_weights[i * 23 + 18]),
                "n d hd -> (n hd) d",
            )
            layer.global_attend_local.to_v[0].weight.data = rearrange(
                torch.from_numpy(model_weights[i * 23 + 19]),
                "n d hd -> (n hd) d",
            )
            layer.global_feedforward[0].weight.data = torch.from_numpy(
                model_weights[i * 23 + 20]
            )
            layer.global_feedforward[0].bias.data = torch.from_numpy(
                model_weights[i * 23 + 21]
            )
            layer.global_feedforward[1].module[0].weight.data = rearrange(
                torch.from_numpy(model_weights[i * 23 + 22]), "in out -> out in"
            )
            layer.global_feedforward[1].module[0].bias.data = torch.from_numpy(
                model_weights[i * 23 + 23]
            )
            layer.global_feedforward[2].weight.data = torch.from_numpy(
                model_weights[i * 23 + 24]
            )
            layer.global_feedforward[2].bias.data = torch.from_numpy(
                model_weights[i * 23 + 25]
            )


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

        self.protein_bert = ProteinBertModel(
            ProteinBertConfig(
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
        )

        self.DNA_bert = ProteinBertModel(
            ProteinBertConfig(
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
            pos_weight=torch.tensor(config.pos_weight)
        )

    def forward(self, input: dict, label: Optional[dict] = None) -> dict:
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
            self.data_collator.minimal_losses.update(
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
        result = self(input=batch["input"])
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
