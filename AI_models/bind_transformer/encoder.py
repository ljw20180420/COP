import torch
from torch import nn

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from einops import einsum
from einops.layers.torch import EinMix
from torchtune.modules import RotaryPositionalEmbeddings, MultiHeadAttention
from .common import Residual


class Second_Encoder(nn.Module):
    def __init__(
        self,
        vocab: int,
        max_num_tokens: int,
        dim_emb: int,
        dim_heads: int,
        num_heads: int,
        depth: int,
        dim_ffn: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()

        self.depth = depth

        # 二级结构编码器: 蛋白质9种二级结构+锌指结构+KRAB总共11种二级结构的token编码（算上mask token有12个）
        self.embed = nn.Sequential(
            nn.Embedding(
                vocab,
                dim_emb,
            ),
            nn.Dropout(dropout),
        )

        self.rms_norms = nn.ModuleList(
            [nn.RMSNorm(dim_emb, eps=norm_eps) for _ in range(depth)]
        )

        self.self_attentions = nn.ModuleList(
            [
                MultiHeadAttention(
                    embed_dim=dim_emb,
                    num_heads=num_heads,
                    num_kv_heads=num_heads,
                    head_dim=dim_heads,
                    q_proj=EinMix(
                        "b s d -> b s nhd",
                        weight_shape="d nhd",
                        bias_shape="nhd",
                        d=dim_emb,
                        nhd=num_heads * dim_heads,
                    ),
                    k_proj=EinMix(
                        "b s d -> b s nhd",
                        weight_shape="d nhd",
                        bias_shape="nhd",
                        d=dim_emb,
                        nhd=num_heads * dim_heads,
                    ),
                    v_proj=EinMix(
                        "b s d -> b s nhd",
                        weight_shape="d nhd",
                        bias_shape="nhd",
                        d=dim_emb,
                        nhd=num_heads * dim_heads,
                    ),
                    output_proj=EinMix(
                        "b s nhd -> b s d",
                        weight_shape="nhd d",
                        bias_shape="d",
                        d=dim_emb,
                        nhd=num_heads * dim_heads,
                    ),
                    pos_embeddings=RotaryPositionalEmbeddings(
                        dim=dim_heads, max_seq_len=max_num_tokens
                    ),
                    max_seq_len=max_num_tokens,
                    is_causal=False,
                    attn_dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.ffns = nn.ModuleList(
            [
                Residual(
                    nn.Sequential(
                        nn.RMSNorm(dim_emb, eps=norm_eps),
                        EinMix(
                            "b s d -> b s d_f",
                            weight_shape="d d_f",
                            bias_shape="d_f",
                            d=dim_emb,
                            d_f=dim_ffn,
                        ),
                        nn.GELU(),
                        EinMix(
                            "b s d_f -> b s d",
                            weight_shape="d_f d",
                            bias_shape="d",
                            d=dim_emb,
                            d_f=dim_ffn,
                        ),
                        nn.Dropout(dropout),
                    )
                )
                for _ in range(depth)
            ]
        )

        self.last_rms_norm = nn.RMSNorm(dim_emb, eps=norm_eps)

    def forward(self, second_ids: torch.Tensor) -> torch.Tensor:
        # 二级结构mask
        mask = second_ids != 0
        # 二级结构编码
        embs = self.embed(second_ids)
        for i in range(self.depth):
            embs_rms = self.rms_norms[i](embs)
            embs = embs + self.self_attentions[i](
                x=embs_rms, y=embs_rms, mask=einsum(mask, mask, "b s1, b s2 -> b s1 s2")
            )
            embs = self.ffns[i](embs)

        embs = self.last_rms_norm(embs)

        return embs, mask


class DNA_Encoder(nn.Module):
    def __init__(
        self,
        vocab: int,
        max_num_tokens: int,
        dim_emb: int,
        dim_heads: int,
        num_heads: int,
        depth: int,
        dim_ffn: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()

        self.depth = depth

        # DNA4个核苷酸的token编码（算上[CLS] token和mask token有6个）
        # 关于[CLS] token，参考BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
        self.embed = nn.Sequential(
            nn.Embedding(
                vocab,
                dim_emb,
            ),
            nn.Dropout(dropout),
        )

        self.rms_norms = nn.ModuleList(
            [nn.RMSNorm(dim_emb, eps=norm_eps) for _ in range(depth)]
        )

        self.self_attentions = nn.ModuleList(
            [
                MultiHeadAttention(
                    embed_dim=dim_emb,
                    num_heads=num_heads,
                    num_kv_heads=num_heads,
                    head_dim=dim_heads,
                    q_proj=EinMix(
                        "b s d -> b s nhd",
                        weight_shape="d nhd",
                        bias_shape="nhd",
                        d=dim_emb,
                        nhd=num_heads * dim_heads,
                    ),
                    k_proj=EinMix(
                        "b s d -> b s nhd",
                        weight_shape="d nhd",
                        bias_shape="nhd",
                        d=dim_emb,
                        nhd=num_heads * dim_heads,
                    ),
                    v_proj=EinMix(
                        "b s d -> b s nhd",
                        weight_shape="d nhd",
                        bias_shape="nhd",
                        d=dim_emb,
                        nhd=num_heads * dim_heads,
                    ),
                    output_proj=EinMix(
                        "b s nhd -> b s d",
                        weight_shape="nhd d",
                        bias_shape="d",
                        d=dim_emb,
                        nhd=num_heads * dim_heads,
                    ),
                    # DNA开头有个[CLS] token, 所以最大长度要+1
                    pos_embeddings=RotaryPositionalEmbeddings(
                        dim=dim_heads, max_seq_len=max_num_tokens + 1
                    ),
                    max_seq_len=max_num_tokens + 1,
                    is_causal=False,
                    attn_dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.dna_protein_rms_norms = nn.ModuleList(
            [nn.RMSNorm(dim_emb, eps=norm_eps) for _ in range(depth)]
        )

        self.dna_protein_cross_attentions = nn.ModuleList(
            [
                MultiHeadAttention(
                    embed_dim=dim_emb,
                    num_heads=num_heads,
                    num_kv_heads=num_heads,
                    head_dim=dim_heads,
                    q_proj=EinMix(
                        "b s d -> b s nhd",
                        weight_shape="d nhd",
                        bias_shape="nhd",
                        d=dim_emb,
                        nhd=num_heads * dim_heads,
                    ),
                    k_proj=EinMix(
                        "b s d -> b s nhd",
                        weight_shape="d nhd",
                        bias_shape="nhd",
                        d=dim_emb,
                        nhd=num_heads * dim_heads,
                    ),
                    v_proj=EinMix(
                        "b s d -> b s nhd",
                        weight_shape="d nhd",
                        bias_shape="nhd",
                        d=dim_emb,
                        nhd=num_heads * dim_heads,
                    ),
                    output_proj=EinMix(
                        "b s nhd -> b s d",
                        weight_shape="nhd d",
                        bias_shape="d",
                        d=dim_emb,
                        nhd=num_heads * dim_heads,
                    ),
                    # protein bert开头结尾有[START]和[END] token, 所以最大长度要+2
                    pos_embeddings=RotaryPositionalEmbeddings(
                        dim=dim_heads, max_seq_len=max_num_tokens + 2
                    ),
                    max_seq_len=max_num_tokens + 2,
                    is_causal=False,
                    attn_dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.dna_second_rms_norms = nn.ModuleList(
            [nn.RMSNorm(dim_emb, eps=norm_eps) for _ in range(depth)]
        )

        self.dna_second_cross_attentions = nn.ModuleList(
            [
                MultiHeadAttention(
                    embed_dim=dim_emb,
                    num_heads=num_heads,
                    num_kv_heads=num_heads,
                    head_dim=dim_heads,
                    q_proj=EinMix(
                        "b s d -> b s nhd",
                        weight_shape="d nhd",
                        bias_shape="nhd",
                        d=dim_emb,
                        nhd=num_heads * dim_heads,
                    ),
                    k_proj=EinMix(
                        "b s d -> b s nhd",
                        weight_shape="d nhd",
                        bias_shape="nhd",
                        d=dim_emb,
                        nhd=num_heads * dim_heads,
                    ),
                    v_proj=EinMix(
                        "b s d -> b s nhd",
                        weight_shape="d nhd",
                        bias_shape="nhd",
                        d=dim_emb,
                        nhd=num_heads * dim_heads,
                    ),
                    output_proj=EinMix(
                        "b s nhd -> b s d",
                        weight_shape="nhd d",
                        bias_shape="d",
                        d=dim_emb,
                        nhd=num_heads * dim_heads,
                    ),
                    # DNA开头有个[CLS] token, 所以最大长度要+1
                    pos_embeddings=RotaryPositionalEmbeddings(
                        dim=dim_heads, max_seq_len=max_num_tokens + 1
                    ),
                    max_seq_len=max_num_tokens + 1,
                    is_causal=False,
                    attn_dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.ffns = nn.ModuleList(
            [
                Residual(
                    nn.Sequential(
                        nn.RMSNorm(dim_emb, eps=norm_eps),
                        EinMix(
                            "b s d -> b s d_f",
                            weight_shape="d d_f",
                            bias_shape="d_f",
                            d=dim_emb,
                            d_f=dim_ffn,
                        ),
                        nn.GELU(),
                        EinMix(
                            "b s d_f -> b s d",
                            weight_shape="d_f d",
                            bias_shape="d",
                            d=dim_emb,
                            d_f=dim_ffn,
                        ),
                        nn.Dropout(dropout),
                    )
                )
                for _ in range(depth)
            ]
        )

        self.last_rms_norm = nn.RMSNorm(dim_emb, eps=norm_eps)

    def forward(
        self,
        dna_ids: torch.Tensor,
        protein_embs: torch.Tensor,
        second_embs: torch.Tensor,
        second_mask: torch.Tensor,
    ) -> torch.Tensor:
        # DNA mask
        dna_mask = dna_ids != 0

        dna_embs = self.embed(dna_ids)
        for i in range(self.depth):
            # DNA自注意力编码
            dna_embs_rms = self.rms_norms[i](dna_embs)
            dna_embs = dna_embs + self.self_attentions[i](
                x=dna_embs_rms,
                y=dna_embs_rms,
                mask=einsum(dna_mask, dna_mask, "b s1, b s2 -> b s1 s2"),
            )

            # Only use the embedding for CLS token.
            dna_embs = dna_embs[:, [0], :]

            # DNA和氨基酸交叉注意力
            dna_embs_rms = self.dna_protein_rms_norms[i](dna_embs)
            dna_embs = dna_embs + self.dna_protein_cross_attentions[i](
                x=dna_embs_rms,
                y=protein_embs,
            )

            # DNA和二级结构交叉注意力
            dna_embs_rms = self.dna_second_rms_norms[i](dna_embs)
            dna_embs = dna_embs + self.dna_second_cross_attentions[i](
                x=dna_embs_rms,
                y=second_embs,
                mask=einsum(dna_mask, second_mask, "b s1, b s2 -> b s1 s2"),
            )

            # DNA前馈网络
            dna_embs = self.ffns[i](dna_embs)

        dna_embs = self.last_rms_norm(dna_embs)

        return dna_embs
