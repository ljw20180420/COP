import torch
from common_ai.non_causality_hyena import HyenaOperator
from common_ai.utils import Residual
from einops import einsum
from einops.layers.torch import EinMix
from torch import nn

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from torchtune.modules import MultiHeadAttention, RotaryPositionalEmbeddings


class SecondEncoder(nn.Module):
    def __init__(
        self,
        vocab: int,
        max_num_tokens: int,
        dim_emb: int,
        dim_head: int,
        heads: int,
        depth: int,
        dim_ffn: int,
        dropout: float,
        use_hyena: bool,
        hyena_order: int,
        hyena_filter_order: int,
    ) -> None:
        super().__init__()

        self.depth = depth
        self.use_hyena = use_hyena

        self.embed = nn.Sequential(
            nn.Embedding(
                vocab,
                dim_emb,
            ),
            nn.Dropout(dropout),
        )

        self.rms_norms = nn.ModuleList([nn.RMSNorm(dim_emb) for _ in range(depth)])

        self.self_attentions = nn.ModuleList(
            [
                (
                    MultiHeadAttention(
                        embed_dim=dim_emb,
                        num_heads=heads,
                        num_kv_heads=heads,
                        head_dim=dim_head,
                        q_proj=EinMix(
                            "b s d -> b s nhd",
                            weight_shape="d nhd",
                            bias_shape="nhd",
                            d=dim_emb,
                            nhd=heads * dim_head,
                        ),
                        k_proj=EinMix(
                            "b s d -> b s nhd",
                            weight_shape="d nhd",
                            bias_shape="nhd",
                            d=dim_emb,
                            nhd=heads * dim_head,
                        ),
                        v_proj=EinMix(
                            "b s d -> b s nhd",
                            weight_shape="d nhd",
                            bias_shape="nhd",
                            d=dim_emb,
                            nhd=heads * dim_head,
                        ),
                        output_proj=EinMix(
                            "b s nhd -> b s d",
                            weight_shape="nhd d",
                            bias_shape="d",
                            d=dim_emb,
                            nhd=heads * dim_head,
                        ),
                        pos_embeddings=RotaryPositionalEmbeddings(
                            dim=dim_head, max_seq_len=max_num_tokens
                        ),
                        max_seq_len=max_num_tokens,
                        is_causal=False,
                        attn_dropout=dropout,
                    )
                    if not use_hyena
                    else HyenaOperator(
                        d_model=dim_emb,
                        l_max=max_num_tokens,
                        order=hyena_order,
                        filter_order=hyena_filter_order,
                        dropout=dropout,
                        filter_dropout=dropout,
                    )
                )
                for _ in range(depth)
            ]
        )

        self.ffns = nn.ModuleList(
            [
                Residual(
                    nn.Sequential(
                        nn.RMSNorm(dim_emb),
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

        self.last_rms_norm = nn.RMSNorm(dim_emb)

    def forward(self, second_ids: torch.Tensor) -> torch.Tensor:
        mask = second_ids != 0
        embs = self.embed(second_ids)
        for i in range(self.depth):
            embs_rms = self.rms_norms[i](embs)
            if not self.use_hyena:
                embs = embs + self.self_attentions[i](
                    x=embs_rms,
                    y=embs_rms,
                    mask=einsum(mask, mask, "b s1, b s2 -> b s1 s2"),
                )
            else:
                # self_attentions[i] is actually hyena
                embs = embs + self.self_attentions[i](embs_rms)
            embs = self.ffns[i](embs)

        embs = self.last_rms_norm(embs)

        return embs, mask


class DNAEncoder(nn.Module):
    def __init__(
        self,
        vocab: int,
        max_num_tokens: int,
        dim_emb: int,
        dim_head: int,
        heads: int,
        depth: int,
        dim_ffn: int,
        dropout: float,
        use_hyena: bool,
        hyena_order: int,
        hyena_filter_order: int,
    ) -> None:
        super().__init__()

        self.depth = depth
        self.use_hyena = use_hyena

        self.embed = nn.Sequential(
            nn.Embedding(
                vocab,
                dim_emb,
            ),
            nn.Dropout(dropout),
        )

        self.rms_norms = nn.ModuleList([nn.RMSNorm(dim_emb) for _ in range(depth)])

        self.self_attentions = nn.ModuleList(
            [
                (
                    MultiHeadAttention(
                        embed_dim=dim_emb,
                        num_heads=heads,
                        num_kv_heads=heads,
                        head_dim=dim_head,
                        q_proj=EinMix(
                            "b s d -> b s nhd",
                            weight_shape="d nhd",
                            bias_shape="nhd",
                            d=dim_emb,
                            nhd=heads * dim_head,
                        ),
                        k_proj=EinMix(
                            "b s d -> b s nhd",
                            weight_shape="d nhd",
                            bias_shape="nhd",
                            d=dim_emb,
                            nhd=heads * dim_head,
                        ),
                        v_proj=EinMix(
                            "b s d -> b s nhd",
                            weight_shape="d nhd",
                            bias_shape="nhd",
                            d=dim_emb,
                            nhd=heads * dim_head,
                        ),
                        output_proj=EinMix(
                            "b s nhd -> b s d",
                            weight_shape="nhd d",
                            bias_shape="d",
                            d=dim_emb,
                            nhd=heads * dim_head,
                        ),
                        # DNA starts with CLS token, so increase max_seq_len by 1
                        pos_embeddings=RotaryPositionalEmbeddings(
                            dim=dim_head, max_seq_len=max_num_tokens + 1
                        ),
                        max_seq_len=max_num_tokens + 1,
                        is_causal=False,
                        attn_dropout=dropout,
                    )
                    if not use_hyena
                    else HyenaOperator(
                        d_model=dim_emb,
                        l_max=max_num_tokens + 1,
                        order=hyena_order,
                        filter_order=hyena_filter_order,
                        dropout=dropout,
                        filter_dropout=dropout,
                    )
                )
                for _ in range(depth)
            ]
        )

        self.dna_protein_rms_norms = nn.ModuleList(
            [nn.RMSNorm(dim_emb) for _ in range(depth)]
        )

        self.dna_protein_cross_attentions = nn.ModuleList(
            [
                MultiHeadAttention(
                    embed_dim=dim_emb,
                    num_heads=heads,
                    num_kv_heads=heads,
                    head_dim=dim_head,
                    q_proj=EinMix(
                        "b s d -> b s nhd",
                        weight_shape="d nhd",
                        bias_shape="nhd",
                        d=dim_emb,
                        nhd=heads * dim_head,
                    ),
                    k_proj=EinMix(
                        "b s d -> b s nhd",
                        weight_shape="d nhd",
                        bias_shape="nhd",
                        d=dim_emb,
                        nhd=heads * dim_head,
                    ),
                    v_proj=EinMix(
                        "b s d -> b s nhd",
                        weight_shape="d nhd",
                        bias_shape="nhd",
                        d=dim_emb,
                        nhd=heads * dim_head,
                    ),
                    output_proj=EinMix(
                        "b s nhd -> b s d",
                        weight_shape="nhd d",
                        bias_shape="d",
                        d=dim_emb,
                        nhd=heads * dim_head,
                    ),
                    # protein bert clip the protein sequence by <start> and <end> tokens, so increase max_seq_len by 2
                    pos_embeddings=RotaryPositionalEmbeddings(
                        dim=dim_head, max_seq_len=max_num_tokens + 2
                    ),
                    max_seq_len=max_num_tokens + 2,
                    is_causal=False,
                    attn_dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.dna_second_rms_norms = nn.ModuleList(
            [nn.RMSNorm(dim_emb) for _ in range(depth)]
        )

        self.dna_second_cross_attentions = nn.ModuleList(
            [
                MultiHeadAttention(
                    embed_dim=dim_emb,
                    num_heads=heads,
                    num_kv_heads=heads,
                    head_dim=dim_head,
                    q_proj=EinMix(
                        "b s d -> b s nhd",
                        weight_shape="d nhd",
                        bias_shape="nhd",
                        d=dim_emb,
                        nhd=heads * dim_head,
                    ),
                    k_proj=EinMix(
                        "b s d -> b s nhd",
                        weight_shape="d nhd",
                        bias_shape="nhd",
                        d=dim_emb,
                        nhd=heads * dim_head,
                    ),
                    v_proj=EinMix(
                        "b s d -> b s nhd",
                        weight_shape="d nhd",
                        bias_shape="nhd",
                        d=dim_emb,
                        nhd=heads * dim_head,
                    ),
                    output_proj=EinMix(
                        "b s nhd -> b s d",
                        weight_shape="nhd d",
                        bias_shape="d",
                        d=dim_emb,
                        nhd=heads * dim_head,
                    ),
                    # DNA starts with CLS token, so increase max_seq_len by 1
                    pos_embeddings=RotaryPositionalEmbeddings(
                        dim=dim_head, max_seq_len=max_num_tokens + 1
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
                        nn.RMSNorm(dim_emb),
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

        self.last_rms_norm = nn.RMSNorm(dim_emb)

    def forward(
        self,
        dna_ids: torch.Tensor,
        protein_embs: torch.Tensor,
        second_embs: torch.Tensor,
        second_mask: torch.Tensor,
    ) -> torch.Tensor:
        dna_mask = dna_ids != 0
        dna_embs = self.embed(dna_ids)
        for i in range(self.depth):
            dna_embs_rms = self.rms_norms[i](dna_embs)
            if not self.use_hyena:
                dna_embs = dna_embs + self.self_attentions[i](
                    x=dna_embs_rms,
                    y=dna_embs_rms,
                    mask=einsum(dna_mask, dna_mask, "b s1, b s2 -> b s1 s2"),
                )
            else:
                # self_attentions[i] is actually hyena
                dna_embs = dna_embs + self.self_attentions[i](dna_embs_rms)

            # DNA and protein cross-attention
            dna_embs_rms = self.dna_protein_rms_norms[i](dna_embs)
            dna_embs = dna_embs + self.dna_protein_cross_attentions[i](
                x=dna_embs_rms,
                y=protein_embs,
            )

            # DNA and secondary structure cross-attention
            dna_embs_rms = self.dna_second_rms_norms[i](dna_embs)
            dna_embs = dna_embs + self.dna_second_cross_attentions[i](
                x=dna_embs_rms,
                y=second_embs,
                mask=einsum(dna_mask, second_mask, "b s1, b s2 -> b s1 s2"),
            )

            # DNA feedforward network
            dna_embs = self.ffns[i](dna_embs)

        # only use the embedding for CLS token
        dna_embs = dna_embs[:, 0, :]
        dna_embs = self.last_rms_norm(dna_embs)

        return dna_embs
