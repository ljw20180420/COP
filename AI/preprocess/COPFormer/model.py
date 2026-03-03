from typing import Union

import torch
import torch.nn as nn
from common_ai.model import MyModelAbstract
from common_ai.protein_bert import ProteinBert
from einops.layers.torch import EinMix, Rearrange

# torch does not import opt_einsum as backend by default. import opt_einsum manually will enable it.
from torch.backends import opt_einsum
from torch.nn import BCEWithLogitsLoss
from transformers import PretrainedConfig, PreTrainedModel

from .encoder import DNA_Encoder, Second_Encoder


class BindTransformerConfig(PretrainedConfig):
    """
    model_type: The type of the model.
    label_names: The labels used by trainer as ground truth.
    """

    model_type = "bind_transformer"
    label_names = ["bind"]

    # Config必须有默认值，否则Trainer在保存模型时会报错
    def __init__(
        self,
        protein_vocab: int = None,
        second_vocab: int = None,
        dna_vocab: int = None,
        max_num_tokens: int = None,
        dim_emb: int = None,
        num_heads: int = None,
        dim_heads: int = None,
        depth: int = None,
        dim_ffn: int = None,
        dropout: float = None,
        norm_eps: float = None,
        pos_weight: float = None,
        reg_l1: float = None,
        reg_l2: float = None,
        initializer_range: float = None,
        use_hyena: bool = None,
        hyena_order: int = None,
        hyena_filter_order: int = None,
        seed: int = None,
        **kwargs,
    ) -> None:
        self.protein_vocab = protein_vocab
        self.second_vocab = second_vocab
        self.dna_vocab = dna_vocab
        self.max_num_tokens = max_num_tokens
        self.dim_emb = dim_emb
        self.num_heads = num_heads
        self.dim_heads = dim_heads
        self.depth = depth
        self.dim_ffn = dim_ffn
        self.dropout = dropout
        self.norm_eps = norm_eps
        self.pos_weight = pos_weight
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.initializer_range = initializer_range
        self.use_hyena = use_hyena
        self.hyena_order = hyena_order
        self.hyena_filter_order = hyena_filter_order
        self.seed = seed
        super().__init__(
            **kwargs,
        )


class BindTransformerModel(PreTrainedModel):
    """
    config_class: The corresponding cofiguration class of BindTransformerModel.
    """

    config_class = BindTransformerConfig

    def __init__(self, config: BindTransformerConfig) -> None:
        """
        config: An instance of the configuration class.
        """
        super().__init__(config)

        # 二级结构编码器
        self.second_encoder = Second_Encoder(
            config.second_vocab,
            config.max_num_tokens,
            config.dim_emb,
            config.dim_heads,
            config.num_heads,
            config.depth,
            config.dim_ffn,
            config.dropout,
            config.norm_eps,
            config.use_hyena,
            config.hyena_order,
            config.hyena_filter_order,
        )

        # DNA编码器
        self.dna_encoder = DNA_Encoder(
            config.dna_vocab,
            config.max_num_tokens,
            config.dim_emb,
            config.dim_heads,
            config.num_heads,
            config.depth,
            config.dim_ffn,
            config.dropout,
            config.norm_eps,
            config.use_hyena,
            config.hyena_order,
            config.hyena_filter_order,
        )

        # huggingface的分类头
        self.classifier = nn.Sequential(
            EinMix(
                "b d -> b d_0",
                weight_shape="d d_0",
                bias_shape="d_0",
                d=config.dim_emb,
                d_0=config.dim_emb,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            EinMix(
                "b d -> b o",
                weight_shape="d o",
                bias_shape="o",
                d=config.dim_emb,
                o=1,
            ),
            Rearrange("b 1 -> b"),
        )

        self.protein_bert_head = nn.Sequential(
            EinMix(
                "b s d_b -> b s d",
                weight_shape="d_b d",
                bias_shape="d",
                d=config.dim_emb,
                d_b=128,
            ),
            nn.Dropout(config.dropout),
            nn.RMSNorm(config.dim_emb, eps=config.norm_eps),
        )

        # 交叉熵
        self.loss_fn = BCEWithLogitsLoss(
            reduction="sum", pos_weight=torch.tensor(config.pos_weight)
        )

        # 弹性网络正则化
        self.elastic_net = Elastic_Net(config.reg_l1, config.reg_l2)

        # 设置随机生成子，让训练可重复
        self.generator = torch.Generator().manual_seed(config.seed)
        # post_init
        #     init_weights
        #         prune_heads
        #         _initialize_weights
        #             _init_weights
        #         tie_weights
        #     _backward_compatibility_gradient_checkpointing
        self.post_init()

        # 先初始化参数, 再读取protein_bert
        # 使用protein_bert编码蛋白
        self.protein_bert = ProteinBERT(
            num_tokens=26,
            dim=128,
            dim_global=512,
            depth=6,
            narrow_conv_kernel=9,
            wide_conv_kernel=9,
            wide_conv_dilation=5,
            attn_heads=4,
            attn_dim_head=64,
            filename="bind_transformer/epoch_92400_sample_23500000.pkl",
        )

    def _init_weights(self, module):
        # 从roformer抄的初始化方法。增加了self.generator来固定随机生成。让参数初始化可重复。
        if isinstance(module, nn.Linear) or isinstance(module, EinMix):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range, generator=self.generator
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range, generator=self.generator
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(
        self,
        protein_ids: torch.Tensor,
        second_ids: torch.Tensor,
        dna_ids: torch.Tensor,
        bind: torch.Tensor = None,
        **kwargs,
    ):
        """
        蛋白质和DNA的mask token都是0。
        protein_ids: (batch_size, protein_length).
        second_ids: (batch_size, protein_length).
        DNA_ids: (batch_size, DNA_length + 1). +1是因为DNA开头要加[CLS] mask.
        bind: (batch_size,), 1.0 for binding and 0.0 for not binding.
        output["logit"]: Binary classification logits (batch_size,).
        output["loss"]: If bind is provided, the loss is cross entropy.
        """
        self.accession_input(protein_ids, second_ids, dna_ids, bind)

        # 氨基酸编码
        protein_embs = self.protein_bert(protein_ids)
        protein_embs = self.protein_bert_head(protein_embs)
        # 二级结构编码
        second_embs, second_mask = self.second_encoder(second_ids)
        # DNA编码
        dna_embs = self.dna_encoder(dna_ids, protein_embs, second_embs, second_mask)
        # 分类头
        logits = self.classifier(dna_embs[:, 0, :])

        if bind is not None:
            # 损失函数加上弹性网络正则化，线性层和卷积层只算权重，不算偏置，其它层不考虑
            loss = self.loss_fn(input=logits, target=bind) + self.elastic_net(self)
            return {"logit": logits, "loss": loss}
        return {"logit": logits}

    def accession_input(
        self,
        protein_ids: torch.Tensor,
        second_ids: torch.Tensor,
        dna_ids: torch.Tensor,
        bind: Union[torch.Tensor, None],
    ) -> None:
        # 检查批处理大小是否一致
        assert (
            protein_ids.shape[0] == second_ids.shape[0]
            and protein_ids.shape[0] == dna_ids.shape[0]
            and (bind is None or protein_ids.shape[0] == bind.shape[0])
        ), "batch size is not consistent"
        # 检查蛋白是否太长, protein bert会在蛋白序列开头和结尾增加<start>和<end> token, 所以长度增加了2
        assert (
            protein_ids.shape[-1] <= self.config.max_num_tokens + 2
        ), "protein is too long"
        # 检查二级结构是否太长
        assert second_ids.shape[-1] <= self.config.max_num_tokens, "second is too long"
        # 检查DNA是否太长, # DNA开头有个[CLS] token, 所以最大长度要+1
        assert dna_ids.shape[-1] <= self.config.max_num_tokens + 1, "DNA is too long"
