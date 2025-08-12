import pandas as pd
from ..utils import SeqTokenizer


class DataCollator:
    def __init__(
        self,
        protein: pd.DataFrame,
        protein_length: int,
        DNA_length: int,
        minimal_unbind_summit_distance: int,
    ):
        self.protein_length = protein_length
        self.DNA_length = DNA_length
        self.minimal_unbind_summit_distance = minimal_unbind_summit_distance
        # protein: ACDEFGHIKLMNPQRSTUVWXYosep->0-25
        # ACDEFGHIKLMNPQRSTVWY: 氨基酸
        # U: 硒半胱氨酸
        # X: 未定义氨基酸
        # o: 其它氨基酸
        # s: 序列起始位置
        # e: 序列终止位置
        # p: pad
        self.protein_bert_tokenizer = SeqTokenizer("ACDEFGHIKLMNPQRSTUVWXYosep")
        # DNA: ACGTNsep -> 0-7
        # s: 序列起始位置
        # e: 序列终止位置
        # p: pad
        self.DNA_bert_tokenizer = SeqTokenizer("ACGTNsep")

    def __call__(self, examples: list[dict], output_label: bool):
        self.protein_bert_tokenizer()
