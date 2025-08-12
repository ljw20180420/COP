import numpy as np
import torch
from typing import List
from einops import pack


class DNA_Tokenizer:
    """
    DNA: mcACGTN->0123456
    m: mask token
    c: [CLS]token
    ACGT: 碱基token
    N: 未知碱基
    从输入的DNA中截取DNA_length长度的DNA序列, 并在开头加入分类token c
    如果DNA_length <= 0, 使用10 + 3 * zinc_num作为DNA_length
    """

    def __init__(self, dna_length: int) -> None:
        self.dna_length = dna_length

        self.dna_tokenmap = torch.zeros(110, dtype=torch.int64)
        self.dna_tokenmap[
            np.frombuffer("mcACGTN".encode(), dtype=np.int8).astype(np.int64)
        ] = torch.arange(7)

    def __call__(self, dnas: List[str], zinc_nums: List[int]) -> torch.Tensor:
        extracted_dnas, max_length = [], 0
        for dna, zinc_num in zip(dnas, zinc_nums):
            dna_length = self.dna_length if self.dna_length > 0 else 10 + 3 * zinc_num
            assert len(dna) >= dna_length, "DNA太短"
            start = (len(dna) - dna_length) // 2
            extracted_dna = "c" + dna[start : start + dna_length]
            max_length = (
                max_length if max_length >= len(extracted_dna) else len(extracted_dna)
            )
            extracted_dnas.append(extracted_dna)

        return pack(
            [
                self.dna_tokenmap[
                    np.frombuffer(
                        (
                            extracted_dna + "m" * (max_length - len(extracted_dna))
                        ).encode(),
                        dtype=np.int8,
                    ).astype(np.int64)
                ]
                for extracted_dna in extracted_dnas
            ],
            "* n",
        )[0]


class Second_Tokenizer:
    """
    second: mHBEGIPTS-KZ->0-11
    m: mask token
    HBEGIPTS-: 二级结构
    K: KRAB
    Z: 锌指
    """

    def __init__(self, max_num_tokens: int = 0) -> None:
        self.max_length = max_num_tokens
        self.second_tokenmap = torch.zeros(110, dtype=torch.int64)
        self.second_tokenmap[
            np.frombuffer("mHBEGIPTS-KZ".encode(), dtype=np.int8).astype(np.int64)
        ] = torch.arange(12)

    def __call__(self, seconds: List[str]) -> torch.Tensor:
        max_length = (
            self.max_length
            if self.max_length > 0
            else max([len(second) for second in seconds])
        )
        return pack(
            [
                self.second_tokenmap[
                    np.frombuffer(
                        (second + "m" * (max_length - len(second))).encode(),
                        dtype=np.int8,
                    ).astype(np.int64)
                ]
                for second in seconds
            ],
            "* n",
        )[0]


class Protein_Tokenizer:
    """
    protein: mACDEFGHIKLMNPQRSTVWY->0-20
    m: mask token
    ACDEFGHIKLMNPQRSTVWY: 氨基酸
    """

    def __init__(self) -> None:
        self.protein_tokenmap = torch.zeros(110, dtype=torch.int64)
        self.protein_tokenmap[
            np.frombuffer("mACDEFGHIKLMNPQRSTVWY".encode(), dtype=np.int8).astype(
                np.int64
            )
        ] = torch.arange(21)

    def __call__(self, proteins: List[str]) -> torch.Tensor:
        max_length = max([len(protein) for protein in proteins])
        return pack(
            [
                self.protein_tokenmap[
                    np.frombuffer(
                        (protein + "m" * (max_length - len(protein))).encode(),
                        dtype=np.int8,
                    ).astype(np.int64)
                ]
                for protein in proteins
            ],
            "* n",
        )[0]


class Protein_Bert_Tokenizer:
    """
    protein: ACDEFGHIKLMNPQRSTUVWXYosep->0-25
    ACDEFGHIKLMNPQRSTVWY: 氨基酸
    U: 硒半胱氨酸
    X: 未定义氨基酸
    o: 其它氨基酸
    s: 蛋白序列起始位置
    e: 蛋白序列终止位置
    p: pad
    max_length: protein bert不使用mask, 而是用pad. 因此需要固定max_length.
    """

    def __init__(self, max_num_tokens: int) -> None:
        self.max_length = max_num_tokens
        self.protein_tokenmap = torch.zeros(116, dtype=torch.int64)
        self.protein_tokenmap[
            np.frombuffer("ACDEFGHIKLMNPQRSTUVWXYosep".encode(), dtype=np.int8).astype(
                np.int64
            )
        ] = torch.arange(26)

    def __call__(self, proteins: List[str]) -> torch.Tensor:
        return pack(
            [
                self.protein_tokenmap[
                    np.frombuffer(
                        (
                            "s" + protein + "e" + "p" * (self.max_length - len(protein))
                        ).encode(),
                        dtype=np.int8,
                    ).astype(np.int64)
                ]
                for protein in proteins
            ],
            "* n",
        )[0]
