import os

import numpy as np
import pandas as pd
import torch
from common_ai.generator import MyGenerator
from common_ai.utils import SeqTokenizer


class DataCollator:
    def __init__(
        self,
        protein_feature: os.PathLike,
        protein_length: int,
        DNA_length: int,
    ):
        self.protein_length = protein_length
        self.DNA_length = DNA_length

        # protein: ACDEFGHIKLMNPQRSTUVWXYosep->0-25
        # ACDEFGHIKLMNPQRSTVWY: amino acids
        # U: Cysteine
        # X: undefined amino acids
        # o: other amino acids
        # s: start token
        # e: end token
        # p: padding token
        self.protein_tokenizer = SeqTokenizer("ACDEFGHIKLMNPQRSTUVWXYosep")

        # second: mHBEGIPTS-KZ->0-11
        # m: mask token
        # HBEGIPTS-: secondary structures
        # K: KRAB
        # Z: C2H2 zinc finger
        self.second_tokenizer = SeqTokenizer("mHBEGIPTS-KZ")

        # DNA: mcACGTN->0-6
        # m: mask token
        # c: CLS token
        # ACGTN: nucleotides
        self.DNA_tokenizer = SeqTokenizer("mcACGTN")

        self.recent_losses = {}

        self.protein_feature = pd.read_csv(protein_feature, header=0, na_filter=False)
        protein_ids = []
        second_ids = []
        for protein, second, zinc_fns, krabs in zip(
            self.protein_feature["sequence"],
            self.protein_feature["secondary_structure"],
            self.protein_feature["zinc_finger"],
            self.protein_feature["KRAB"],
        ):
            assert len(protein) == len(
                second
            ), "protein sequence and secondary structure have diference length"

            zinc_fns = [int(pos) for pos in zinc_fns.split(":")]
            for zf_start, zf_end in zip(zinc_fns[::2], zinc_fns[1::2]):
                second = second[:zf_start] + "Z" * (zf_end - zf_start) + second[zf_end:]
            krabs = [int(pos) for pos in krabs.split(":")]
            for kb_start, kb_end in zip(krabs[::2], krabs[1::2]):
                second = second[:kb_start] + "K" * (kb_end - kb_start) + second[kb_end:]

            if len(protein) <= self.protein_length:
                protein = (
                    "s" + protein + "e" + "p" * (self.protein_length - len(protein))
                )
                second = second + "m" * (self.protein_length - len(protein))
            else:
                zinc_fn_center = np.array(zinc_fns).mean().item()
                protein_start = int(
                    zinc_fn_center * (1 - self.protein_length / len(protein))
                )
                protein = (
                    "s"
                    + protein[protein_start : protein_start + self.protein_length]
                    + "e"
                )
                second = second[protein_start : protein_start + self.protein_length]

            protein_ids.append(self.protein_tokenizer(protein))
            second_ids.append(self.second_tokenizer(second))

        self.protein_feature["protein_id"] = protein_ids
        self.protein_feature["second_id"] = second_ids

    def __call__(
        self, examples: list[dict], output_label: bool, my_generator: MyGenerator
    ):
        protein_ids, second_ids, DNA_ids = [], [], []
        if output_label:
            binds = []
        for example in examples:
            if len(example["DNA"]) >= self.DNA_length:
                DNA_start = (len(example["DNA"]) - self.DNA_length) // 2
                DNA = "c" + example["DNA"][DNA_start : DNA_start + self.DNA_length]
            else:
                DNA = (
                    "c" + example["DNA"] + (self.DNA_length - len(example["DNA"])) * "m"
                )
            DNA_ids.append(self.DNA_tokenizer(DNA))

            protein_ids.append(
                self.protein_feature.loc[
                    self.protein_feature["Entry"] == example["protein"], "protein_id"
                ]
            )
            second_ids.append(
                self.protein_feature.loc[
                    self.protein_feature["Entry"] == example["protein"], "second_id"
                ]
            )

            if output_label:
                binds.append(example["bind"])

        protein_ids = torch.from_numpy(np.stack(protein_ids))
        second_ids = torch.from_numpy(np.stack(second_ids))
        DNA_ids = torch.from_numpy(np.stack(DNA_ids))
        if output_label:
            return {
                "input": {
                    "protein_id": protein_ids,
                    "second_id": second_ids,
                    "DNA_id": DNA_ids,
                },
                "label": {
                    "bind": torch.tensor(binds),
                },
            }
        return {
            "input": {
                "protein_id": protein_ids,
                "second_id": second_ids,
                "DNA_id": DNA_ids,
            },
        }
