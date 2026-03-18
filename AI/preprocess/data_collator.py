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
        dna_length: int,
    ) -> None:
        self.protein_length = protein_length
        self.dna_length = dna_length

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
        self.dna_tokenizer = SeqTokenizer("mcACGTN")

        protein_ids = []
        second_ids = []
        protein_feature = pd.read_csv(protein_feature, header=0, na_filter=False).drop(
            columns=["Reviewed", "Entry Name", "disorder"]
        )
        for protein, second, zinc_fns, krabs in zip(
            protein_feature["sequence"],
            protein_feature["secondary_structure"],
            protein_feature["zinc_finger"],
            protein_feature["KRAB"],
        ):
            assert len(protein) == len(
                second
            ), "protein sequence and secondary structure have diference length"

            zinc_fns = [int(pos) for pos in zinc_fns.split(":")]
            for zf_start, zf_end in zip(zinc_fns[::2], zinc_fns[1::2]):
                second = second[:zf_start] + "Z" * (zf_end - zf_start) + second[zf_end:]
            if krabs:
                krabs = [int(pos) for pos in krabs.split(":")]
                for kb_start, kb_end in zip(krabs[::2], krabs[1::2]):
                    second = (
                        second[:kb_start] + "K" * (kb_end - kb_start) + second[kb_end:]
                    )

            if len(protein) <= self.protein_length:
                protein = (
                    "s" + protein + "e" + "p" * (self.protein_length - len(protein))
                )
                second = second + "m" * (self.protein_length - len(second))
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

        self.protein_ids = pd.DataFrame(
            data=np.stack(protein_ids), index=protein_feature["Entry"]
        )
        self.second_ids = pd.DataFrame(
            data=np.stack(second_ids), index=protein_feature["Entry"]
        )

    def __call__(
        self, examples: list[dict], output_label: bool, my_generator: MyGenerator
    ) -> dict[str, dict]:
        proteins, dna_ids = [], []
        if output_label:
            binds = []
        for example in examples:
            dna = example["DNA"]
            if len(dna) >= self.dna_length:
                dna_start = (len(dna) - self.dna_length) // 2
                dna = "c" + dna[dna_start : dna_start + self.dna_length]
            else:
                dna = "c" + dna + (self.dna_length - len(dna)) * "m"
            dna_ids.append(self.dna_tokenizer(dna))

            proteins.append(example["protein"])

            if output_label:
                binds.append(example["bind"])

        protein_ids = torch.from_numpy(
            self.protein_ids.loc[proteins, :].to_numpy().copy()
        )
        second_ids = torch.from_numpy(
            self.second_ids.loc[proteins, :].to_numpy().copy()
        )
        dna_ids = torch.from_numpy(np.stack(dna_ids))
        if output_label:
            return {
                "input": {
                    "protein_id": protein_ids,
                    "second_id": second_ids,
                    "dna_id": dna_ids,
                },
                "label": {
                    "bind": torch.tensor(binds, dtype=torch.float32),
                },
            }
        return {
            "input": {
                "protein_id": protein_ids,
                "second_id": second_ids,
                "dna_id": dna_ids,
            },
        }
