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
        minimal_unbind_summit_distance: int,
        select_worst_loss_ratio: float,
    ):
        df = pd.read_csv(
            os.fspath(protein_feature),
            header=0,
            na_filter=False,
        )
        self.protein_length = protein_length
        self.DNA_length = DNA_length
        self.minimal_unbind_summit_distance = minimal_unbind_summit_distance
        self.select_worst_loss_ratio = select_worst_loss_ratio

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

        self.protein_ids = []
        self.second_ids = []
        for protein, second, zinc_fns, krabs in zip(
            df["sequence"], df["secondary_structure"], df["zinc_finger"], df["KRAB"]
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

            self.protein_ids.append(self.protein_tokenizer(protein))
            self.second_ids.append(self.second_tokenizer(second))

    def __call__(
        self, examples: list[dict], output_label: bool, my_generator: MyGenerator
    ):
        protein_ids, second_ids, DNA_ids = [], [], []
        if output_label:
            rns, actual_indices, binds = [], [], []
        for example in examples:
            if len(example["DNA"]) >= self.DNA_length:
                DNA_start = (len(example["DNA"]) - self.DNA_length) // 2
                DNA = "c" + example["DNA"][DNA_start : DNA_start + self.DNA_length]
            else:
                DNA = (
                    "c" + example["DNA"] + (self.DNA_length - len(example["DNA"])) * "m"
                )
            DNA_ids.append(self.DNA_tokenizer(DNA))

            if "protein" not in example:
                # for training, protein is not given
                actual_index = None
                if my_generator.np_rng.random() < 0.5:
                    unbind_indices = [
                        index
                        for index, distance in enumerate(example["distance"].split(":"))
                        if int(distance) == -1
                        or int(distance) >= self.minimal_unbind_summit_distance
                    ]
                    if len(unbind_indices) > 0:
                        if my_generator.np_rng.random() > self.select_worst_loss_ratio:
                            actual_index = my_generator.np_rng.choice(unbind_indices)
                        else:
                            unbind_recent_losses = np.array(
                                [
                                    self.recent_losses.get(
                                        (example["rn"], unbind_index), np.inf
                                    )
                                    for unbind_index in unbind_indices
                                ]
                            )
                            actual_index = unbind_indices[
                                my_generator.np_rng.choice(
                                    np.where(
                                        unbind_recent_losses
                                        == unbind_recent_losses.max()
                                    )[0]
                                ).item()
                            ]
                        bind = 0.0
                if actual_index is None:
                    actual_index = example["index"]
                    bind = 1.0
                protein_ids.append(self.protein_ids[actual_index])
                second_ids.append(self.second_ids[actual_index])
            else:
                # TODO: need to prepare secondary structure, zinc finger, KRAB for all mouse C2H2 zinc finger protein for evaluation
                # For evaluation, the protein is given.
                assert "second" in example, "need secondary structure to evaluate"
                if len(example["protein"]) > self.protein_length:
                    protein_start = (len(example["protein"]) - self.protein_length) // 2
                    protein_fix = (
                        "s"
                        + example["protein"][
                            protein_start : protein_start + self.protein_length
                        ]
                        + "e"
                    )
                else:
                    protein_fix = (
                        "s"
                        + example["protein"]
                        + "e"
                        + "p" * (self.protein_length - len(example["protein"]))
                    )
                protein_ids.append(self.protein_bert_tokenizer(protein_fix))

            if output_label:
                rns.append(example["rn"])
                actual_indices.append(actual_index)
                binds.append(bind)

        protein_ids = torch.from_numpy(np.stack(protein_ids))
        DNA_ids = torch.from_numpy(np.stack(DNA_ids))
        if output_label:
            return {
                "input": {
                    "protein_id": protein_ids,
                    "DNA_id": DNA_ids,
                },
                "label": {
                    "rn": rns,
                    "actual_index": actual_indices,
                    "bind": binds,
                },
            }
        return {
            "input": {
                "protein_id": protein_ids,
                "DNA_id": DNA_ids,
            },
        }
