import os
import numpy as np
import pandas as pd
import torch
from common_ai.utils import SeqTokenizer
from common_ai.generator import MyGenerator


class DataCollator:
    preprocess = "PDBert"

    def __init__(
        self,
        protein_data: os.PathLike,
        protein_length: int,
        DNA_length: int,
        minimal_unbind_summit_distance: int,
        select_worst_loss_ratio: float,
    ):
        df = pd.read_csv(
            os.fspath(protein_data),
            header=0,
            na_filter=False,
        )
        self.protein_length = protein_length
        self.DNA_length = DNA_length
        self.minimal_unbind_summit_distance = minimal_unbind_summit_distance
        self.select_worst_loss_ratio = select_worst_loss_ratio
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
        self.recent_losses = {}

        self.protein_ids = []
        for protein, zinc_fn in zip(df["sequence"], df["zinc_finger"]):
            if len(protein) <= self.protein_length:
                protein_fix = (
                    "s" + protein + "e" + "p" * (self.protein_length - len(protein))
                )
            else:
                zinc_fn_center = int(
                    np.array([int(pos) for pos in zinc_fn.split(":")]).mean().item()
                )
                protein_start = int(
                    zinc_fn_center * (1 - self.protein_length / len(protein))
                )
                protein_fix = (
                    "s"
                    + protein[protein_start : protein_start + self.protein_length]
                    + "e"
                )

            self.protein_ids.append(self.protein_bert_tokenizer(protein_fix))

    def __call__(
        self, examples: list[dict], output_label: bool, my_generator: MyGenerator
    ):
        protein_ids, DNA_ids = [], []
        if output_label:
            rns, actual_indices, binds = [], [], []
        for example in examples:
            if len(example["DNA"]) >= self.DNA_length:
                DNA_start = (len(example["DNA"]) - self.DNA_length) // 2
                DNA_fix = (
                    "s" + example["DNA"][DNA_start : DNA_start + self.DNA_length] + "e"
                )
            else:
                DNA_fix = (
                    "s"
                    + example["DNA"]
                    + "e"
                    + (self.DNA_length - len(example["DNA"])) * "p"
                )
            DNA_ids.append(self.DNA_bert_tokenizer(DNA_fix))

            if "protein" not in example:
                # For training, protein is not given.
                actual_index = None
                if my_generator.np_rng.random() < 0.5:
                    unbind_indices = [
                        index
                        for index, distance in enumerate(example["distance"].split(":"))
                        if int(distance) == -1
                        or distance >= self.minimal_unbind_summit_distance
                    ]
                    if len(unbind_indices) > 0:
                        if my_generator.np_rng.random() < self.select_worst_loss_ratio:
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
            else:
                # For evaluation, the protein is given.
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
