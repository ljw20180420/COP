import os
import re

import pandas as pd
import torch
from common_ai.generator import MyGenerator


class DataCollator:
    def __init__(
        self,
        protein_feature: os.PathLike,
        dna_length: int,
        zf_padding: int,
    ) -> None:
        self.dna_length = dna_length

        prog = re.compile(r"..C(?:..|.{4})C(.{12})H.{3,5}H")
        protein_feature = pd.read_csv(protein_feature, header=0, na_filter=False)[
            ["Entry", "sequence"]
        ]
        self.protein2zf = {}
        for accession, sequence in zip(
            protein_feature["Entry"], protein_feature["sequence"]
        ):
            self.protein2zf[accession] = []
            for result in prog.finditer(sequence):
                zf_start = result.span()[0]
                zf_end = result.span()[1]
                zf_ctx = (
                    sequence[max(0, zf_start - zf_padding) : zf_start]
                    + result.group(0)
                    + sequence[zf_end : zf_end + zf_padding]
                )
                res12 = result.group(1)
                self.protein2zf[accession].append(
                    {
                        "zf_ctx": zf_ctx,
                        "res12": res12,
                    }
                )

    def __call__(
        self, examples: list[dict], output_label: bool, my_generator: MyGenerator
    ) -> dict[str, dict]:
        dnas = []
        if output_label:
            binds = []
        for example in examples:
            dna = example["DNA"]
            if len(dna) >= self.dna_length:
                dna_start = (len(dna) - self.dna_length) // 2
                dna = "c" + dna[dna_start : dna_start + self.dna_length]
            else:
                dna = "c" + dna + (self.dna_length - len(dna)) * "m"
            dnas.append(dna)

            if output_label:
                binds.append(example["bind"])

        if output_label:
            return {
                "input": {
                    "dna": dnas,
                },
                "label": {
                    "bind": torch.tensor(binds, dtype=torch.float32),
                },
            }
        return {
            "input": {
                "dna": dnas,
            },
        }
