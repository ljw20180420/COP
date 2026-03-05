#!/usr/bin/env python

import os
import pathlib
import re

import numpy as np
import pandas as pd


def random_protein() -> None:
    protein_feature = pd.read_csv("../../protein_feature.csv", header=0).assign(
        sequence=lambda df: [
            "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), len(protein)))
            for protein in df["sequence"]
        ],
        secondary_structure=lambda df: [
            "".join(np.random.choice(list("HBEGIPTS-KZ"), len(second)))
            for second in df["secondary_structure"]
        ],
        zinc_finger="",
        disorder="",
        KRAB="",
    )
    protein_feature.to_csv("../data/random_protein_feature.csv", index=False)


def shuffle_protein() -> None:
    protein_feature = pd.read_csv("../../protein_feature.csv", header=0)
    shuffled_protein = protein_feature[
        ["sequence", "secondary_structure", "zinc_finger", "disorder", "KRAB"]
    ].sample(frac=1.0)
    protein_feature = protein_feature.assign(
        sequence=shuffled_protein["sequence"],
        secondary_structure=shuffled_protein["secondary_structure"],
        zinc_finger=shuffled_protein["zinc_finger"],
        disorder=shuffled_protein["disorder"],
        KRAB=shuffled_protein["KRAB"],
    )
    protein_feature.to_csv("../data/shuffle_protein_feature.csv", index=False)


def mutate_C2H2_protein() -> None:
    protein_feature = pd.read_csv("../../protein_feature.csv", header=0)
    zf_pattern = re.compile(r"..(C)(?:..|....)(C).{12}(H).{3,5}(H)")
    proteins = []
    for protein in protein_feature["sequence"]:
        segs, start = [], 0
        for zf in zf_pattern.finditer(protein):
            segs.append(protein[start : zf.span()[0]])
            mut_zf = (
                protein[zf.regs[0][0] : zf.regs[1][0]]
                + np.random.choice(list("ADEFGHIKLMNPQRSTVWY"), 1).item()
                + protein[zf.regs[1][1] : zf.regs[2][0]]
                + np.random.choice(list("ADEFGHIKLMNPQRSTVWY"), 1).item()
                + protein[zf.regs[2][1] : zf.regs[3][0]]
                + np.random.choice(list("ACDEFGIKLMNPQRSTVWY"), 1).item()
                + protein[zf.regs[3][1] : zf.regs[4][0]]
                + np.random.choice(list("ACDEFGIKLMNPQRSTVWY"), 1).item()
            )
            segs.append(mut_zf)
            start = zf.span()[1]
        segs.append(protein[start:])
        proteins.append("".join(segs))

    protein_feature = protein_feature.assign(sequence=proteins, zinc_finger="")
    protein_feature.to_csv("../data/mutate_C2H2_protein_feature.csv", index=False)


if __name__ == "__main__":
    # change to the script directory
    os.chdir(pathlib.Path(__file__).resolve().parent)
    os.makedirs("../data", exist_ok=True)

    random_protein()
    shuffle_protein()
    mutate_C2H2_protein()
