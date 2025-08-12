#!/usr/bin/env python

import os

# 把运行文件夹切换为脚本文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import re
import pandas as pd
import numpy as np

df = pd.read_table(
    "uniprot_mouse_C2H2_protein.tsv",
    header=0,
    usecols=["Entry", "Reviewed", "Entry Name", "Zinc finger", "Domain [CC]", "Region"],
    na_filter=False,
)

df = (
    df.merge(
        right=pd.read_table("secondary_structure.tsv", header=0, na_filter=False),
        left_on="Entry",
        right_on="accession",
        how="left",
    )
    .drop(columns="accession")
    .fillna("")
)

# 去掉没有结构的蛋白和长度不符合的蛋白
df = df.loc[df["Length"] == df["sequence"].str.len()].reset_index(drop=True)


def parse_intervals(reg: str, literals: pd.Series):
    intervalss = []
    for literal in literals:
        intervals = ""
        for mat in reg.finditer(literal):
            intervals += mat.group(1) + ":" + mat.group(2) + ":"
        intervalss.append(intervals.rstrip(":"))
    return intervalss


df["zinc_finger"] = parse_intervals(
    re.compile(r'ZN_FING (\d+)\.\.(\d+); /note="C2H2'), df.pop("Zinc finger")
)
df["disorder"] = parse_intervals(
    re.compile(r'REGION (\d+)\.\.(\d+); /note="Disordered"'), df.pop("Region")
)
df["KRAB"] = parse_intervals(
    re.compile(r'DOMAIN (\d+)\.\.(\d+); /note="KRAB"'), df.pop("Domain [CC]")
)

secondary_structures = []
for zinc_fingers, KRABs, secondary_structure in zip(
    df["zinc_finger"], df["KRAB"], df["secondary_structure"]
):
    secondary_structure_array = np.array(list(secondary_structure))
    for zinc_finger in zinc_fingers:
        secondary_structure_array[zinc_finger[0] : zinc_finger[1]] = "Z"
    for KRAB in KRABs:
        secondary_structure_array[KRAB[0] : KRAB[1]] = "K"

    secondary_structures.append("".join(secondary_structure_array))

df["secondary_structure"] = secondary_structures

with open(3, "w") as fd:
    df.loc[
        :, ["Entry", "Reviewed", "Entry Name", "sequence", "secondary_structure"]
    ].to_csv(fd, sep="\t", header=True, index=False)
