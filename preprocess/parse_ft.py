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
        right=pd.read_csv("secondary_structure.csv", header=0, na_filter=False),
        left_on="Entry",
        right_on="accession",
        how="left",
    )
    .drop(columns="accession")
    .fillna("")
)


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

df.to_csv("protein.csv", header=True, index=False)
