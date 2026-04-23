#!/usr/bin/env python

import os
import pathlib
import sys

import numpy as np
import pandas as pd

small_line_num = int(sys.argv[1])
seed = int(sys.argv[2])
accessions = sys.argv[3:]

rng = np.random.default_rng(seed)
entries = pd.read_csv("protein_feature.csv", header=0)["Entry"].to_list()
accessions_not_in_entries = [
    accession for accession in accessions if accession not in entries
]
dfs = []
for accession in accessions:
    if accession in accessions_not_in_entries:
        print(f"{accession} has inconsistent uniprot and alphafoldDB sequence")
        continue
    print(f"sample {accession}")
    df = pd.read_csv(
        pathlib.Path(os.environ["DATA_DIR"]) / "train_data" / f"{accession}.csv",
        header=0,
    ).drop(columns=accessions_not_in_entries)
    dfs.append(df.sample(n=min(small_line_num, df.shape[0]), random_state=rng))
pd.concat(dfs).to_csv(f"S{small_line_num}_data.csv", index=False)
