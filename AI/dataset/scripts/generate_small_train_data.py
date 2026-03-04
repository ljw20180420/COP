#!/usr/bin/env python

import os
import pathlib
import sys

import pandas as pd

small_line_num = int(sys.argv[1])
accessions = sys.argv[2:]
dfs = []
for accession in accessions:
    dfs.append(
        pd.read_csv(
            pathlib.Path(os.environ["DATA_DIR"]) / "train_data" / f"{accession}.csv",
            header=0,
        ).sample(n=small_line_num)
    )
pd.concat(dfs).reset_index(drop=True).reset_index(names=["rn"]).to_csv(
    "small_DNA_data.csv", index=False
)
