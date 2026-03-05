#!/usr/bin/env python

import os
import pathlib
import sys

import pandas as pd

small_line_num = int(sys.argv[1])
accessions = sys.argv[2:]
dfs = []
for accession in accessions:
    print(f"sample {accession}")
    df = pd.read_csv(
        pathlib.Path(os.environ["DATA_DIR"]) / "train_data" / f"{accession}.csv",
        header=0,
    )
    dfs.append(df.sample(n=min(small_line_num, df.shape[0])))
pd.concat(dfs).to_csv("small_data.csv", index=False)
