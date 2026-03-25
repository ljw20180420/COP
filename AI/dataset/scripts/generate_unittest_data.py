#!/usr/bin/env python

import os
import pathlib

import pandas as pd

os.makedirs("balanced_unittest_data", exist_ok=True)

small_dir = pathlib.Path("balanced_small_data")
unit_dir = pathlib.Path("balanced_unittest_data")
for file, size in zip(["train.csv", "validation.csv", "test.csv"], [4000, 400, 400]):
    pd.read_csv(small_dir / file, header=0).sample(n=size, random_state=63036).to_csv(
        unit_dir / file, index=False
    )
