#!/usr/bin/env python

import os
import pathlib

import pandas as pd
from common_ai.config import get_config
from common_ai.test import MyTest
from common_ai.train import MyTrain

from AI.inference import MyInference

# change directory to the current script
os.chdir(pathlib.Path(__file__).parent)

# parse arguments
(
    parser,
    train_parser,
    test_parser,
    infer_parser,
    explain_parser,
    app_parser,
    hta_parser,
    hpo_parser,
) = get_config()
cfg = parser.parse_args()

if cfg.subcommand == "train":
    for epoch in MyTrain(**cfg.train.train.as_dict())(train_parser):
        pass

elif cfg.subcommand == "test":
    epoch = MyTest(**cfg.test.as_dict())(train_parser)

elif cfg.subcommand == "infer":
    params = (
        {}
        if not hasattr(cfg.infer.inference, "init_args")
        else cfg.infer.inference.init_args.as_dict()
    )
    MyInference(**params)(
        infer_df=pd.read_csv(cfg.infer.input),
        test_cfg=cfg.infer.test,
        train_parser=train_parser,
    ).to_csv(cfg.infer.output, index=False)
