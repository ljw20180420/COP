#!/usr/bin/env python

import os
import pathlib

import jsonargparse
import pandas as pd
from common_ai.config import get_config, get_train_parser
from common_ai.hpo import MyHpo
from common_ai.test import MyTest
from common_ai.train import MyTrain
from common_ai.utils import reproduce

from AI.inference import MyInference

# change directory to the current script
os.chdir(pathlib.Path(__file__).parent)

# improve reproducibility
reproduce()

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
    MyInference(
        **cfg.infer.inference.get("init_args", jsonargparse.Namespace()).as_dict()
    )(
        infer_df=pd.read_csv(cfg.infer.input),
        test_cfg=cfg.infer.test,
        train_parser=train_parser,
    ).to_csv(
        cfg.infer.output, index=False
    )

elif cfg.subcommand == "hpo":
    MyHpo(**cfg.hpo.hpo.as_dict())(hpo_parser, get_train_parser)
