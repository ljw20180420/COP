#!/usr/bin/env python

import os
import pathlib
from AI.preprocess.config import get_config
from AI.preprocess.dataset import get_dataset
from common_ai.train import MyTrain
from common_ai.test import MyTest

# change directory to the current script
os.chdir(pathlib.Path(__file__).parent)

# parse arguments
parser, train_parser, test_parser = get_config()
cfg = parser.parse_args()
if cfg.subcommand == "train":
    dataset = get_dataset(**cfg.train.dataset.as_dict())
    for performance in MyTrain(**cfg.train.train.as_dict())(
        train_parser=train_parser, cfg=cfg.train, dataset=dataset
    ):
        pass

elif cfg.subcommand == "test":
    my_test = MyTest(**cfg.test.test.as_dict())
    best_train_cfg = my_test.get_best_cfg(train_parser=train_parser)
    dataset = get_dataset(**best_train_cfg.dataset.as_dict())
    my_test(cfg=best_train_cfg, dataset=dataset)
