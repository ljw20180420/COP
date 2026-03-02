#!/usr/bin/env python

import os
import pathlib

import numpy as np
import pandas as pd
from common_ai.config import get_config, get_train_parser
from common_ai.hpo import MyHpo
from common_ai.hta import MyHta
from common_ai.test import MyTest
from common_ai.train import MyTrain

from AI.gradio_fn import MyGradioFn
from AI.inference import MyInference
from AI.shap import MyShap

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
