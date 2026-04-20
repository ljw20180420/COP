#!/usr/bin/env python

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common_ai.config import get_app_parser, get_train_parser

from AI.gradio_fn import MyGradioFn

app_parser = get_app_parser()
train_parser = get_train_parser()

cfg = app_parser.parse_path("AI/app_space.yaml")

MyGradioFn(cfg, train_parser).launch()
