#!/usr/bin/env python

from common_ai.config import get_app_parser, get_train_parser

from AI.gradio_fn import MyGradioFn

app_parser = get_app_parser()
train_parser = get_train_parser()

cfg = app_parser.parse_path("AI/app_space.yaml")

MyGradioFn(cfg, train_parser).launch()
