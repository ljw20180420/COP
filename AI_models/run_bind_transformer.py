#!/usr/bin/env python

import os

# 把运行文件夹切换为脚本文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from bind_transformer.config import get_config, get_logger

args = get_config(
    [
        "bind_transformer/config_default.ini",
        "bind_transformer/config_custom.ini",
    ]
)
logger = get_logger(args.log_level)

if args.command == "download":
    from bind_transformer.metric import download_metrics

    download_metrics()
    exit(0)

from datasets import load_dataset
from bind_transformer.load_data import train_validation_test_split

logger.info("loading data")
ds_protein = load_dataset(
    "csv",
    data_files=(args.data_dir / "protein_data.csv").as_posix(),
    column_names=["accession", "protein", "second", "zinc_num"],
)["train"]

if args.command == "train":
    ds = load_dataset("json", data_dir=args.data_dir / "DNA_data")

    ds = train_validation_test_split(
        ds, args.validation_ratio, args.test_ratio, args.seed
    )

    from bind_transformer.train import train

    train(
        ds,
        ds_protein["protein"],
        ds_protein["second"],
        ds_protein["zinc_num"],
        args.train_output_dir,
        args.seed,
        args.device,
        args.fp16,
        logger,
        args.batch_size,
        args.dna_length,
        args.minimal_unbind_summit_distance,
        args.select_worst_neg_loss_ratio,
        args.optimizer,
        args.learning_rate,
        args.beta1,
        args.beta2,
        args.epsilon,
        args.scheduler,
        args.num_epochs,
        args.warmup_ratio,
        args.protein_vocab,
        args.second_vocab,
        args.dna_vocab,
        args.max_num_tokens,
        args.dim_emb,
        args.num_heads,
        args.dim_heads,
        args.depth,
        args.dim_ffn,
        args.dropout,
        args.norm_eps,
        args.pos_weight,
        args.reg_l1,
        args.reg_l2,
        args.initializer_range,
        args.use_hyena,
        args.hyena_order,
        args.hyena_filter_order,
        args.hp_study_name,
        args.hp_storage,
        args.redundant_parameters,
        args.n_trials,
    )

elif args.command == "test":
    ds = load_dataset(
        "json",
        data_dir=args.data_dir / "DNA_data",
    )
    ds = train_validation_test_split(
        ds, args.validation_ratio, args.test_ratio, args.seed
    )
    from bind_transformer.test import test

    test(
        ds,
        ds_protein["protein"],
        ds_protein["second"],
        ds_protein["zinc_num"],
        args.train_output_dir,
        args.pipeline_output_dir,
        args.device,
        logger,
        args.batch_size,
        args.dna_length,
        args.minimal_unbind_summit_distance,
        args.max_num_tokens,
        args.seed,
    )

elif args.command == "inference":
    ds = load_dataset(
        "csv",
        data_dir=args.inference_data_dir / "DNA_data",
    )["train"]
    from bind_transformer.inference import inference
    from datasets import Dataset
    import sys

    header = True
    for input, output in zip(
        ds.iter(args.batch_size),
        inference(
            ds,
            ds_protein["protein"],
            ds_protein["second"],
            ds_protein["zinc_num"],
            args.pipeline_output_dir,
            args.device,
            logger,
            args.batch_size,
            args.dna_length,
            args.minimal_unbind_summit_distance,
            args.max_num_tokens,
            args.seed,
        ),
    ):
        input["bind"] = output
        Dataset.from_dict(input).to_csv(sys.stdout.buffer, header=header)
        header = False

elif args.command == "app":
    from bind_transformer.app import app

    app(
        args.pipeline_output_dir,
        args.device,
        logger,
        args.dna_length,
        args.max_num_tokens,
    )
