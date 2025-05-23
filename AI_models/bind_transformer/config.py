import configargparse
import pathlib
import logging
import sys
import yaml


def get_config(config_files):
    """
    config_files: Files contain hyper-parameters. The later config files will override the former ones.
    For example, if  config_files=['AI_models/bind_transformer/config_default.ini', 'AI_models/bind_transformer/config_custom.ini'], then settings in config_custom.ini will override settings in config_default.ini. A good practice is to put default settings in config_default.ini (do not modify config_default.ini), and then override default behaviors in config_custom.ini.
    """
    parser = configargparse.ArgumentParser(
        description="Arguments for transcriptor binding roformer model.",
        default_config_files=config_files,
        auto_env_var_prefix="BIND_TRANSFORMER_",
        config_file_parser_class=configargparse.ConfigparserConfigFileParser,
    )

    # command parameters
    parser_command = parser.add_argument_group(
        title="command", description="Command parameters."
    )
    parser_command.add_argument(
        "--command",
        type=str,
        required=True,
        choices=["download", "train", "test", "inference", "app"],
        help="What to do.",
    )

    # common parameters
    parser_common = parser.add_argument_group(
        title="common", description="Common parameters."
    )
    parser_common.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Input directory contains csv files with header protein,secondary_structure,DNA,bind. Used for train and test.",
    )
    parser_common.add_argument(
        "--train_output_dir",
        type=pathlib.Path,
        required=True,
        help="Output directory of training process, which contains model checkpoints for epochs.",
    )
    parser_common.add_argument(
        "--pipeline_output_dir",
        type=pathlib.Path,
        required=True,
        help="Output directory to save pipeline.",
    )
    parser_common.add_argument("--seed", type=int, required=True, help="random seed")
    parser_common.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["cpu", "cuda"],
        help="Device for computation (cuda or cpu). If not specified, use cuda if available",
    )
    parser_common.add_argument(
        "--log_level",
        type=str,
        required=True,
        choices=["CRITICAL", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        help="Set logging level.",
    )
    parser_common.add_argument(
        "--inference_data_dir",
        type=pathlib.Path,
        required=True,
        help="Input directory contains csv files with header protein,secondary_structure,DNA,bind. Used for inference.",
    )
    parser_common.add_argument(
        "--fp16",
        type=bool,
        required=True,
        help="Whether to use fp16.",
    )

    # dataset parameters
    parser_dataset = parser.add_argument_group(
        title="dataset", description="Parameters for loading and split dataset."
    )
    parser_dataset.add_argument(
        "--validation_ratio",
        type=float,
        required=True,
        help="Proportion for validation samples.",
    )
    parser_dataset.add_argument(
        "--test_ratio", type=float, required=True, help="Proportion for test samples."
    )

    # data loader parameters
    parser_data_loader = parser.add_argument_group(
        title="data loader", description="Parameters for data loader."
    )
    parser_data_loader.add_argument(
        "--dna_length",
        type=int,
        required=True,
        help="DNA of this length will be extracted from the input if possible. If set to 0, will determined the length by the number of zinc finger of the protein.",
    )
    parser_data_loader.add_argument(
        "--batch_size", type=int, required=True, help="Batch size."
    )
    parser_data_loader.add_argument(
        "--minimal_unbind_summit_distance",
        type=int,
        required=True,
        help="Minimal distance between summit such that the protein is considered not bind to the target peak.",
    )
    parser_data_loader.add_argument(
        "--select_worst_neg_loss_ratio",
        type=float,
        required=True,
        help="The ratio of the negative examples select with the worest loss.",
    )

    # optimizer parameters
    parser_optimizer = parser.add_argument_group(
        title="optimizer", description="Parameters for optimizer."
    )
    parser_optimizer.add_argument(
        "--optimizer",
        type=str,
        required=True,
        choices=[
            "adamw_torch",
            "adamw_torch_fused",
            "adafactor",
        ],
        help="Optimizer for training.",
    )
    parser_optimizer.add_argument(
        "--learning_rate", type=float, required=True, help="Learn rate of training."
    )
    parser_optimizer.add_argument(
        "--beta1", type=float, required=True, help="beta1 for AdamW."
    )
    parser_optimizer.add_argument(
        "--beta2", type=float, required=True, help="beta2 for AdamW."
    )
    parser_optimizer.add_argument(
        "--epsilon", type=float, required=True, help="epsilon for AdamW."
    )

    # scheduler parameters
    parser_scheduler = parser.add_argument_group(
        title="scheduler", description="Parameters for learning rate scheduler."
    )
    parser_scheduler.add_argument(
        "--scheduler",
        type=str,
        required=True,
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
            "reduce_lr_on_plateau",
            "cosine_with_min_lr",
            "warmup_stable_decay",
        ],
        help="The learning rate scheduler to use.",
    )
    parser_scheduler.add_argument(
        "--num_epochs",
        type=float,
        required=True,
        help="Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).",
    )
    parser_scheduler.add_argument(
        "--warmup_ratio",
        type=float,
        required=True,
        help="Ratio of total training steps used for a linear warmup from 0 to learning_rate",
    )

    # roformer parameters
    parser_roformer = parser.add_argument_group(
        title="roformer", description="Parameters for roformer."
    )
    parser_roformer.add_argument(
        "--protein_vocab",
        type=int,
        required=True,
        help="The vocabulary size of protein animo acids. 20 animo acids and 1 mask token, totally 21.",
    )
    parser_roformer.add_argument(
        "--second_vocab",
        type=int,
        required=True,
        help="The vocabulary size of protein secondary structure. 11 secondary structrue and 1 mask token, totally 12.",
    )
    parser_roformer.add_argument(
        "--dna_vocab",
        type=int,
        required=True,
        help="The vocabulary size of DNA. 4 nucletides and 1 mask token and 1 [CLS] token, totally 6.",
    )
    parser_roformer.add_argument(
        "--max_num_tokens",
        type=int,
        required=True,
        help="The maximum length of DNA and protein length, needed by rotatory position embedding.",
    )
    parser_roformer.add_argument(
        "--dim_emb", type=int, required=True, help="Model embedding dimension."
    )
    parser_roformer.add_argument(
        "--dim_heads",
        type=int,
        required=True,
        help="Dimension of attention heads.",
    )
    parser_roformer.add_argument(
        "--num_heads",
        type=int,
        required=True,
        help="Number of attention heads.",
    )
    parser_roformer.add_argument(
        "--depth", type=int, required=True, help="Number of EncoderLayer."
    )
    parser_roformer.add_argument(
        "--dim_ffn",
        type=int,
        required=True,
        help="FeedForward intermediate dimension size.",
    )
    parser_roformer.add_argument(
        "--dropout",
        type=float,
        required=True,
        help="The dropout probability.",
    )
    parser_roformer.add_argument(
        "--norm_eps",
        type=float,
        required=True,
        help="The epsilon used by the normalization layers.",
    )
    parser_roformer.add_argument(
        "--pos_weight",
        type=float,
        required=True,
        help="Weight for positive samples (https://www.tensorflow.org/tutorials/structured_data/imbalanced_data).",
    )
    parser_roformer.add_argument(
        "--reg_l1",
        type=float,
        required=True,
        help="l1 weight decay.",
    )
    parser_roformer.add_argument(
        "--reg_l2",
        type=float,
        required=True,
        help="l2 weight decay.",
    )
    parser_roformer.add_argument(
        "--initializer_range",
        type=float,
        required=True,
        help="The standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
    )

    # hyena parameters
    parser_hyena = parser.add_argument_group(
        title="hyena", description="Parameters for hyena."
    )
    parser_hyena.add_argument(
        "--use_hyena",
        type=bool,
        required=True,
        help="Whether replace self-attention by hyena.",
    )
    parser_hyena.add_argument(
        "--hyena_order",
        type=int,
        required=True,
        help="Order of hyena.",
    )
    parser_hyena.add_argument(
        "--hyena_filter_order",
        type=int,
        required=True,
        help="Dimension of hyena MLP hidden layers.",
    )

    # hpo (hyperparameter optimization) parameters
    parser_hpo = parser.add_argument_group(
        title="hpo", description="Hyperparameter optimization."
    )
    parser_hpo.add_argument(
        "--hp_study_name",
        type=str,
        required=True,
        help="The job name of hyperparameter search.",
    )
    parser_hpo.add_argument(
        "--hp_storage",
        type=str,
        required=True,
        help="The sql url for optuna.",
    )
    parser_hpo.add_argument(
        "--redundant_parameters",
        action="append",
        type=yaml.safe_load,
        default=[],
        help="Redundant parameters to apply hyperparameter search.",
    )
    parser_hpo.add_argument(
        "--n_trials", type=int, required=True, help="Number of optuna trials."
    )

    args = parser.parse_args()
    for pr in args.redundant_parameters:
        if pr["type"] == "float":
            pr["low"], pr["high"] = float(pr["low"]), float(pr["high"])
        if pr["type"] == "int":
            pr["low"], pr["high"] = int(pr["low"]), int(pr["high"])

    return args


def get_logger(log_level):
    logger = logging.getLogger("logger")
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(log_level)
    logger.addHandler(handler)
    return logger
