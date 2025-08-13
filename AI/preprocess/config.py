import jsonargparse
from .dataset import get_dataset
from .metric import MyBCELoss
from .PDBert.model import PDBertConfig
from common_ai import config


def get_config() -> tuple[jsonargparse.ArgumentParser]:
    parser, train_parser, test_parser = config.get_config()

    train_parser.add_function_arguments(
        function=get_dataset,
        nested_key="dataset",
    )

    train_parser.add_argument(
        "--metric",
        nargs="+",
        type=MyBCELoss,
        required=True,
        enable_path=True,
    )

    train_parser.add_subclass_arguments(
        baseclass=(PDBertConfig,),
        nested_key="model",
    )

    return parser, train_parser, test_parser
