import jsonargparse
from common_ai import config

from ..dataset import get_dataset
from ..metric import (
    AccuracyMetric,
    BrierScoreMetric,
    F1Metric,
    MatthewsCorrelationMetric,
    PrAucMetric,
    PrecisionMetric,
    RecallMetric,
    RocAucMetric,
)
from .PDBert.model import PDBertModel


def get_config() -> tuple[jsonargparse.ArgumentParser]:
    parser, train_parser, test_parser = config.get_config()

    train_parser.add_function_arguments(
        function=get_dataset,
        nested_key="dataset",
    )

    train_parser.add_argument(
        "--metric",
        nargs="+",
        type=F1Metric
        | AccuracyMetric
        | RecallMetric
        | PrecisionMetric
        | MatthewsCorrelationMetric
        | RocAucMetric
        | PrAucMetric
        | BrierScoreMetric,
        required=True,
        enable_path=True,
    )

    train_parser.add_subclass_arguments(
        baseclass=(PDBertModel,),
        nested_key="model",
    )

    return parser, train_parser, test_parser
