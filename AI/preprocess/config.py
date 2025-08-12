import jsonargparse
from .CRIformer import model as CRIformer_M
from .CRIfuser import model as CRIfuser_M
from .DeepHF import model as DeepHF_M
from .FOREcasT import model as FOREcasT_M
from .inDelphi import model as inDelphi_M
from .Lindel import model as Lindel_M
from .train import MyTrain
from .test import MyTest
from .metric import NonWildTypeCrossEntropy
from .dataset import get_dataset
from .utils import MyGenerator, get_logger


def get_config() -> tuple[jsonargparse.ArgumentParser]:
    parser = jsonargparse.ArgumentParser(
        description="Arguments of AI models.",
    )
    subcommands = parser.add_subcommands(required=True, dest="subcommand")

    test_parser = jsonargparse.ArgumentParser(description="Test AI models.")
    test_parser.add_argument("--config", action="config")
    test_parser.add_class_arguments(theclass=MyTest, nested_key="test")
    subcommands.add_subcommand(name="test", parser=test_parser)

    train_parser = jsonargparse.ArgumentParser(description="Train AI models.")
    train_parser.add_argument("--config", action="config")
    train_parser.add_class_arguments(theclass=MyTrain, nested_key="train")
    train_parser.add_method_arguments(
        theclass=MyTrain,
        themethod="get_initializer",
        nested_key="initializer",
    )
    train_parser.add_method_arguments(
        theclass=MyTrain,
        themethod="get_optimizer",
        nested_key="optimizer",
    )
    train_parser.add_method_arguments(
        theclass=MyTrain,
        themethod="get_lr_scheduler",
        nested_key="lr_scheduler",
    )

    train_parser.add_class_arguments(
        theclass=MyGenerator,
        nested_key="generator",
    )
    train_parser.add_function_arguments(
        function=get_logger,
        nested_key="logger",
    )

    train_parser.add_function_arguments(
        function=get_dataset,
        nested_key="dataset",
    )

    train_parser.add_argument(
        "--metric",
        nargs="+",
        type=NonWildTypeCrossEntropy,
        required=True,
        enable_path=True,
    )

    train_parser.add_subclass_arguments(
        baseclass=(
            CRIformer_M.CRIformerConfig,
            CRIfuser_M.CRIfuserConfig,
            DeepHF_M.DeepHFConfig,
            DeepHF_M.MLPConfig,
            DeepHF_M.CNNConfig,
            DeepHF_M.XGBoostConfig,
            DeepHF_M.RidgeConfig,
            FOREcasT_M.FOREcasTConfig,
            inDelphi_M.inDelphiConfig,
            Lindel_M.LindelConfig,
        ),
        nested_key="model",
    )

    subcommands.add_subcommand(name="train", parser=train_parser)

    return parser, train_parser, test_parser
