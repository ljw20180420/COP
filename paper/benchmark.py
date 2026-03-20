#!/usr/bin/env python

import os
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from tbparse import SummaryReader


def get_test_metric_df(
    preprocess_model_cls_pairs: list[tuple[str, str]],
    data_names: list[str],
    metrics: list[str],
    output_dir: os.PathLike,
) -> pd.DataFrame:
    test_dict = {
        "preprocess": [],
        "model_cls": [],
        "data_name": [],
        "metric": [],
        "value": [],
        "best_epoch": [],
    }

    for preprocess, model_cls in preprocess_model_cls_pairs:
        for data_name in data_names:
            for metric in metrics:
                tb_path = (
                    output_dir
                    / preprocess
                    / model_cls
                    / data_name
                    / "default"
                    / "test"
                    / metric
                )
                df = SummaryReader(tb_path.as_posix()).scalars
                best_epoch, _, value = (
                    df.loc[df["tag"] == f"test/{metric}"].to_numpy().flatten()
                )
                test_dict["preprocess"].append(preprocess)
                test_dict["model_cls"].append(model_cls)
                test_dict["data_name"].append(data_name)
                test_dict["metric"].append(metric)
                test_dict["value"].append(value)
                test_dict["best_epoch"].append(best_epoch)

    test_df = pd.DataFrame(test_dict)

    # save
    os.makedirs("paper/benchmark", exist_ok=True)
    test_df.to_csv("paper/benchmark/default.csv", index=False)
    test_df.query("metric == 'AccuracyMetric'").drop(
        columns=["data_name", "metric", "best_epoch"]
    ).rename(
        columns={
            "model_cls": "model",
            "value": "accuracy",
        }
    ).to_latex(
        "paper/benchmark/default.tex", index=False, escape=True
    )

    return test_df


def draw_benchmark(test_df: pd.DataFrame) -> None:
    for data_name in test_df["data_name"].drop_duplicates().to_list():
        for metric in test_df["metric"].drop_duplicates().to_list():
            ax = (
                test_df.query("data_name == @data_name and metric == @metric")
                .sort_values("value")
                .set_index(
                    keys=[
                        "preprocess",
                        "model_cls",
                        "data_name",
                    ]
                )
                .plot.bar(y="value", figsize=(20, 10))
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha="right")
            ax.get_figure().savefig(f"paper/benchmark/default_{data_name}_{metric}.pdf")


# Swith to non-gui backend (https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop).
plt.switch_backend("agg")
# Editable axis in illustrator (https://stackoverflow.com/questions/54101529/how-can-i-export-a-matplotlib-figure-as-a-vector-graphic-with-editable-text-fiel)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


preprocess_model_cls_pairs = [
    ("LightGBM", "LightGBM"),
    ("XGBoost", "XGBoost"),
    ("XGBoost", "RandomForest"),
    ("XGBoost", "DecisionTree"),
    ("Scikit", "CategoricalNB"),
    ("Scikit", "SGDClassifier"),
    ("Scikit", "Perceptron"),
    ("Scikit", "PassiveAggressiveClassifier"),
]
data_names = ["mouse_C2H2"]
metrics = [
    "F1Metric",
    "AccuracyMetric",
    "RecallMetric",
    "PrecisionMetric",
    "MatthewsCorrelationMetric",
    "RocAucMetric",
    "PrAucMetric",
    "BrierScoreMetric",
]
output_dir = pathlib.Path("/home/ljw/sdc1/MOTIF_results/formal/default/logs")

test_df = get_test_metric_df(
    preprocess_model_cls_pairs, data_names, metrics, output_dir
)
draw_benchmark(test_df)
