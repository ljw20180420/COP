#!/usr/bin/env python

import pathlib

import evaluate
import numpy as np
import pandas as pd
from common_ai.metric import MyMetricAbstract
from sklearn.metrics import average_precision_score


class F1Metric(MyMetricAbstract):
    def __init__(self, threshold: float):
        """F1Metric arguments.

        Args:
            threshold: predicted probability larger than threshold is considered as positive.
        """
        self.threshold = threshold
        self.f1 = evaluate.load(
            (pathlib.Path(__file__).resolve().parent / "metric" / "f1.py").as_posix()
        )
        self.probas = []
        self.binds = []

    def step(self, df: pd.DataFrame, examples: list, batch: dict):
        self.probas.append(df["proba"].to_numpy())
        self.binds.extend(batch["label"]["bind"])

    def epoch(self):
        results = self.f1.compute(
            predictions=np.concatenate(self.probas) > self.threshold,
            references=np.array(self.binds),
        )
        self.probas = []
        self.binds = []
        return results["f1"]


class AccuracyMetric(MyMetricAbstract):
    def __init__(self, threshold: float):
        """AccuracyMetric arguments.

        Args:
            threshold: predicted probability larger than threshold is considered as positive.
        """
        self.threshold = threshold
        self.accuracy = evaluate.load(
            (
                pathlib.Path(__file__).resolve().parent / "metric" / "accuracy.py"
            ).as_posix()
        )
        self.probas = []
        self.binds = []

    def step(self, df: pd.DataFrame, examples: list, batch: dict):
        self.probas.append(df["proba"].to_numpy())
        self.binds.extend(batch["label"]["bind"])

    def epoch(self):
        results = self.accuracy.compute(
            predictions=np.concatenate(self.probas) > self.threshold,
            references=np.array(self.binds),
        )
        self.probas = []
        self.binds = []
        return results["accuracy"]


class RecallMetric(MyMetricAbstract):
    def __init__(self, threshold: float):
        """RecallMetric arguments.

        Args:
            threshold: predicted probability larger than threshold is considered as positive.
        """
        self.threshold = threshold
        self.recall = evaluate.load(
            (
                pathlib.Path(__file__).resolve().parent / "metric" / "recall.py"
            ).as_posix()
        )
        self.probas = []
        self.binds = []

    def step(self, df: pd.DataFrame, examples: list, batch: dict):
        self.probas.append(df["proba"].to_numpy())
        self.binds.extend(batch["label"]["bind"])

    def epoch(self):
        results = self.recall.compute(
            predictions=np.concatenate(self.probas) > self.threshold,
            references=np.array(self.binds),
        )
        self.probas = []
        self.binds = []
        return results["recall"]


class PrecisionMetric(MyMetricAbstract):
    def __init__(self, threshold: float):
        """PrecisionMetric arguments.

        Args:
            threshold: predicted probability larger than threshold is considered as positive.
        """
        self.threshold = threshold
        self.precision = evaluate.load(
            (
                pathlib.Path(__file__).resolve().parent / "metric" / "precision.py"
            ).as_posix()
        )
        self.probas = []
        self.binds = []

    def step(self, df: pd.DataFrame, examples: list, batch: dict):
        self.probas.append(df["proba"].to_numpy())
        self.binds.extend(batch["label"]["bind"])

    def epoch(self):
        results = self.precision.compute(
            predictions=np.concatenate(self.probas) > self.threshold,
            references=np.array(self.binds),
        )
        self.probas = []
        self.binds = []
        return results["precision"]


class MatthewsCorrelationMetric(MyMetricAbstract):
    def __init__(self, threshold: float):
        """MatthewsCorrelationMetric arguments.

        Args:
            threshold: predicted probability larger than threshold is considered as positive.
        """
        self.threshold = threshold
        self.matthews_correlation = evaluate.load(
            (
                pathlib.Path(__file__).resolve().parent
                / "metric"
                / "matthews_correlation.py"
            ).as_posix()
        )
        self.probas = []
        self.binds = []

    def step(self, df: pd.DataFrame, examples: list, batch: dict):
        self.probas.append(df["proba"].to_numpy())
        self.binds.extend(batch["label"]["bind"])

    def epoch(self):
        results = self.matthews_correlation.compute(
            predictions=np.concatenate(self.probas) > self.threshold,
            references=np.array(self.binds),
        )
        self.probas = []
        self.binds = []
        return results["matthews_correlation"]


class RocAucMetric(MyMetricAbstract):
    def __init__(self):
        self.roc_auc = evaluate.load(
            (
                pathlib.Path(__file__).resolve().parent / "metric" / "roc_auc.py"
            ).as_posix()
        )
        self.probas = []
        self.binds = []

    def step(self, df: pd.DataFrame, examples: list, batch: dict):
        self.probas.append(df["proba"].to_numpy())
        self.binds.extend(batch["label"]["bind"])

    def epoch(self):
        results = self.roc_auc.compute(
            predictions=np.concatenate(self.probas),
            references=np.array(self.binds),
        )
        self.probas = []
        self.binds = []
        return results["roc_auc"]


class PrAucMetric(MyMetricAbstract):
    def __init__(self):
        self.probas = []
        self.binds = []

    def step(self, df: pd.DataFrame, examples: list, batch: dict):
        self.probas.append(df["proba"].to_numpy())
        self.binds.extend(batch["label"]["bind"])

    def epoch(self):
        results = average_precision_score(
            y_true=np.array(self.binds),
            y_score=np.concatenate(self.probas),
        )
        self.probas = []
        self.binds = []
        return results


class BrierScoreMetric(MyMetricAbstract):
    def __init__(self):
        self.brier_score = evaluate.load(
            (
                pathlib.Path(__file__).resolve().parent / "metric" / "brier_score.py"
            ).as_posix()
        )
        self.probas = []
        self.binds = []

    def step(self, df: pd.DataFrame, examples: list, batch: dict):
        self.probas.append(df["proba"].to_numpy())
        self.binds.extend(batch["label"]["bind"])

    def epoch(self):
        results = self.brier_score.compute(
            predictions=np.concatenate(self.probas),
            references=np.array(self.binds),
        )
        self.probas = []
        self.binds = []
        return results["brier_score"]


# Download all metrics from huggingface.
if __name__ == "__main__":
    import os
    import pathlib

    from huggingface_hub import HfFileSystem

    # change directory to the current script
    os.chdir(pathlib.Path(__file__).resolve().parent)

    fs = HfFileSystem()
    for metric in [
        "accuracy",
        "recall",
        "precision",
        "f1",
        "matthews_correlation",
        "confusion_matrix",
        "roc_auc",
        "brier_score",
    ]:
        with fs.open(f"spaces/evaluate-metric/{metric}/{metric}.py", "rb") as rd, open(
            f"metric/{metric}.py", "wb"
        ) as wd:
            wd.write(rd.read())
