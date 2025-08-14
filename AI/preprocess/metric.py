#!/usr/bin/env python

import pandas as pd
import numpy as np
import evaluate
from sklearn.metrics import average_precision_score


class F1Metric:
    def __init__(self, threshold):
        self.threshold = threshold
        self.f1 = evaluate.load("AI/preprocess/metrics/f1.py")
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


class AccuracyMetric:
    def __init__(self, threshold):
        self.threshold = threshold
        self.accuracy = evaluate.load("AI/preprocess/metrics/accuracy.py")
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


class RecallMetric:
    def __init__(self, threshold):
        self.threshold = threshold
        self.recall = evaluate.load("AI/preprocess/metrics/recall.py")
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


class PrecisionMetric:
    def __init__(self, threshold):
        self.threshold = threshold
        self.precision = evaluate.load("AI/preprocess/metrics/precision.py")
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


class MatthewsCorrelationMetric:
    def __init__(self, threshold):
        self.threshold = threshold
        self.matthews_correlation = evaluate.load(
            "AI/preprocess/metrics/matthews_correlation.py"
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


class RocAucMetric:
    def __init__(self):
        self.roc_auc = evaluate.load("AI/preprocess/metrics/roc_auc.py")
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


class PrAucMetric:
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


class BrierScoreMetric:
    def __init__(self):
        self.brier_score = evaluate.load("AI/preprocess/metrics/brier_score.py")
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


if __name__ == "__main__":
    import os
    import pathlib
    from huggingface_hub import HfFileSystem

    # change directory to the current script
    os.chdir(pathlib.Path(__file__).parent)

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
            f"metrics/{metric}.py", "wb"
        ) as wd:
            wd.write(rd.read())
