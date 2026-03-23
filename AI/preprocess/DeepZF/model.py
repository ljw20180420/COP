import io
import os
import pathlib
import pickle
import re
import subprocess
import tempfile

import evaluate
import jsonargparse
import numpy as np
import optuna
import pandas as pd
import torch
from common_ai.generator import MyGenerator
from common_ai.initializer import MyInitializer
from common_ai.model import MyModelAbstract
from common_ai.optimizer import MyOptimizer
from common_ai.profiler import MyProfiler
from common_ai.train import MyTrain
from scipy import special
from tqdm import tqdm

from .data_collator import DataCollator


class DeepZF(MyModelAbstract):
    def __init__(
        self,
        protein_feature: os.PathLike,
        dna_length: int,
        zf_padding: int,
        pwm_thres: float,
    ) -> None:
        """DeepZF arguments.

        Args:
            protein_feature: file contains info for mouse C2H2 zinc fingers.
            dna_length: maximally allowed DNA length.
            zf_padding: the padding size around C2H2 zinc finger for the prediction of binding PWM.
            pwm_thres: the probability threshold to use the pwn of a zinc finger.
        """
        self.pwm_thres = pwm_thres

        self.data_collator = DataCollator(protein_feature, dna_length, zf_padding)

        self.best_thres = 0.5
        self.motifs = {}

    def eval_output(
        self, examples: list[dict], batch: dict, my_generator: MyGenerator
    ) -> pd.DataFrame:
        scores = self._get_scores(batch["input"], examples)
        batch_size = len(scores)
        probas = self._predict_proba(scores)
        df = pd.DataFrame(
            {
                "sample_idx": np.arange(batch_size),
                "proba": probas,
                "DNA": [example["DNA"] for example in examples],
                "protein": [example["protein"] for example in examples],
            }
        )

        return df

    def state_dict(self) -> dict:
        return {
            "motifs": self.motifs,
            "best_thres": self.best_thres,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.motifs = state_dict["motifs"]
        self.best_thres = state_dict["best_thres"]

    def _get_motifs(self) -> dict:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            with open(tmpdir_path / "input.csv", "w") as fd:
                fd.write("label,seq,res_12,groups\n")
                for accession, zfs in self.data_collator.protein2zf.items():
                    for zf in zfs:
                        fd.write(f"0.0,{zf['zf_ctx']},{zf['res12']},{accession}\n")

            # predict the usage probability of zinc fingers
            subprocess.run(
                [
                    "DeepZF/.conda/bin/python",
                    "DeepZF/BindZF_predictor/code/main_bindzfpredictor_predict.py",
                    "-in",
                    (tmpdir_path / "input.csv").as_posix(),
                    "-out",
                    (tmpdir_path / "output.csv").as_posix(),
                    "-m",
                    "DeepZF/BindZF_predictor/code/model.p",
                    "-e",
                    "DeepZF/BindZF_predictor/code/encoder.p",
                    "-r",
                    "1",
                ],
            )

            # predict PWM of zinc fingers
            subprocess.run(
                [
                    "DeepZF/.conda/bin/python",
                    "DeepZF/PWMpredictor/code/main_PWMpredictor.py",
                    "-in",
                    (tmpdir_path / "input.csv").as_posix(),
                    "-out",
                    (tmpdir_path / "PWM.csv").as_posix(),
                    "-m",
                    "DeepZF/PWMpredictor/code/transfer_model100.h5",
                ],
            )

            df_input = (
                pd.read_csv(tmpdir_path / "input.csv", header=0)
                .assign(
                    proba=pd.read_csv(tmpdir_path / "output.csv", names=["proba"])[
                        "proba"
                    ],
                    thres=lambda df: df.groupby("groups")["proba"]
                    .transform("max")
                    .clip(upper=self.pwm_thres),
                    pwm=list(
                        pd.read_csv(tmpdir_path / "PWM.csv", names=["pwm"])["pwm"]
                        .to_numpy()
                        .reshape([-1, 3, 4])
                    ),
                )
                .query("proba >= thres")
                .reset_index(drop=True)
            )

            motifs = {}
            for accession in df_input["groups"].drop_duplicates().to_list():
                buf = io.StringIO()
                np.savetxt(
                    buf,
                    np.concatenate(
                        df_input.loc[df_input["groups"] == accession, "pwm"].tolist()
                    ),
                    delimiter="\t",
                )
                with subprocess.Popen(
                    ["matrix2meme", "-dna"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                ) as process:
                    meme_str, _ = process.communicate(input=buf.getvalue())

                meme_str = re.sub(
                    r"\nMOTIF .+ .+\n", f"\nMOTIF {accession}\n", meme_str
                )
                motifs[accession] = meme_str

        return motifs

    def _get_scores(self, input: dict, examples: dict) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            dfs = []
            for accession in set([example["protein"] for example in examples]):
                if accession not in self.motifs:
                    dfs.append(
                        pd.DataFrame(
                            {
                                "index": [
                                    idx
                                    for idx, example in enumerate(examples)
                                    if example["protein"] == accession
                                ],
                            }
                        ).assign(score=float("nan"))
                    )
                    continue

                with open(tmpdir_path / "fimo.meme", "w") as fd:
                    fd.write(self.motifs[accession])

                with open(tmpdir_path / "fimo.fa", "w") as fd:
                    for idx, (
                        example,
                        dna,
                    ) in enumerate(
                        zip(
                            examples,
                            input["dna"],
                        )
                    ):
                        if example["protein"] != accession:
                            continue
                        fd.write(f">{idx}\n{dna}\n")

                output = subprocess.run(
                    [
                        "fimo",
                        "--best-site",
                        "--thresh",
                        "1",
                        "--no-qvalue",
                        "--max-strand",
                        "--max-stored-scores",
                        "99999999",
                        (tmpdir_path / "fimo.meme").as_posix(),
                        (tmpdir_path / "fimo.fa").as_posix(),
                    ],
                    capture_output=True,
                    text=True,
                )

                dfs.append(
                    pd.read_csv(
                        io.StringIO(output.stdout),
                        sep="\t",
                        names=[
                            "index",
                            "start",
                            "end",
                            "motif",
                            "color",
                            "strand",
                            "score",
                            "pValue",
                            "qValue",
                            "peak",
                        ],
                    )[["index", "score"]]
                )

        return pd.concat(dfs).sort_values("index")["score"].to_numpy()

    def _select_threshold(self, score: np.ndarray, bind: np.ndarray) -> float:
        min_score = min(score)
        max_score = max(score)

        metric = evaluate.load("AI/metric/accuracy.py")
        best_accuracy = -1
        for thres in np.linspace(min_score, max_score, 99):
            pred = score >= thres
            result = metric.compute(predictions=pred, references=bind)
            if result["accuracy"] > best_accuracy:
                best_accuracy = result["accuracy"]
                best_thres = thres
        return best_thres

    def _predict_proba(self, score: np.ndarray) -> np.ndarray:
        return special.expit(score - self.best_thres)

    def _predict_log_proba(self, score: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.maximum(special.log_expit(-(score - self.best_thres)), -1000),
                np.maximum(special.log_expit(score - self.best_thres), -1000),
            ],
            axis=1,
        )

    def my_train_epoch(
        self,
        my_train: MyTrain,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
        my_generator: MyGenerator,
        my_optimizer: MyOptimizer,
        my_profiler: MyProfiler,
        metrics: dict,
    ) -> tuple:
        self.motifs = self._get_motifs()

        scores, binds = [], []
        for examples in tqdm(train_dataloader):
            batch = self.data_collator(
                examples, output_label=True, my_generator=my_generator
            )
            scores.append(self._get_scores(batch["input"], examples))
            binds.append(batch["label"]["bind"].cpu().numpy())

        scores = np.concatenate(scores)
        binds = np.concatenate(binds)
        self.best_thres = self._select_threshold(score=scores, bind=binds)

        log_proba = self._predict_log_proba(scores)
        train_loss = -log_proba[np.arange(len(binds)), binds.astype(int)].sum().item()

        return train_loss, train_dataloader.dataset.num_rows, float("nan")

    def my_eval_epoch(
        self,
        my_train: MyTrain,
        eval_dataloader: torch.utils.data.DataLoader,
        my_generator: MyGenerator,
        metrics: dict,
    ) -> tuple:
        eval_loss = 0.0
        for examples in tqdm(eval_dataloader):
            batch = self.data_collator(
                examples, output_label=True, my_generator=my_generator
            )

            binds = batch["label"]["bind"].cpu().numpy()
            df = self.eval_output(examples, batch, my_generator)
            log_proba = np.stack(
                [
                    np.ma.log(1 - df["proba"]).filled(-1000),
                    np.ma.log(df["proba"]).filled(-1000),
                ],
                axis=1,
            )
            eval_loss += (
                -log_proba[np.arange(len(binds)), binds.astype(int)].sum().item()
            )

            for metric_name, metric_fun in metrics.items():
                metric_fun.step(
                    df=df,
                    examples=examples,
                    batch=batch,
                )

        metric_loss_dict = {}
        for metric_name, metric_fun in metrics.items():
            metric_loss_dict[metric_name] = metric_fun.epoch()

        return eval_loss, eval_dataloader.dataset.num_rows, metric_loss_dict

    def my_initialize_model(
        self, my_initializer: MyInitializer, my_generator: MyGenerator
    ) -> None:
        pass

    @classmethod
    def hpo(cls, trial: optuna.Trial, cfg: jsonargparse.Namespace) -> None:
        pass
