import io
import os
import pathlib
import re
import subprocess
import tempfile
from typing import Optional

import evaluate
import jsonargparse
import numpy as np
import optuna
import pandas as pd
import torch
from common_ai.generator import MyGenerator
from common_ai.initializer import MyInitializer
from common_ai.optimizer import MyOptimizer
from common_ai.profiler import MyProfiler
from common_ai.train import MyTrain
from tqdm import tqdm

from .data_collator import DataCollator


class DeepZF:
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
        X_value = self._get_feature(
            input=batch["input"],
            label=None,
        )
        batch_size = X_value.shape[0]
        probas = self.booster.predict(data=X_value)
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
        # TODO: save threshold and meme strings
        # torch.frombuffer(bytearray(meme_str.encode()), dtype=torch.uint8)
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        # TODO: load threshold and meme strings
        # state_dict["meme_str"].numpy().tobytes().decode()
        pass

    def get_motifs(self) -> dict:
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
                        pd.read_csv(tmpdir_path / "pwm.csv", names=["pwm"])["pwm"]
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
                ) as process:
                    process.stdin.write(buf.getvalue())
                    meme_str = process.stdout.read()

                meme_str = re.sub(
                    r"\nMOTIF .+ .+\n", f"\nMOTIF {accession}\n", meme_str
                )
                motifs[accession] = meme_str

        return motifs

    def get_score(self, input: dict) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            dfs = []
            for accession in set(input["protein"]):
                with open(tmpdir_path / "fimo.meme", "w") as fd:
                    fd.write(self.motifs[accession])

                with open(tmpdir_path / "fimo_in.csv", "w") as fd:
                    for idx, (
                        protein,
                        dna,
                    ) in enumerate(
                        zip(
                            input["protein"],
                            input["dna"],
                        )
                    ):
                        if protein != accession:
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
                        (tmpdir_path / "fimo_in.csv").as_posix(),
                    ],
                    capture_output=True,
                )
                dfs.append(
                    pd.read_csv(
                        io.StringIO(output.stdout.decode()),
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
                    ).assign(score=lambda df: -df["pValue"].log10())[["index", "score"]]
                )

        return pd.concat(dfs).sort_values("index")["score"].to_numpy()

    def select_threshold(
        self, score: np.ndarray, bind: np.ndarray
    ) -> tuple[float, float]:
        min_val = min(score)
        max_val = max(score)

        metric = evaluate.load("AI/metric/accuracy.py")
        best_result = {"accuracy": -1}
        for thres in np.linspace(min_val, max_val, 99):
            pred = score >= thres
            result = metric.compute(predictions=pred, references=bind)
            if result["accuracy"] > best_result["accuracy"]:
                best_result = result
                best_thres = thres
        return best_thres, best_result

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
        self.motifs = self.get_motifs()

        scores, binds, train_loss_num = [], [], 0.0
        for examples in tqdm(train_dataloader):
            train_loss_num += len(examples)
            batch = self.data_collator(
                examples, output_label=True, my_generator=my_generator
            )
            scores.append(self.get_score(batch["input"]))
            binds.append(batch["label"]["bind"].cpu().numpy())

        self.best_thres, best_result = self.select_threshold(
            score=np.concatenate(scores), bind=np.concatenate(binds)
        )

        return best_result * train_loss_num, train_loss_num, float("nan")

    def my_eval_epoch(
        self,
        my_train: MyTrain,
        eval_dataloader: torch.utils.data.DataLoader,
        my_generator: MyGenerator,
        metrics: dict,
    ) -> tuple:
        eval_loss = (
            self.booster.eval(data=self.eval_data, name="eval")[0][2].item()
            * self.eval_data.num_data()
        )
        for examples in tqdm(eval_dataloader):
            batch = self.data_collator(
                examples, output_label=True, my_generator=my_generator
            )
            df = self.eval_output(examples, batch, my_generator)
            for metric_name, metric_fun in metrics.items():
                metric_fun.step(
                    df=df,
                    examples=examples,
                    batch=batch,
                )

        metric_loss_dict = {}
        for metric_name, metric_fun in metrics.items():
            metric_loss_dict[metric_name] = metric_fun.epoch()

        return eval_loss, self.eval_data.num_data(), metric_loss_dict

    def my_initialize_model(
        self, my_initializer: MyInitializer, my_generator: MyGenerator
    ) -> None:
        pass

    def _get_feature(
        self,
        input: dict,
        label: Optional[dict],
    ) -> tuple[np.ndarray]:
        X_value = np.concatenate(
            (
                input["dna_id"].cpu().numpy(),
                input["protein_id"].cpu().numpy(),
                input["second_id"].cpu().numpy(),
            ),
            axis=1,
        )

        if label is not None:
            y_value = label["bind"].cpu().numpy()
            return X_value, y_value

        return X_value

    @classmethod
    def hpo(cls, trial: optuna.Trial, cfg: jsonargparse.Namespace) -> None:
        pass

    def hpo(cls, trial: optuna.Trial, cfg: jsonargparse.Namespace) -> None:
        pass
        pass
