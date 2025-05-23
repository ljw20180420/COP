from transformers import (
    Trainer,
    TrainingArguments,
)
from transformers.trainer_pt_utils import nested_detach
from datasets import Dataset
from pathlib import Path
from logging import Logger
from typing import Union, Any, Optional
import optuna
from datetime import datetime
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
from .model import BindTransformerConfig, BindTransformerModel
from .load_data import DataCollator
from .metric import compute_metrics


class MyTrainer(Trainer):
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, output = self.compute_loss(
                model,
                inputs,
                num_items_in_batch=num_items_in_batch,
                return_outputs=True,
            )

        with torch.no_grad():
            self.data_collator.neg_loss[inputs["rows"], inputs["cols"]] = (
                -F.logsigmoid(-output["logit"][inputs["bind"] == 0.0])
                .cpu()
                .detach()
                .numpy()
            )

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()

        kwargs = {}

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # Finally we need to normalize the loss for reporting
        if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss, **kwargs)

        return loss.detach()

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config,
                    "keys_to_ignore_at_inference",
                    ["past_key_values"],
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True
                    )
                loss = loss.detach().mean()

                if isinstance(outputs, dict):
                    logits = tuple(
                        v for k, v in outputs.items() if k not in ignore_keys + ["loss"]
                    )
                else:
                    logits = outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(
                        v for k, v in outputs.items() if k not in ignore_keys
                    )
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        with torch.no_grad():
            self.data_collator.neg_loss[inputs["rows"], inputs["cols"]] = (
                -F.logsigmoid(-logits[labels == 0.0]).cpu().detach().numpy()
            )

        return (loss, logits, labels)


def train(
    ds: Dataset,
    proteins: list[str],
    seconds: list[str],
    zinc_nums: list[int],
    train_output_dir: Path,
    seed: int,
    device: str,
    fp16: bool,
    logger: Logger,
    batch_size: int,
    dna_length: int,
    minimal_unbind_summit_distance: int,
    select_worst_neg_loss_ratio: float,
    optimizer: str,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    scheduler: str,
    num_epochs: float,
    warmup_ratio: float,
    protein_vocab: int,
    second_vocab: int,
    dna_vocab: int,
    max_num_tokens: int,
    dim_emb: int,
    num_heads: int,
    dim_heads: int,
    depth: int,
    dim_ffn: int,
    dropout: float,
    norm_eps: float,
    pos_weight: float,
    reg_l1: float,
    reg_l2: float,
    initializer_range: float,
    use_hyena: bool,
    hyena_order: int,
    hyena_filter_order: int,
    hp_study_name: str,
    hp_storage: str,
    redundant_parameters: list[dict[str, Union[str, int, float, list]]],
    n_trials: int,
):
    """
    For the meanings of parameters, execute: AI_models/run_bind_transformer.py -h.
    """

    do_hyperparameter_search = (
        hp_study_name and hp_storage and len(redundant_parameters) and n_trials > 0
    )
    if do_hyperparameter_search:
        hp_study_name = f"""{hp_study_name}.{datetime.now().isoformat()}"""

    logger.info("initialize model")
    BindTransformerConfig.register_for_auto_class()
    BindTransformerModel.register_for_auto_class()

    def model_init(trial: optuna.trial.Trial) -> BindTransformerModel:
        config = {
            "protein_vocab": protein_vocab,
            "second_vocab": second_vocab,
            "dna_vocab": dna_vocab,
            "max_num_tokens": max_num_tokens,
            "dim_emb": dim_emb,
            "num_heads": num_heads,
            "dim_heads": dim_heads,
            "depth": depth,
            "dim_ffn": dim_ffn,
            "dropout": dropout,
            "norm_eps": norm_eps,
            "pos_weight": pos_weight,
            "reg_l1": reg_l1,
            "reg_l2": reg_l2,
            "initializer_range": initializer_range,
            "use_hyena": use_hyena,
            "hyena_order": hyena_order,
            "hyena_filter_order": hyena_filter_order,
            "seed": seed,
        }

        if trial:
            for key, val in trial.params.items():
                if key in config:
                    config[key] = val
        return BindTransformerModel(BindTransformerConfig(**config))

    logger.info("set train arguments")
    training_args = TrainingArguments(
        output_dir=(
            train_output_dir / "train"
            if not do_hyperparameter_search
            else train_output_dir / hp_study_name
        ),
        eval_strategy="epoch",
        eval_accumulation_steps=1,  # 省GPU
        torch_empty_cache_steps=1,  # 省GPU
        lr_scheduler_kwargs=None,  # 太复杂了
        logging_strategy="epoch",
        save_strategy="epoch",
        save_safetensors=False,
        use_cpu=True if device == "cpu" else False,
        seed=seed,
        fp16=fp16,
        remove_unused_columns=False,
        label_names=BindTransformerConfig.label_names,
        load_best_model_at_end=True,
    )
    training_args.set_dataloader(
        train_batch_size=batch_size, eval_batch_size=batch_size
    )
    # 不要用优化器自带的weight_decay, 它会对所有参数，包括线性层的偏置和归一化层的参数等都进行L2正则化
    training_args.set_optimizer(
        name=optimizer,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
    )
    training_args.set_lr_scheduler(
        name=scheduler, num_epochs=num_epochs, warmup_ratio=warmup_ratio
    )

    if os.path.exists(train_output_dir / "train" / "neg_loss.npy"):
        neg_loss = np.load(train_output_dir / "train" / "neg_loss.npy")
    else:
        neg_loss = np.full((sum(ds.num_rows.values()), len(proteins)), np.inf)
    trainer = MyTrainer(
        args=training_args,
        data_collator=DataCollator(
            proteins,
            seconds,
            zinc_nums,
            minimal_unbind_summit_distance,
            select_worst_neg_loss_ratio,
            neg_loss,
            dna_length,
            max_num_tokens,
            seed,
        ),
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        model_init=model_init,
        compute_metrics=lambda eval_prediction: compute_metrics(
            eval_prediction.predictions, eval_prediction.label_ids
        ),
    )

    if not do_hyperparameter_search:
        logger.info("train model")
        try:
            trainer.train(resume_from_checkpoint=True)
        except (ValueError, FileNotFoundError):
            trainer.train()

        logger.info("save model")
        trainer.save_model()
        trainer.create_model_card()
        np.save(
            train_output_dir / "train" / "neg_loss.npy", trainer.data_collator.neg_loss
        )
    else:
        logger.info("search hyperparameter")

        def hp_space(trial: optuna.trial.Trial):
            for param in redundant_parameters:
                if param["type"] == "categorical":
                    param.pop("type")
                    trial.suggest_categorical(**param)
                elif param["type"] == "int":
                    param.pop("type")
                    trial.suggest_int(**param)
                else:
                    assert (
                        param["type"] == "float"
                    ), "redundant parameter type is not in categorical, int, float"
                    param.pop("type")
                    trial.suggest_float(**param)
            return trial.params

        def compute_objective(metrics):
            return metrics["eval_loss"]

        # Trainer会自动调用_report_to_hp_search方法，从而调用optuna.TrialPruned()
        trainer.hyperparameter_search(
            hp_space=hp_space,
            compute_objective=compute_objective,
            n_trials=n_trials,
            direction="minimize",
            backend="optuna",
            hp_name=None,
            storage=optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(hp_storage),
            ),
            sampler=None,  # 默认optuna.samplers.TPESampler()
            pruner=None,  # 默认optuna.pruners.MedianPruner()
            study_name=hp_study_name,
        )
