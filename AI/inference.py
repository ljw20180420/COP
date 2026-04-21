import datasets
import jsonargparse
import pandas as pd
import torch
from common_ai.inference import MyInferenceAbstract
from common_ai.test import MyTest


class MyInference(MyInferenceAbstract):
    def __init__(self, **kwargs):
        """Inference arguments.

        Args:
        """

    @torch.no_grad()
    def __call__(
        self,
        infer_df: pd.DataFrame,
        test_cfg: jsonargparse.Namespace,
        train_parser: jsonargparse.ArgumentParser,
    ) -> pd.DataFrame:
        # load model for the first call
        if not hasattr(self, "model"):
            self.load_model(test_cfg, train_parser)

        self.logger.info("prepare data loader")
        inference_dataloader = torch.utils.data.DataLoader(
            dataset=datasets.Dataset.from_pandas(infer_df),
            batch_size=self.batch_size,
            collate_fn=lambda examples: examples,
        )

        self.logger.info("inference")
        dfs, accum_sample_idx = [], 0
        for examples in inference_dataloader:
            batch = self.model.data_collator(
                examples, output_label=False, my_generator=self.my_generator
            )
            df = self.model.eval_output(examples, batch, self.my_generator)
            df["sample_idx"] += accum_sample_idx
            accum_sample_idx += len(examples)
            dfs.append(df)

        return pd.concat(dfs).reset_index(drop=True)
