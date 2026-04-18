import importlib

import gradio as gr
import pandas as pd
import torch
from common_ai.gradio_fn import MyGradioFnAbstract
from common_ai.test import MyTest


class MyGradioFn(MyGradioFnAbstract):
    @torch.no_grad()
    def __call__(self, protein: str, DNA: str) -> pd.DataFrame:
        infer_df = pd.DataFrame({
            "DNA": [DNA],
            "protein": [protein],
        })
        infer_out = self.my_inference(infer_df, test_cfg=None, train_parser=None)
        proba = infer_out.loc[0, "proba"].item()

        return proba

    def launch(self):
        assert len(self.inference_dict) == 1, "more than one model loaded"
        repo_id = list(self.inference_dict.keys())[0]
        assert repo_id == "COP_COP_mouse_C2H2", "model is not COP"
        self.load_inference(repo_id)

        default_protein = "Q8K1M4"
        default_DNA = "GGTGGGCTTTTAAGTATCCCGGCGCAAAATTGCGATTGCATTTGGGCGGCTGTCATGCGTCCCGTCTCGCAGGATAATAAGCATTAACGGCCGGGAGGCTGACAACATCTTCCCCAGCTACGCCGGTAGAACCGGGTTACCTCATCGCAGGTGCGTGCAGGATAGCCAGCCGACCCTGACCACATCCTTTCGTAAGCGTGTATTGACGATGTAAGTGTCTCTTGGAAACGAAATCTTTTAAGAGCCCCTTAGCTTT"
        gr.Interface(
            fn=self,
            inputs=[
                gr.Dropdown(
                    choices=self.my_inference.model.data_collator.protein_ids.index.to_list(),
                    value=default_protein,
                    label="select mouse C2H2 protein",
                    info="the protein to predict the occupancy",
                ),
                gr.Textbox(
                    placeholder=default_DNA,
                    label="site DNA sequence",
                    info="The DNA sequence of the site for the mouse C2H2 protein to occupy",
                ),
            ],
            outputs=[gr.Number(label="occupancy probability")],
            examples=[[default_protein, default_DNA]],
            cache_examples=True,
            cache_mode="eager",
            description="""
# Welcome. This app predicts the occupancy probability of the mouse C2H2 protein.
            """,
            flagging_mode="never",
        ).launch()

    def load_inference(self, repo_id: str) -> None:
        inference_cfg, test_cfg = self.inference_dict[repo_id]
        inference_module, inference_cls = inference_cfg.class_path.rsplit(".", 1)
        self.my_inference = getattr(
            importlib.import_module(inference_module), inference_cls
        )(**inference_cfg.init_args.as_dict())
        (
            _,
            train_cfg,
            self.my_inference.logger,
            self.my_inference.model,
            self.my_inference.my_generator,
        ) = MyTest(**test_cfg.as_dict()).load_model(self.train_parser)
        self.my_inference.batch_size = train_cfg.train.batch_size
