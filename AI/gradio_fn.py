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
        self.my_inference = self.load_inference(repo_id)

        protein_feature = pd.read_csv("AI/dataset/protein_feature.csv", header=0)
        protein_choices = list(
            zip(
                protein_feature["Entry Name"].to_list(),
                protein_feature["Entry"].to_list(),
            )
        )

        default_protein = "Q8K1M4"
        default_DNA = "GGTGGGCTTTTAAGTATCCCGGCGCAAAATTGCGATTGCATTTGGGCGGCTGTCATGCGTCCCGTCTCGCAGGATAATAAGCATTAACGGCCGGGAGGCTGACAACATCTTCCCCAGCTACGCCGGTAGAACCGGGTTACCTCATCGCAGGTGCGTGCAGGATAGCCAGCCGACCCTGACCACATCCTTTCGTAAGCGTGTATTGACGATGTAAGTGTCTCTTGGAAACGAAATCTTTTAAGAGCCCCTTAGCTTT"
        gr.Interface(
            fn=self,
            inputs=[
                gr.Dropdown(
                    choices=protein_choices,
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
