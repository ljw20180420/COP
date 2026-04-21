#!/usr/bin/env python

import os
import pathlib

import yaml

# change to the project folder
os.chdir(pathlib.Path(__file__).resolve().parent.parent.parent)

from huggingface_hub import create_repo, upload_file, upload_folder, whoami

# Use cpu in hf space
with open("AI/app.yaml", "r") as rd, open("AI/app_space.yaml", "w") as wd:
    app_cfg = yaml.safe_load(rd)
    app_cfg["test"][0]["overwrite"]["train.device"] = "cpu"
    yaml.safe_dump(app_cfg, wd)

username = whoami()["name"]
create_repo(
    repo_id=f"{username}/COP", repo_type="space", exist_ok=True, space_sdk="gradio"
)
upload_folder(
    repo_id=f"{username}/COP",
    folder_path="AI",
    path_in_repo="AI",
    repo_type="space",
    ignore_patterns=[
        "__pycache__/*",
        "**/__pycache__/*",
        "dataset/*",
        "*.pkl",
    ],
    delete_patterns="*",
)
upload_file(
    path_or_fileobj="AI/dataset/protein_feature.csv",
    path_in_repo="AI/dataset/protein_feature.csv",
    repo_id=f"{username}/COP",
    repo_type="space",
)
upload_file(
    path_or_fileobj="app_space.py",
    path_in_repo="app.py",
    repo_id=f"{username}/COP",
    repo_type="space",
)
upload_file(
    path_or_fileobj="requirements_space.txt",
    path_in_repo="requirements.txt",
    repo_id=f"{username}/COP",
    repo_type="space",
)
