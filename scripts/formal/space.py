#!/usr/bin/env python

import os
import pathlib

# change to the project folder
os.chdir(pathlib.Path(__file__).resolve().parent.parent.parent)

from huggingface_hub import create_repo, upload_file, upload_folder, whoami

username = whoami()["name"]
common_ai_path = f"{os.environ['PYTHONPATH']}/common_ai"
create_repo(
    repo_id=f"{username}/COP", repo_type="space", exist_ok=True, space_sdk="gradio"
)
upload_folder(
    repo_id=f"{username}/COP",
    folder_path=common_ai_path,
    path_in_repo="common_ai",
    repo_type="space",
    ignore_patterns=["__pycache__/*", "**/__pycache__", "*.yaml"],
    delete_patterns="*",
)
upload_folder(
    repo_id=f"{username}/COP",
    folder_path="AI",
    path_in_repo="AI",
    repo_type="space",
    ignore_patterns=["__pycache__/*", "**/__pycache__/*", "dataset/*", "*.pkl"],
    delete_patterns="*",
)
upload_file(
    path_or_fileobj="app.py",
    path_in_repo="app_space.py",
    repo_id=f"{username}/COP",
    repo_type="space",
)
upload_file(
    path_or_fileobj="requirements_space.txt",
    path_in_repo="requirements.txt",
    repo_id=f"{username}/COP",
    repo_type="space",
)
