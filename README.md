- [TODO](#todo)
- [运行流程](#运行流程)
- [Install](#install)
- [Train](#train)
- [Test and save pipeline](#test-and-save-pipeline)
- [Inference](#inference)
- [App](#app)

# TODO

- Review the hpo result on 107 machine
- Try to train for 3000 for COP
- Upload model to huggingface, deploy space for both versions
- README
- Add Likelihood metric.
- Use PRROC and AUCROC to choose a good threshold for COP to show that COP is good. (https://stats.stackexchange.com/questions/354704/what-does-it-mean-if-the-roc-auc-is-high-and-the-average-precision-is-low>
- 增加DNA或蛋白长度
- 发表benchmark
- Add complexity only when there is provable improvements.
- Agent.


- Replace huggingface evaluation by scikit-learn metrics for offline usage.
- Apply incremental learning of scikit-learn.
- Use the method at [XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/c_api_tutorial.html#install-xgboost-on-conda-environment) to install dssp in project conda.
- Use one-hot encoding for models not support categorial features.
- The complexity of cross-attention can be decreased from quadratic to linear.
- zero-shot learning
- contrastive learning
- 阴性对照（随机蛋白序列, 打乱蛋白顺序, 突变锌指蛋白, 未训练的锌指蛋白, 非锌指蛋白）
- baseline需要加上24年的briefings bioinformatics和李天杰说的review,以及DeepDF的引用
- Model interpretability. (attention)
- Baseline model. (DummyClassifier, scikit-learn, lightGBM, DeepZF)
- Add comment.
- Write paper.
- use all data
- remove duplicate DNA sequences
- Select DNA based on hyper-sensitive sites
- Use diff peak rather than random shuffle as negative samples.


# 运行流程

```shell
preprocess/run.sh
AI_models/run_bind_transformer.py --command download
AI_models/run_bind_transformer.py --command test
```

# Install

```shell
conda create --name ENVIRONMENT --file conda.yaml
conda create -prefix PATH --file conda.yaml
```

# Train

```python

```

# Test and save pipeline

```python
from AI_models.bind_transformer.test import test

test(data_files="test/data.csv")
```

# Inference

```python
from AI_models.bind_transformer.inference import inference

for output in inference(data_files="test/inference.csv"):
    pass
```

# App

```python
from AI_models.bind_transformer.app import app

app()
```
