#!/usr/bin/env python

import os

# 把运行文件夹切换为脚本文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 读取参数和日志记录器
from bind_transformer.config import get_config, get_logger

args = get_config(
    [
        "bind_transformer/config_default.ini",
        "bind_transformer/config_custom.ini",
    ]
)
if args.command != "inference":
    raise Exception("command must be inference")
logger = get_logger(args.log_level)

# 读取度量
logger.warning("load metric")
from bind_transformer.metric import hard_metric

# 读取数据
logger.warning("load data")
from datasets import load_dataset
from bind_transformer.load_data import DataCollator, train_validation_test_split

ds_protein = load_dataset(
    "csv",
    data_files=(args.data_dir / "protein_data.csv").as_posix(),
    column_names=["accession", "protein", "second", "zinc_num"],
)["train"]

ds = load_dataset("json", data_dir=args.data_dir / "DNA_data")
ds = train_validation_test_split(ds, args.validation_ratio, args.test_ratio, args.seed)
data_collater = DataCollator(
    ds_protein["protein"],
    ds_protein["second"],
    ds_protein["zinc_num"],
    args.minimal_unbind_summit_distance,
    0.0,  # select negative sample randomly by set select_worst_neg_loss_ratio=0.0
    None,
    args.dna_length,
    args.max_num_tokens,
    args.seed,
)
ds = ds.map(data_collater.neg_map, batched=True, remove_columns=["rn", "distance"])

logger.warning("tokenize")
import numpy as np
from bind_transformer.tokenizers import (
    DNA_Tokenizer,
    Protein_Bert_Tokenizer,
    Second_Tokenizer,
)

dna_tokenizer = DNA_Tokenizer(args.dna_length)
protein_tokenizer = Protein_Bert_Tokenizer(args.max_num_tokens)
second_tokenizer = Second_Tokenizer(args.max_num_tokens)


def tokenizer_ids_bind(ds, ds_protein):
    ids = np.concat(
        [
            dna_tokenizer(ds["dna"], [0] * len(ds)).numpy(),
            protein_tokenizer(
                [ds_protein["protein"][index] for index in ds["index"]]
            ).numpy(),
            second_tokenizer(
                [ds_protein["second"][index] for index in ds["index"]]
            ).numpy(),
        ],
        axis=1,
    )
    bind = np.array(ds["bind"])
    return ids, bind


train_ids, train_bind = tokenizer_ids_bind(ds["train"], ds_protein)
eval_ids, eval_bind = tokenizer_ids_bind(ds["validation"], ds_protein)
test_ids, test_bind = tokenizer_ids_bind(ds["test"], ds_protein)

del ds
del ds_protein

logger.warning("train LGBMClassifier")
from lightgbm import LGBMClassifier

import os
import joblib

classifier = LGBMClassifier(random_state=63036)
if os.path.exists("scikit-learn/lightGBM.pkl"):
    classifier = joblib.load("scikit-learn/lightGBM.pkl")
else:
    classifier.fit(
        train_ids,
        train_bind,
        eval_set=[(eval_ids, eval_bind)],
    )
    joblib.dump(classifier, f"scikit-learn/lightGBM.pkl")
pred = classifier.predict(test_ids)
with open("scikit-learn/bench.log", "a") as fd:
    fd.write(f"""lightGBM\n{hard_metric(pred, test_bind)}\n""")
    fd.flush()

logger.warning("merge train and eval data")
train_eval_ids = np.concat([train_ids, eval_ids])
del train_ids
del eval_ids
train_eval_bind = np.concat([train_bind, eval_bind])
del train_bind
del eval_bind

logger.warning("train scikit classifiers")
import resource

soft, hard = resource.getrlimit(resource.RLIMIT_AS)
# limit memory usage to 50 GB
resource.setrlimit(resource.RLIMIT_AS, (50 * 1024**3, hard))

# out of memory classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

classifiers = {
    "LDA": LinearDiscriminantAnalysis(),
    "Gaussian Process": GaussianProcessClassifier(random_state=63036),
    "QDA": QuadraticDiscriminantAnalysis(),
    "Ridge Classifier": RidgeClassifierCV(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Neural Net": MLPClassifier(
        hidden_layer_sizes=(256, 256),
        activation="logistic",
        learning_rate_init=1e-5,
        max_iter=1000,
        random_state=63036,
        tol=1e-8,
        n_iter_no_change=50,
        early_stopping=True,
    ),
    "Linear SVC": LinearSVC(max_iter=1000, random_state=63036),
    "Logistic Regression": LogisticRegressionCV(max_iter=1000, random_state=63036),
}

# enough memory classifier
from sklearn.linear_model import (
    SGDClassifier,
    Perceptron,
    PassiveAggressiveClassifier,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

classifiers = {
    "SGD Classifier": SGDClassifier(max_iter=1000, random_state=63036),
    "Perceptron": Perceptron(max_iter=1000, random_state=63036),
    "PassiveAggressiveClassifier": PassiveAggressiveClassifier(
        max_iter=1000, random_state=63036
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=63036),
    "Random Forest": RandomForestClassifier(random_state=63036),
    "AdaBoost": AdaBoostClassifier(random_state=63036),
    "Naive Bayes": GaussianNB(),
    "Dummy": DummyClassifier(strategy="stratified", random_state=63036),
}

import os
import joblib

for name, classifier in classifiers.items():
    if os.path.exists(f"scikit-learn/{name}.pkl"):
        logger.warning(f"load {classifier}")
        classifier = joblib.load(f"scikit-learn/{name}.pkl")
    else:
        logger.warning(f"train {classifier}")
        classifier.fit(train_eval_ids, train_eval_bind)
        joblib.dump(classifier, f"scikit-learn/{name}.pkl")
    logger.warning(f"predict {classifier}")
    pred = classifier.predict(test_ids)
    with open("scikit-learn/bench.log", "a") as fd:
        fd.write(f"""{name}\n{hard_metric(pred, test_bind)}\n""")
