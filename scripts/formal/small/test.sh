#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ../../..

title() {
    sharps="#################################"
    printf "\n%s\n%s\n%s\n" ${sharps} $1 ${sharps}
}

test_config=AI/test.yaml
output_dir=${OUTPUT_DIR:-$HOME"/COP_results"}
run_type="formal"
run_name="small"
data_name=mouse_C2H2
trial_name="default"

for pre_model in \
    LightGBM:LightGBM \
    XGBoost:XGBoost \
    XGBoost:RandomForest \
    XGBoost:DecisionTree \
    Scikit:CategoricalNB \
    Scikit:SGDClassifier \
    Scikit:Perceptron \
    Scikit:PassiveAggressiveClassifier \
    DeepZF:DeepZF \
    COP:COP
do
    title ${pre_model}

    IFS=":" read preprocess model_cls <<<${pre_model}
    checkpoints_path=${output_dir}/${run_type}/${run_name}/checkpoints/${preprocess}/${model_cls}/${data_name}/${trial_name}
    logs_path=${output_dir}/${run_type}/${run_name}/logs/${preprocess}/${model_cls}/${data_name}/${trial_name}

    title Test
    for target in \
        F1Metric \
        AccuracyMetric \
        RecallMetric \
        PrecisionMetric \
        MatthewsCorrelationMetric \
        RocAucMetric \
        PrAucMetric \
        BrierScoreMetric
    do
        ./run.py test --config ${test_config} --checkpoints_path ${checkpoints_path} --logs_path ${logs_path} --target ${target}
    done
done
