#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ../..

title() {
    sharps="#################################"
    printf "\n%s\n%s\n%s\n" ${sharps} $1 ${sharps}
}

train_config=AI/train.yaml
output_dir=${OUTPUT_DIR:-"${HOME}/COP_results"}
run_type="formal"
run_name="default"
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
    model_config=AI/preprocess/${preprocess}/${model_cls}.yaml

    title Train
    case ${model_cls} in
        COP)
            ./run.py train \
                --config ${train_config} \
                --train.output_dir ${output_dir}/${run_type}/${run_name} \
                --train.trial_name ${trial_name} \
                --train.num_epochs 103 \
                --train.evaluation_only false \
                --model ${model_config}
        ;;
        LightGBM)
            ./run.py train \
                --config ${train_config} \
                --train.output_dir ${output_dir}/${run_type}/${run_name} \
                --train.trial_name ${trial_name} \
                --train.evaluation_only false \
                --train.device cpu \
                --model ${model_config}
        ;;
        XGBoost)
            ./run.py train \
                --config ${train_config} \
                --train.output_dir ${output_dir}/${run_type}/${run_name} \
                --train.trial_name ${trial_name} \
                --train.num_epochs 63 \
                --train.evaluation_only false \
                --train.device cpu \
                --model ${model_config}
        ;;
        RandomForest|DecisionTree)
            ./run.py train \
                --config ${train_config} \
                --train.output_dir ${output_dir}/${run_type}/${run_name} \
                --train.trial_name ${trial_name} \
                --train.num_epochs 1 \
                --train.evaluation_only false \
                --train.device cpu \
                --model ${model_config}
        ;;
        DeepZF)
            ./run.py train \
                --config ${train_config} \
                --train.output_dir ${output_dir}/${run_type}/${run_name} \
                --train.trial_name ${trial_name} \
                --train.num_epochs 1 \
                --train.evaluation_only false \
                --model ${model_config}
        ;;
        *)
            ./run.py train \
                --config ${train_config} \
                --train.output_dir ${output_dir}/${run_type}/${run_name} \
                --train.trial_name ${trial_name} \
                --train.evaluation_only false \
                --model ${model_config}
        ;;
    esac
done