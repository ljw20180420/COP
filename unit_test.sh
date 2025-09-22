#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

train_config=AI/preprocess/train.yaml
output_dir=${OUTPUT_DIR:-$HOME}
data_file=${DATA_DIR:-$HOME}/small_train_data/DNA_data.csv
test_config=AI/preprocess/test.yaml

for data_name in small65
do
    for pre_model in \
        PDBert:PDBert
    do
        IFS=":" read preprocess model_type <<<${pre_model}
        model_config=AI/preprocess/${preprocess}/${model_type}.yaml

        # Train
        ./run.py train --config ${train_config} --train.output_dir ${output_dir} --train.trial_name unit_test --train.num_epochs 1 --dataset.data_file ${data_file} --dataset.name ${data_name} --model ${model_config}

        # Eval
        ./run.py train --config ${train_config} --train.output_dir ${output_dir} --train.trial_name unit_test --train.num_epochs 1 --train.evaluation_only true --dataset.data_file ${data_file} --dataset.name ${data_name} --model ${model_config}

        # Test
        model_path=${output_dir}/${preprocess}/${model_type}/${data_name}/unit_test
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
            ./run.py test --config ${test_config} --test.model_path ${model_path} --test.target ${target}
        done
    done
done
