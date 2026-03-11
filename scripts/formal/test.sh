#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ../..

title() {
    sharps="#################################"
    printf "\n%s\n%s\n%s\n" ${sharps} $1 ${sharps}
}

test_config=AI/test.yaml
output_dir=${OUTPUT_DIR:-$HOME"/MOTIF_results"}/formal/default
data_name=mouse_C2H2

for pre_model in \
    COP:COP
do
    title ${pre_model}

    IFS=":" read preprocess model_cls <<<${pre_model}
    checkpoints_path=${output_dir}/checkpoints/${preprocess}/${model_cls}/${data_name}/default
    logs_path=${output_dir}/logs/${preprocess}/${model_cls}/${data_name}/default

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
