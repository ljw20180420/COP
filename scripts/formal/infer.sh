#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ../..

title() {
    sharps="#################################"
    printf "\n%s\n%s\n%s\n" ${sharps} $1 ${sharps}
}

infer_config=AI/infer.yaml
output_dir=${OUTPUT_DIR:-"${HOME}/MOTIF_results"}/formal/default

for pre_model in \
    COP:COP
do
    title ${pre_model}

    IFS=":" read preprocess model_cls <<<${pre_model}
    checkpoints_path=${output_dir}/checkpoints/${preprocess}/${model_cls}/${data_name}/default
    logs_path=${output_dir}/logs/${preprocess}/${model_cls}/${data_name}/default

    title Infer
    ./run.py infer --config ${infer_config} --test.checkpoints_path ${checkpoints_path} --test.logs_path ${logs_path}
done