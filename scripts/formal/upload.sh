#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ../..

title() {
    sharps="#################################"
    printf "\n%s\n%s\n%s\n" ${sharps} $1 ${sharps}
}

upload_config=AI/upload.yaml
output_dir=${OUTPUT_DIR:-"${HOME}/COP_results"}

for pre_model in \
    COP:COP
do
    title ${pre_model}

    IFS=":" read preprocess model_cls <<<${pre_model}

    title Upload
    ./run.py upload \
        --config ${upload_config} \
        --output_dir ${output_dir} \
        --preprocess ${preprocess} \
        --model_cls ${model_cls}
done