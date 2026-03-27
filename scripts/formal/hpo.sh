#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ../..

function title() {
    sharps="#################################"
    printf "\n%s\n%s\n%s\n" ${sharps} $1 ${sharps}
}

hpo_config="AI/hpo.yaml"
output_dir=${OUTPUT_DIR:-"${HOME}/MOTIF_results"}/formal/hpo
trial_name="trial"
data_name=mouse_C2H2

for pre_model in \
    COP:COP
do
    title ${pre_model}

    IFS=":" read preprocess model_cls <<<${pre_model}
    model_config=AI/preprocess/${preprocess}/${model_cls}.yaml

    title Hpo
    # trial_name will be appended by trial id like trial_name-0, trial_name-1 and so on.
    ./run.py hpo \
        --config ${hpo_config} \
        --hpo.target AccuracyMetric \
        --hpo.study_name study \
        --hpo.n_trials 100 \
        --hpo.load_if_exists true \
        --train.train.output_dir ${output_dir} \
        --train.train.trial_name ${trial_name} \
        --train.train.batch_size 50 \
        --train.train.accumulate_steps 8 \
        --train.train.num_epochs 103 \
        --train.model ${model_config}
done