#!/bin/bash

# change to the dir of the script
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# change to the dir to the project
cd ../..

function title() {
    sharps="#################################"
    printf "\n%s\n%s\n%s\n" ${sharps} $1 ${sharps}
}

app_config=AI/app.yaml
target="AccuracyMetric"
maximize_target="true"
device=${device:-"cuda"}
owner="ljw20180420"

printf "inference:\n" > ${app_config}
printf "  - inference/inference.yaml\n" >> ${app_config}

data_name="mouse_C2H2"
printf "\ntest:\n" >> ${app_config}
printf "  - checkpoints_path: %s/COP_COP_%s\n" \
    ${owner} ${data_name} \
    >> ${app_config}
printf "    logs_path: %s/COP_COP_%s\n" \
    ${owner} ${data_name} \
    >> ${app_config}
printf "    target: %s\n" \
    ${target} \
    >> ${app_config}
printf "    maximize_target: %s\n" \
    ${maximize_target} \
    >> ${app_config}
printf "    overwrite:\n      train.device: %s\n" \
    ${device} \
    >> ${app_config}

./run.py app --config ${app_config}
