#!/bin/bash

printf '\\begin{tabular}{|'
for((i=0; i<10; ++i))
do
    printf "l|"
done
# printf '}\n    \\hline\n    algorithm & f1 & accuracy & recall & precision & matthews correlation & true negative & false positive & false negative & true positive \\\\\n    \\hline\n'
printf '}\n    \\hline\n    algorithm & f1 & accuracy & recall & precision & matthews correlation \\\\\n    \\hline\n'

while read line
do
    if [[ $line == {*} ]]
    then
        # for value in $(echo $line | sed -nr 's/'\''/"/g;s/np\.int64\(([0-9]+)\)/\1/g;p;q' | jq '.["f1"], .["accuracy"], .["recall"], .["precision"], .["matthews_correlation"], .["true_negative"], .["false_positive"], .["false_negative"], .["true_positive"]')
        for value in $(echo $line | sed -nr 's/'\''/"/g;s/np\.int64\(([0-9]+)\)/\1/g;p;q' | jq '.["f1"], .["accuracy"], .["recall"], .["precision"], .["matthews_correlation"]')
        do
            if [[ $value == *\.* ]]
            then
                printf ' & %.3f' $value
            else
                printf ' & %d' $value
            fi
        done
        printf ' \\\\\n'
    else
        printf '    %s' "$line"
    fi
done

printf '    \\hline\n\\end{tabular}\n'