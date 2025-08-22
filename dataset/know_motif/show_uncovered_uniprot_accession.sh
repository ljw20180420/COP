#!/bin/bash

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 收集所有accession
accessions=()
for narrowPeak in $(ls $DATA_DIR/sorted/*.sorted.narrowPeak)
do
    accession=$(basename ${narrowPeak%%.*})
    accessions+=($accession)
done

# 只用jaspar和hocomoco，其它数据库不稳定
covered_accessions=()
for subdir in "jaspar" "hocomoco"
do
    subdir=$(basename $subdir)
    readarray -t -O "${#covered_accessions[@]}" covered_accessions \
        < <(
            sed -nr '/^MOTIF /{s/^MOTIF (.+) (.+)$/\1/;p}' ${subdir}/${subdir}_motifs.dbs
        )
done

grep -v \
    -f  <(printf "%s\n" "${covered_accessions[@]}" | sort | uniq) \
    <(printf "%s\n" "${accessions[@]}")
