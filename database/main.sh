#!/bin/bash

# 切换到当前脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) 

. ./download_genome.sh

sra_acc_list="${1:-"SRA_ACCESSION_LIST_FILE.txt"}"
gse_acc_list="${2:-"my_scripts/GSE.txt"}"
cache_dir="${3:-"${CACHE_DIR}"}"

# 下载基因组并解压.
download_accession "GCF_000001635.27" "${SRA_CACHE}/human_dataset.zip"
