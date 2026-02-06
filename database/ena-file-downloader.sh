#!/bin/bash

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

java -jar bin/ena-file-downloader.jar --accessions=ERR164407 --format=READS_FASTQ --location=$SRA_CACHE --protocol=ASPERA --asperaLocation=./
