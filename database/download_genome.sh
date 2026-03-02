#!/bin/bash


# 查看物种基因组信息
get_accession()
{
    # 人: homo sapiens
    # 小鼠: mus musculus
    local taxon=$1
    datasets summary genome taxon ${taxon} --assembly-source refseq --as-json-lines | dataformat tsv genome --fields accession,assminfo-name,organism-name | column -ts $'\t'
}

# 用分类名下载，会下载很多基因组
download_taxon()
{
    local taxon=$1
    local filename=$2 # zip文件
    datasets download genome taxon ${taxon} --filename ${filename}
}


# 用accession编号下载，保证每次下载的一样。
download_accession()
{
    local accession=$1
    local filename=$2 # zip文件
    until datasets download genome accession ${accession} --include genome,gtf,gff3,seq-report --filename ${filename}
    do
        datasets download genome accession ${accession} --include genome,gtf,gff3,seq-report --filename ${filename}
    done
    unzip ${filename} -d ${filename%.zip}
}
