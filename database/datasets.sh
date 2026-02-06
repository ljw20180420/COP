#!/bin/bash

# 查看当前的小鼠基因组信息
# datasets summary genome taxon 'mus musculus' --assembly-source refseq --as-json-lines | dataformat tsv genome --fields accession,assminfo-name,organism-name | column -ts $'\t'

# 用分类名下载，会下载很多基因组。 
# datasets download genome taxon 'mus musculus' --filename $SRA_CACHE/mouse_dataset.zip

# 用accession编号下载，保证每次下载的一样。
until datasets download genome accession GCF_000001635.27 --include genome,gtf,gff3 --filename $SRA_CACHE/mouse_dataset.zip
do
    datasets download genome accession GCF_000001635.27 --include genome,gtf,gff3 --filename $SRA_CACHE/mouse_dataset.zip
done
unzip $SRA_CACHE/mouse_dataset.zip
