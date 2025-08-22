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

# 把uniprot accession映射到基因名字
gene_names=()
while read accession gene_name
do
    gene_names+=($gene_name)
done < <(
    printf "%s\n" "${accessions[@]}" |
    ../uniprot_gene_to_id.py \
        "UniProtKB_AC-ID" \
        "Gene_Name"
)

# 把下了数据的蛋白的motif提取出来
mkdir -p c2h2_motifs
for ((i=0; i<"${#accessions[@]}"; i++))
do
    while read filename
    do
        if [ "$filename" = "." ]
        then
            continue
        fi
        cut -f2-5 pwms_all_motifs/$filename.txt |
        tail -n+2
        echo
    done < <(
        grep -iE "\s${gene_names[$i]}\s" TF_Information.txt |
        cut -f4
    ) |
    matrix2meme |
    sed -r "s/^MOTIF [0-9]+ .+$/MOTIF CIS-BP_${accessions[$i]} ${gene_names[$i]}/" \
        > c2h2_motifs/CIS-BP_${accessions[$i]}.meme
    if [ ! -s "c2h2_motifs/CIS-BP_${accessions[$i]}.meme" ]
    then
        rm "c2h2_motifs/CIS-BP_${accessions[$i]}.meme"
    fi
done

# 把下了数据的蛋白的motif合并在一起
meme2meme $(ls c2h2_motifs/CIS-BP_*.meme) \
    > CIS-BP_motifs.dbs
