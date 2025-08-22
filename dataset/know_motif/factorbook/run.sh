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
    ENC=$(
        grep -iE "\s${gene_names[$i]}\s" factorbook_chipseq_meme_motifs.tsv |
        head -n1 |
        cut -f4
    )
    if grep "^MOTIF ${ENC}_" complete-factorbook-catalog.meme > /dev/null
    then
        cmd_ids=""
        while read MOTIF id
        do
            cmd_ids="$cmd_ids -id $id"
        done < <(
            grep "^MOTIF ${ENC}_" complete-factorbook-catalog.meme
        )
        if [ -n "$cmd_ids" ]
        then
            meme-get-motif \
                $cmd_ids complete-factorbook-catalog.meme |
            sed -r "s/^MOTIF .+$/MOTIF factorbook_${accessions[$i]} ${gene_names[$i]}/" \
                > c2h2_motifs/factorbook_${accessions[$i]}.meme
        fi
    fi
done

# 把下了数据的蛋白的motif合并在一起
meme2meme $(ls c2h2_motifs/factorbook_*.meme) \
    > factorbook_motifs.dbs
