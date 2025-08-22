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

# 先找UNVALIDATED数据库，再用CORE数据库覆盖UNVALIDATED的结果
for collection in "UNVALIDATED" "CORE"
do
    transfac="JASPAR2024_${collection}_non-redundant_pfms_transfac.txt"
    meme="JASPAR2024_${collection}_non-redundant_pfms_meme.txt"
    # 收集jaspar的motif id，uniprot accessions，基因名
    ids=()
    uniprot_idss=()
    names=()
    while IFS="|" read id uniprot_ids name
    do
        ids+=("$id")
        uniprot_idss+=("$uniprot_ids")
        names+=("$name")
    done < <(
        gawk '
            $0 ~ /^AC / {
                id = $0;
                sub(/^AC /, "", id);
            }
            $0 ~ /^CC uniprot_ids:/ {
                uniprot_ids = $0;
                sub(/^CC uniprot_ids:/, "", uniprot_ids)
                sub(/ /, "", uniprot_ids)
            }
            $0 ~ /^ID / {
                name = $0;
                sub(/^ID /, "", name)
            }
            $0 ~ /^\/\/$/ {
                printf("%s|%s|%s\n", id, uniprot_ids, name)
            }
        ' \
            < $transfac
    )

    # 把下了数据的蛋白的motif提取出来
    mkdir -p c2h2_motifs
    # 先找对应的基因名
    for ((i=0; i<"${#accessions[@]}"; i++))
    do
        cmd_id=""
        for ((j=0; j<"${#ids[@]}"; ++j))
        do
            name_match="false"
            while IFS="::" read single_name
            do
                if [ "${gene_names[$i]^^}" = "${single_name^^}" ]
                then
                    name_match="true"
                    break
                fi
            done <<< "${names[$j]}"
            if [[ "${uniprot_idss[$j]}" =~ "${accessions[$i]}" ]] || [ "$name_match" = "true" ]
            then
                cmd_id="$cmd_id -id ${ids[$j]}"
            fi
        done
        if [ -n "$cmd_id" ]
        then
            meme-get-motif $cmd_id \
                $meme |
            sed -r "s/^MOTIF .+$/MOTIF jaspar_${accessions[$i]} ${gene_names[$i]}/" \
                > "c2h2_motifs/jaspar_${accessions[$i]}.meme"
        fi
    done
done



# 把下了数据的蛋白的motif合并在一起
meme2meme $(ls c2h2_motifs/jaspar_*.meme) \
    > jaspar_motifs.dbs
