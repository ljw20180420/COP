#!/bin/bash

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 使用严格模式
set -euo pipefail

title() {
    printf "\n----------\n%s\n----------\n" $1 >&2
}

# title "下载蛋白文件"
# ./uniprot_download.py \
#     'ft_zn_fing:C2H2' \
#     'organism_name:"Mus musculus"'

# title "下载蛋白结构"
# ./get_mmcif_from_alphafoldDB.py

# title "计算蛋白二级结构"
# printf "accession,sequence,secondary_structure\n" >secondary_structure.csv
# for mmcif in $(find get_mmcif_from_alphafoldDB/ -name "*.mmcif")
# do
#     stem=${mmcif##*/}
#     stem=${stem%.mmcif}
#     printf "%s," ${stem} >> secondary_structure.csv
#     mkdssp --output-format dssp $mmcif | sed '1,/^  #/d' | cut  -c14 | tr -d '\n' >> secondary_structure.csv
#     printf "," >> secondary_structure.csv
#     mkdssp --output-format dssp $mmcif | sed '1,/^  #/d' | cut  -c17 | tr -d '\n' | tr ' ' '-' >> secondary_structure.csv
#     printf "\n" >> secondary_structure.csv
# done

# title "融合蛋白和二级结构,标注zinc-finger,KRAB,disorder"
# ./parse_ft.py

title "收集所有accession"
accessions=()
for narrowPeak in $(ls $DATA_DIR/sorted/*.sorted.narrowPeak)
do
    accession=$(basename ${narrowPeak%%.*})
    accessions+=($accession)
done

# title "清洗sorted.narrowPeak"
# for accession in "${accessions[@]}"
# do
#     printf "clean sorted narrowPeak for %s\n" $accession
#     awk '
#         NF == 10 {
#             print
#         }
#     ' $DATA_DIR/sorted/$accession.sorted.narrowPeak \
#     > $DATA_DIR/sorted/$accession.sorted.narrowPeak2
#     mv $DATA_DIR/sorted/$accession.sorted.narrowPeak2 $DATA_DIR/sorted/$accession.sorted.narrowPeak
# done

# title "把黑名单的peak去掉|把peak聚类"
# mkdir -p $DATA_DIR/clustered
# cluster_max_distance="-50"
# for accession in "${accessions[@]}"
# do
#     if [ -f "$DATA_DIR/clustered/$accession.clustered.narrowPeak" ]
#     then
#         continue
#     fi
#     printf "calculate peak cluster for %s\n" $accession
#     bedtools intersect -sorted -v \
#         -a $DATA_DIR/sorted/$accession.sorted.narrowPeak \
#         -b <(
#             bedtools sort -i $GENOME_BLACK
#         ) |
#     bedtools cluster \
#         -d $cluster_max_distance \
#         > $DATA_DIR/clustered/$accession.clustered.narrowPeak
# done

# title "每个聚类选择好的分位数|不选择最大值|防止异常值"
# mkdir -p $DATA_DIR/selected
# cluster_quantile=0.9
# for accession in "${accessions[@]}"
# do
#     if [ -f "$DATA_DIR/selected/$accession.selected.narrowPeak" ]
#     then
#         continue
#     fi
#     printf "select peak for %s\n" $accession
#     ./peak_cluster_select.py \
#         < $DATA_DIR/clustered/$accession.clustered.narrowPeak \
#         $cluster_quantile \
#         > $DATA_DIR/selected/$accession.selected.narrowPeak
# done

# title "去掉太宽太窄的峰|去掉太显著太不显著的峰"
# mkdir -p $DATA_DIR/filtered
# for accession in "${accessions[@]}"
# do
#     if [ -f "$DATA_DIR/filtered/$accession.filtered.narrowPeak" ]
#     then
#         continue
#     fi
#     printf "filtered peak for %s\n" $accession
#     wlb=$(
#         awk '{print $3 - $2}' \
#             < $DATA_DIR/selected/$accession.selected.narrowPeak |
#         sort -n |
#         perl -e '$d=0.1;@l=<>;print $l[int($d*$#l)]'
#     )
#     wub=$(
#         awk '{print $3 - $2}' \
#             < $DATA_DIR/selected/$accession.selected.narrowPeak |
#         sort -n |
#         perl -e '$d=0.9;@l=<>;print $l[int($d*$#l)]'
#     )
#     plb=$(
#         awk '{print $8}' \
#             < $DATA_DIR/selected/$accession.selected.narrowPeak |
#         sort -g |
#         perl -e '$d=0.1;@l=<>;print $l[int($d*$#l)]'
#     )
#     pub=$(
#         awk '{print $8}' \
#             < $DATA_DIR/selected/$accession.selected.narrowPeak |
#         sort -g |
#         perl -e '$d=0.9;@l=<>;print $l[int($d*$#l)]'
#     )
#     awk -v wlb=$wlb -v wub=$wub -v plb=$plb -v pub=$pub '$3 - $2 <= wub && $3 - $2 >= wlb && $8 >= plb && $8 <= pub {print}' \
#         < $DATA_DIR/selected/$accession.selected.narrowPeak \
#         > $DATA_DIR/filtered/$accession.filtered.narrowPeak
# done

# title "调整peak大小|按照summit位置排序"
# mkdir -p $DATA_DIR/sized
# seq_len=256
# for accession in "${accessions[@]}"
# do
#     if [ -f "$DATA_DIR/sized/$accession.sized.narrowPeak" ]
#     then
#         continue
#     fi
#     printf "resize peak for %s\n" $accession
#     bedClip \
#         <(
#             awk -v seq_len=$seq_len '
#                 {
#                     start = $2
#                     end = $3
#                     summit = $10
#                     new_start = start + summit - int(seq_len / (end - start) * summit)
#                     new_end = new_start + seq_len
#                     new_summit = start + simmit
#                     printf("%s\t%d\t%d\t%d\n", $1, new_start, new_end, new_summit)
#                 }
#             ' $DATA_DIR/filtered/$accession.filtered.narrowPeak |
#             sort -k1,1 -k4,4n
#         ) \
#         $GENOME_SIZE \
#         $DATA_DIR/sized/$accession.sized.narrowPeak
# done

# title "提取结合位点序列"
# mkdir -p $DATA_DIR/positive
# for accession in "${accessions[@]}"
# do
#     if [ -f "$DATA_DIR/positive/$accession.positive" ]
#     then
#         continue
#     fi
#     printf "extract peak sequence for %s\n" $accession
#     # --line-width 0 防止fasta换行
#     paste \
#         $DATA_DIR/sized/$accession.sized.narrowPeak \
#         <(
#             seqkit subseq \
#                 < $GENOME \
#                 --update-faidx \
#                 --line-width 0 \
#                 --bed $DATA_DIR/sized/$accession.sized.narrowPeak |
#             sed -nr '2~2{y/acgtn/ACGTN/; p}'
#         ) |
#     grep -vE "\sN|[ACGT]N" \
#         > $DATA_DIR/positive/$accession.positive
# done

# title "计算最近交叉peak距离|生成训练数据集的DNA"
# mkdir -p $DATA_DIR/train_data/DNA_data
# for ((i=0;i<${#accessions[@]};++i))
# do
#     if [ -f "$DATA_DIR/train_data/DNA_data/${accessions[$i]}.csv" ]
#     then
#         continue
#     fi
#     printf "calculate the closest peak from other proteins for %s\n" ${accessions[$i]}
#     accession="${accessions[$i]}"
#     dis_files=()
#     for accession2 in "${accessions[@]}"
#     do
#         if [ "${accession2}" != "${accession}" ]
#         then
#             bedtools closest -sorted -d -t first \
#                 -a <(
#                     awk '
#                         {
#                             printf("%s\t%s\t%s\n", $1, $4, $4 + 1)
#                         }
#                     ' $DATA_DIR/positive/${accession}.positive
#                 ) \
#                 -b <(
#                     awk '
#                         {
#                             printf("%s\t%s\t%s\n", $1, $4, $4 + 1)
#                         }
#                     ' $DATA_DIR/positive/${accession2}.positive
#                 ) |
#             cut -f7 \
#                 > $DATA_DIR/train_data/DNA_data/${accession}_${accession2}
#         else
#             awk '{print 0}' \
#                 $DATA_DIR/positive/${accession}.positive \
#                 > $DATA_DIR/train_data/DNA_data/${accession}_${accession2}
#         fi
#         dis_files+=($DATA_DIR/train_data/DNA_data/${accession}_${accession2})
#     done
#     paste -d, \
#         <(
#             awk -v idx=$i '
#                 {
#                     printf("%d,%s\n", idx, $5)
#                 }
#             ' "$DATA_DIR/positive/${accessions[$i]}.positive"
#         ) \
#         <(
#             paste -d: \
#                 "${dis_files[@]}"
#         ) \
#         > "$DATA_DIR/train_data/DNA_data/${accessions[$i]}.csv"
#     rm "${dis_files[@]}"
# done

# title "生成训练数据集的蛋白"
# mkdir -p $DATA_DIR/train_data
# printf "Entry,sequence,secondary_structure,zinc_finger,disorder,KRAB\n" > $DATA_DIR/train_data/protein_data.csv
# for ((i=0;i<${#accessions[@]};++i))
# do
#     grep -F "${accessions[$i]}" \
#         protein.csv |
#     cut -d, -f1,4-8 \
#         >> $DATA_DIR/train_data/protein_data.csv
# done

get_seeded_random()
{
    seed="$1"
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

title "生成小训练数据集"
mkdir -p $DATA_DIR/small_train_data
cp $DATA_DIR/train_data/protein_data.csv $DATA_DIR/small_train_data/protein_data.csv
small_line_num=3000
> $DATA_DIR/small_train_data/DNA_data.csv
for accession in "${accessions[@]}"
do
    shuf -n $small_line_num \
        --random-source=<(get_seeded_random 63036) \
        $DATA_DIR/train_data/DNA_data/${accession}.csv \
        >> $DATA_DIR/small_train_data/DNA_data.csv
done
nl -w1 -v0 -s, $DATA_DIR/small_train_data/DNA_data.csv | sed '1i rn,index,DNA,distance' > $DATA_DIR/small_train_data/DNA_data.csv2
mv $DATA_DIR/small_train_data/DNA_data.csv2 $DATA_DIR/small_train_data/DNA_data.csv
