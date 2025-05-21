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
#     'organism_name:"Mus musculus"' \
#     3> uniprot_mouse_C2H2_protein.tsv

# title "下载蛋白结构"
# ./get_mmcif_from_alphafoldDB.py \
#     < uniprot_mouse_C2H2_protein.tsv

# title "计算蛋白二级结构"
# # dssp需要libcifpp: sudo apt install libcifpp-dev
# # dssp需要更新libcifpp的资源文件: https://github.com/PDB-REDO/dssp/issues/3
# # curl -o /var/cache/libcifpp/components.cif https://files.wwpdb.org/pub/pdb/data/monomers/components.cif
# # curl -o /var/cache/libcifpp/mmcif_pdbx.dic https://mmcif.wwpdb.org/dictionaries/ascii/mmcif_pdbx_v50.dic
# # curl -o /var/cache/libcifpp/mmcif_ma.dic https://github.com/ihmwg/ModelCIF/raw/master/dist/mmcif_ma.dic
# ./dssp.py \
#     3> secondary_structure.tsv

# title "去掉没有结构的蛋白"
# # 去掉uniprot和alphafoldDB蛋白长度不相同的蛋白
# # 在原来9种二级结构（包括没有结构）的基础上，标注KRAB和锌指蛋白结构
# ./parse_ft.py \
#     < uniprot_mouse_C2H2_protein.tsv \
#     4< secondary_structure.tsv \
#     3> protein.tsv

title "收集所有accession"
accessions=()
for narrowPeak in $(ls $DATA_DIR/sorted/*.sorted.narrowPeak)
do
    accession=$(basename ${narrowPeak%%.*})
    # 如果accession蛋白没有二级结构
    if ! grep -F $accession protein.tsv > /dev/null
    then
        printf "%s没有二级结构, 不处理\n" $accession >&2
        continue
    fi
    accessions+=($accession)
done

# title "把黑名单的peak去掉|把peak聚类"
# mkdir -p $DATA_DIR/clustered
# cluster_max_distance="-50"
# for accession in "${accessions[@]}"
# do
#     bedtools intersect -v \
#         -a $DATA_DIR/sorted/$accession.sorted.narrowPeak \
#         -b $GENOME_BLACK |
#     bedtools cluster \
#         -d $cluster_max_distance \
#         > $DATA_DIR/clustered/$accession.clustered.narrowPeak
# done

# title "每个聚类选择好的分位数|不选择最大值|防止异常值"
# mkdir -p $DATA_DIR/selected
# cluster_quantile=0.9
# for accession in "${accessions[@]}"
# do
#     ./peak_cluster_select.py \
#         < $DATA_DIR/clustered/$accession.clustered.narrowPeak \
#         $cluster_quantile \
#         > $DATA_DIR/selected/$accession.selected.narrowPeak
# done

# title "去掉太宽太窄的峰|去掉太显著太不显著的峰"
# mkdir -p $DATA_DIR/filtered
# for accession in "${accessions[@]}"
# do
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
#     accession="${accessions[$i]}"
#     dis_files=()
#     for accession2 in "${accessions[@]}"
#     do
#         if [ "${accession2}" != "${accession}" ]
#         then
#             bedtools closest -d -t first \
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
#         "${dis_files[@]}" \
#         > "$DATA_DIR/train_data/DNA_data/${accessions[$i]}.csv"
#     rm "${dis_files[@]}"
# done

# title "生成训练数据集的蛋白"
# mkdir -p $DATA_DIR/train_data
# >$DATA_DIR/train_data/protein_data.csv
# for ((i=0;i<${#accessions[@]};++i))
# do
#     grep -F "${accessions[$i]}" \
#         protein.tsv |
#     cut -f1,4,5 |
#     tr '\n\t' ',' \
#         >> $DATA_DIR/train_data/protein_data.csv

#     # 得到每个蛋白的锌指数量
#     zinc_num=$(
#         grep -F "${accessions[$i]}" \
#             < uniprot_mouse_C2H2_protein.tsv |
#             sed -r 's/^.+note="C2H2-type ([0-9]+)".+$/\1/' |
#             sed -r '/^[^1-9]/s/^.+$/1/'
#     )
#     printf "%d\n" $zinc_num >> $DATA_DIR/train_data/protein_data.csv
# done

title "生成小训练数据集"
get_seeded_random()
{
    seed="$1"
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

to_json()
{
    awk -F "," -v start_rn=$1 '
        {
            printf("{\"rn\": %d, \"index\": %d, \"dna\": \"%s\", \"distance\": [%d", start_rn + NR - 1, $1, $2, $3)
            for (i=4; i<=NF; ++i) {
                printf(",%d", $i)
            }
            printf("]}\n")
        }
    '
}

# mkdir -p $DATA_DIR/small_train_data/DNA_data
# cp $DATA_DIR/train_data/protein_data.csv $DATA_DIR/small_train_data/protein_data.csv
# small_line_num=3000
# start_rn=0
# for accession in "${accessions[@]}"
# do
#     shuf -n $small_line_num \
#         --random-source=<(get_seeded_random 63036) \
#         $DATA_DIR/train_data/DNA_data/${accession}.csv |
#     to_json $start_rn \
#         > $DATA_DIR/small_train_data/DNA_data/${accession}.json
#     total_line_num=$(wc -l < $DATA_DIR/train_data/DNA_data/${accession}.csv)
#     line_num=$(( small_line_num > total_line_num ? total_line_num : small_line_num ))
#     start_rn=$(( start_rn + line_num ))
# done

# title "生成极小程序测试用训练数据集"
# mkdir -p $DATA_DIR/train_data_for_test/DNA_data
# cp $DATA_DIR/train_data/protein_data.csv $DATA_DIR/train_data_for_test/protein_data.csv
# line_num_for_test=10
# start_rn=0
# for accession in "${accessions[@]}"
# do
#     shuf -n $line_num_for_test \
#         --random-source=<(get_seeded_random 63036) \
#         $DATA_DIR/train_data/DNA_data/${accession}.csv |
#     to_json $start_rn \
#         > $DATA_DIR/train_data_for_test/DNA_data/${accession}.json
#     total_line_num=$(wc -l < $DATA_DIR/train_data/DNA_data/${accession}.csv)
#     line_num=$(( line_num_for_test > total_line_num ? total_line_num : line_num_for_test ))
#     start_rn=$(( start_rn + line_num ))
# done

# title "生成极小程序测试用推理数据集"
# mkdir -p $DATA_DIR/inference_data_for_test/DNA_data
# cp $DATA_DIR/train_data/protein_data.csv $DATA_DIR/inference_data_for_test/protein_data.csv
# line_num_for_test=10
# for accession in "${accessions[@]}"
# do
#     printf "index,dna\n" \
#         > $DATA_DIR/inference_data_for_test/DNA_data/${accession}.csv
#     shuf -n $line_num_for_test \
#         --random-source=<(get_seeded_random 63036) \
#         $DATA_DIR/train_data/DNA_data/${accession}.csv |
#     cut -d, -f1-2 \
#         >> $DATA_DIR/inference_data_for_test/DNA_data/${accession}.csv
# done
