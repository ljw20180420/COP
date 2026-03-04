#!/bin/bash

# 切换运行路径到脚本路径
cd $( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 使用严格模式
set -euo pipefail

title() {
    printf "\n----------\n%s\n----------\n" "$1" >&2
}

download_mm9() {
    title "download mm9"
    mkdir -p genome
    pushd genome
    wget https://github.com/Boyle-Lab/Blacklist/raw/refs/heads/master/lists/Blacklist_v1/mm9-blacklist.bed.gz
    gzip -d mm9-blacklist.bed.gz
    wget https://hgdownload.cse.ucsc.edu/goldenpath/mm9/bigZips/mm9.chrom.sizes
    wget https://hgdownload.cse.ucsc.edu/goldenpath/mm9/bigZips/mm9.2bit
    twoBitToFa mm9.2bit mm9.fa
    popd
}

download_uniprot_C2H2_protein_table() {
    title "download uniprot C2H2 protein table"
    ./scripts/download_uniprot_C2H2_protein_table.py \
        'ft_zn_fing:C2H2' \
        'organism_name:"Mus musculus"'
}

download_alphafoldDB_mmcif() {
    title "download alphafoldDB mmcif"
    ./scripts/download_alphafoldDB_mmcif.py
}

infer_secondary_structure() {
    title "infer secondary structure"
    printf "accession,sequence,secondary_structure\n" >secondary_structure.csv
    for mmcif in $(find alphafoldDB_mmcif/ -name "*.mmcif")
    do
        stem=${mmcif##*/}
        stem=${stem%.mmcif}
        printf "%s," ${stem} >> secondary_structure.csv
        mkdssp --output-format dssp $mmcif | sed '1,/^  #/d' | cut  -c14 | tr -d '\n' >> secondary_structure.csv
        printf "," >> secondary_structure.csv
        mkdssp --output-format dssp $mmcif | sed '1,/^  #/d' | cut  -c17 | tr -d '\n' | tr ' ' '-' >> secondary_structure.csv
        printf "\n" >> secondary_structure.csv
    done
}

parse_protein_feature() {
    title "parse protein feature"
    ./scripts/parse_protein_feature.py
}

collect_accession() {
    local -n ref_accessions="$1"
    for narrowPeak in $(ls $DATA_DIR/sorted/*.sorted.narrowPeak)
    do
        accession=$(basename ${narrowPeak%%.*})
        ref_accessions+=($accession)
    done
}

clean_sorted_peak() {
    title "clean sorted peak"
    local accessions=()
    collect_accession accessions
    for accession in "${accessions[@]}"
    do
        printf "clean sorted narrowPeak for %s\n" $accession
        awk '
            NF == 10 {
                print
            }
        ' $DATA_DIR/sorted/$accession.sorted.narrowPeak \
        > $DATA_DIR/sorted/$accession.sorted.narrowPeak2
        mv $DATA_DIR/sorted/$accession.sorted.narrowPeak2 $DATA_DIR/sorted/$accession.sorted.narrowPeak
    done
}

remove_black_peak_and_cluster_peak() {
    title "remove black peak and cluster peak"
    local accessions=()
    collect_accession accessions
    mkdir -p $DATA_DIR/clustered
    cluster_max_distance="-50"
    for accession in "${accessions[@]}"
    do
        if [ -f "$DATA_DIR/clustered/$accession.clustered.narrowPeak" ]
        then
            continue
        fi
        printf "calculate peak cluster for %s\n" $accession
        bedtools intersect -sorted -v \
            -a $DATA_DIR/sorted/$accession.sorted.narrowPeak \
            -b <(
                bedtools sort -i genome/mm9-blacklist.bed
            ) |
        bedtools cluster \
            -d $cluster_max_distance \
            > $DATA_DIR/clustered/$accession.clustered.narrowPeak
    done
}

choose_peak_by_pvalue_quantile_from_cluster() {
    title "choose peak by pvalue quantile from cluster"
    local accessions=()
    collect_accession accessions
    mkdir -p $DATA_DIR/selected
    cluster_quantile=0.9
    for accession in "${accessions[@]}"
    do
        if [ -f "$DATA_DIR/selected/$accession.selected.narrowPeak" ]
        then
            continue
        fi
        printf "select peak for %s\n" $accession
        ./scripts/choose_peak_by_pvalue_quantile_from_cluster.py \
            < $DATA_DIR/clustered/$accession.clustered.narrowPeak \
            $cluster_quantile \
            > $DATA_DIR/selected/$accession.selected.narrowPeak
    done
}

filter_peak_by_width_and_pvalue() {
    title "filter peak by width and pvalue"
    local accessions=()
    collect_accession accessions
    mkdir -p $DATA_DIR/filtered
    for accession in "${accessions[@]}"
    do
        if [ -f "$DATA_DIR/filtered/$accession.filtered.narrowPeak" ]
        then
            continue
        fi
        printf "filtered peak for %s\n" $accession
        wlb=$(
            awk '{print $3 - $2}' \
                < $DATA_DIR/selected/$accession.selected.narrowPeak |
            sort -n |
            perl -e '$d=0.1;@l=<>;print $l[int($d*$#l)]'
        )
        wub=$(
            awk '{print $3 - $2}' \
                < $DATA_DIR/selected/$accession.selected.narrowPeak |
            sort -n |
            perl -e '$d=0.9;@l=<>;print $l[int($d*$#l)]'
        )
        plb=$(
            awk '{print $8}' \
                < $DATA_DIR/selected/$accession.selected.narrowPeak |
            sort -g |
            perl -e '$d=0.1;@l=<>;print $l[int($d*$#l)]'
        )
        pub=$(
            awk '{print $8}' \
                < $DATA_DIR/selected/$accession.selected.narrowPeak |
            sort -g |
            perl -e '$d=0.9;@l=<>;print $l[int($d*$#l)]'
        )
        awk -v wlb=$wlb -v wub=$wub -v plb=$plb -v pub=$pub '$3 - $2 <= wub && $3 - $2 >= wlb && $8 >= plb && $8 <= pub {print}' \
            < $DATA_DIR/selected/$accession.selected.narrowPeak \
            > $DATA_DIR/filtered/$accession.filtered.narrowPeak
    done
}

resize_peak_and_sort_by_summit() {
    title "resize peak and sort by summit"
    local accessions=()
    collect_accession accessions
    mkdir -p $DATA_DIR/sized
    seq_len=256
    for accession in "${accessions[@]}"
    do
        if [ -f "$DATA_DIR/sized/$accession.sized.narrowPeak" ]
        then
            continue
        fi
        printf "resize peak for %s\n" $accession
        bedClip \
            <(
                awk -v seq_len=$seq_len '
                    {
                        start = $2
                        end = $3
                        summit = $10
                        new_start = start + summit - int(seq_len / (end - start) * summit)
                        new_end = new_start + seq_len
                        new_summit = start + summit
                        printf("%s\t%d\t%d\t%d\n", $1, new_start, new_end, new_summit)
                    }
                ' $DATA_DIR/filtered/$accession.filtered.narrowPeak |
                sort -k1,1 -k4,4n
            ) \
            genome/mm9.chrom.sizes \
            $DATA_DIR/sized/$accession.sized.narrowPeak
    done
}

extract_peak_site_sequence() {
    title "extract peak site sequence"
    local accessions=()
    collect_accession accessions
    mkdir -p $DATA_DIR/positive
    for accession in "${accessions[@]}"
    do
        if [ -f "$DATA_DIR/positive/$accession.positive" ]
        then
            continue
        fi
        printf "extract peak sequence for %s\n" $accession
        # --line-width 0 防止fasta换行
        paste \
            $DATA_DIR/sized/$accession.sized.narrowPeak \
            <(
                seqkit subseq \
                    < genome/mm9.fa \
                    --update-faidx \
                    --line-width 0 \
                    --bed $DATA_DIR/sized/$accession.sized.narrowPeak |
                sed -nr '2~2{y/acgtn/ACGTN/; p}'
            ) |
        grep -vE "\sN|[ACGT]N" \
            > $DATA_DIR/positive/$accession.positive
    done
}

get_summit_sorted_peak_before_filter() {
    title "get summit sorted peak before filter"
    local accessions=()
    collect_accession accessions
    mkdir -p "$DATA_DIR/before_filter"
    for accession in "${accessions[@]}"
    do
        if [ -f "$DATA_DIR/before_filter/${accession}.bed" ]
        then
            continue
        fi
        awk '
            {
                printf("%s\t%s\t%s\n", $1, $2 + $10, $2 + $10 + 1)
            }
        ' "$DATA_DIR/selected/${accession}.selected.narrowPeak" |
        bedtools sort \
            > "$DATA_DIR/before_filter/${accession}.bed"
    done
}

get_protein_pairwise_closest_peak_distance() {
    title "get protein pairwise closest peak distance"
    local accessions=()
    collect_accession accessions
    mkdir -p $DATA_DIR/train_data
    local dis_files
    for accession in "${accessions[@]}"
    do
        if [ -f "$DATA_DIR/train_data/${accession}.csv" ]
        then
            continue
        fi
        printf "calculate the closest peak from other proteins for %s\n" ${accession}
        dis_files=()
        for accession2 in "${accessions[@]}"
        do
            if [ "${accession2}" != "${accession}" ]
            then
                bedtools closest -sorted -d -t first \
                    -a <(
                        awk '
                            {
                                printf("%s\t%s\t%s\n", $1, $4, $4 + 1)
                            }
                        ' $DATA_DIR/positive/${accession}.positive
                    ) \
                    -b $DATA_DIR/before_filter/${accession2}.bed |
                awk -F $'\t' -v accession=${accession2} '
                    BEGIN{print accession}
                    {print $7}
                ' > $DATA_DIR/train_data/${accession}_${accession2}
            else
                awk -v accession=${accession2} '
                    BEGIN{print accession}
                    {print 0}
                ' $DATA_DIR/positive/${accession}.positive \
                    > $DATA_DIR/train_data/${accession}_${accession2}
            fi
            dis_files+=($DATA_DIR/train_data/${accession}_${accession2})
        done
        paste -d, \
            <(
                awk -v accession=${accession} '
                    BEGIN{
                        printf("protein,DNA\n")
                    }
                    {
                        printf("%s,%s\n", accession, $5)
                    }
                ' "$DATA_DIR/positive/${accession}.positive"
            ) "${dis_files[@]}" \
            > "$DATA_DIR/train_data/${accession}.csv"
        rm "${dis_files[@]}"
    done
}

get_seeded_random()
{
    seed="$1"
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

generate_small_data() {
    title "generate small data"
    local accessions=()
    collect_accession accessions
    small_line_num=3000
    scripts/generate_small_data.py ${small_line_num} "${accessions[@]}"
}

split_and_balance_small_data() {
    title "split and balance small data"
    minimal_unbind_summit_distance=300
    validation_ratio=0.05
    test_ratio=0.05
    seed=63036
    scripts/split_and_balance_small_data ${minimal_unbind_summit_distance} ${validation_ratio} ${test_ratio} ${seed}
}

# download_mm9

# download_uniprot_C2H2_protein_table

# download_alphafoldDB_mmcif

# infer_secondary_structure

# parse_protein_feature

# clean_sorted_peak

# remove_black_peak_and_cluster_peak

# choose_peak_by_pvalue_quantile_from_cluster

# filter_peak_by_width_and_pvalue

# resize_peak_and_sort_by_summit

# extract_peak_site_sequence

# get_summit_sorted_peak_before_filter

get_protein_pairwise_closest_peak_distance

# generate_small_train_data

split_and_balance_small_data
