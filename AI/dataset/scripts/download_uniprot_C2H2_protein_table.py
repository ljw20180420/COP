#!/usr/bin/env python

import os

# 把运行文件夹切换为脚本文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sys

import requests

# available fields are at https://www.uniprot.org/help/return_fields
# path/to/uniprot_download.py \
#     field1:value1 \
#     field2:value2 \
#     field3:value3 \
#     ... \
#     fieldN:valueN

# Use /uniprotkb/stream instead of /uniprotkb/search. stream endpoint support up to 10 million entries.
url = (
    "https://rest.uniprot.org/uniprotkb/stream?query="
    + " AND ".join([f"({field})" for field in sys.argv[1:]])
    + "&format=tsv"
    + "&fields="
    + ",".join(
        [
            "accession",  # Entry
            "reviewed",  # Reviewed
            "id",  # Entry Name
            "protein_name",  # Protein names
            "gene_names",  # Gene Names
            "organism_name",  # Organism
            "length",  # Length
            "ft_zn_fing",  # Zinc finger
            "ft_coiled",  # Coiled coil
            "ft_compbias",  # Compositional bias
            "cc_domain",  # Domain [CC]
            "ft_domain",  # Domain [FT]
            "ft_repeat",  # Repeat
            "ft_region",  # Region
            "protein_families",  # Protein families
            "ft_motif",  # Motif
            "sequence",  # Sequence
        ]
    )
)

with open("uniprot_mouse_C2H2_protein.tsv", "w") as fd:
    fd.write(requests.get(url).text)
