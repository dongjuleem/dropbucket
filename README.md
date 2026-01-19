[![PyPI version](https://badge.fury.io/py/dropbucket.svg)](https://pypi.org/project/dropbucket/)

## dropbucket
Robust SNP-based demultiplexing tool designed to accurately assign cells to donors in pooled scRNA-seq, even under extremely unbalanced pooling condition and samples with genetically similar genotypes.


## Prerequisites
dropbucket requires the following external tools
- FreeBayes
- Vartrix


## Installation
dropbucket is available through pypi
```
pip install dropbucket
```


## Usage
```
usage: dropbucket [-h] -r REFERENCE -b BAM -c CELLBARCODE -k CLUSTERS -o OUTPUT_DIR

set parameters for demultiplexing

optional arguments:
  -h, --help            show this help message and exit
  -r REFERENCE, --reference REFERENCE
                        reference fasta file
  -b BAM, --bam BAM     possorted_genome_bam file
  -c CELLBARCODE, --cellbarcode CELLBARCODE
                        cell barcode tsv file
  -k CLUSTERS, --clusters CLUSTERS
                        number of clusters
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        directory of output file
```
