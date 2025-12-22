### dropbucket
a robust SNP-based demultiplexing tool designed to accurately assign cells to donors in pooled scRNA-seq datasets, even under extremely unbalanced pooling condition and samples with genetically similar genotypes.

### Usage
```bash
   python dropbucket_pipeline.py \
     --mtx_path dir/mtx_path \
     --cellbarcode_path dir/barcodes.tsv \
     --k 8 \
     --output_dir results
```
### Parameters
 --mtx_path, Path to the `.mtx` file \
 --cellbarcode_path, Path to the cell barcode text file \
 --k, Number of clusters \
 --output_dir, Directory to save the output files
