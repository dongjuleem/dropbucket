# dropbucket
# ------------------------------------------------------------
# A robust demultiplexing tool for pooled scRNA-seq datasets.
# Accurately assigns single cells to donors even under
# extremely unbalanced pooling conditions or with genetically
# similar samples.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Usage
# ------------------------------------------------------------
# Run Dropbucket with the required arguments:
#
#   python dropbucket.py \
#     --mtx_path data/matrix.mtx \
#     --cellbarcode_path data/barcodes.txt \
#     --k 8 \
#     --output_dir results/
#
# ------------------------------------------------------------
# ⚙️ Parameters
# ------------------------------------------------------------
# | Argument              | Description                              
# |------------------------|-----------------------------------
# | --mtx_path             | Path to the input `.mtx` file
# | --cellbarcode_path     | Path to the cell barcode text file
# | --k                    | Number of clusters
# | --output_dir           | Directory to save the output files

# ------------------------------------------------------------
# Output Files
# ------------------------------------------------------------
# The following files will be generated in `--output_dir`:
#
# - clusters.csv  → Cluster assignments for each cell

