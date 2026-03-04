#!/bin/bash

## 1. Input
 
# Arguments
# $1: samples_tsv.tsv
#   - Format: [BAM]   [BARCODE]   [NUM_CELLS]
# $2: work_dir 
#   - Path to working directory 
# $3: id
#   - Output file prefix (used to name final BAM/TSV files)
# $4: cores
#   - Number of cores to use

samples_tsv=$1
work_dir=$2
id=$3
cores=$4

mkdir -p $work_dir

bams=()
barcodes=()
cell_nums=()

while read -r bam bc n; do

    [[ -z "$bam" ]] && continue 
    bams+=("$bam")
    barcodes+=("$bc")
    cell_nums+=("$n")

done < "$samples_tsv"

## 2. Execution
# For each sample:
# 1) Randomly select NUM_CELLS barcodes from the barcode file
# 2) Subset the BAM to reads whose CB tag is in the selected barcode list
# 3) Prefix CB values and barcode list with the sample ID to make them unique across samples
# After all samples:
# 4) Merge prefixed BAMs and concatenate prefixed barcode lists
# 5) Remove @RG lines from the merged BAM header and sort/index the final BAM

merged_bams=()
merged_barcodes=()

for i in "${!bams[@]}"; do
  sample_id=$((i+1))
  bam=${bams[i]}
  barcode=${barcodes[i]}
  cell_num=${cell_nums[i]}
  
  cat "$barcode" | shuf -n "$cell_num" > "$work_dir/selected_barcodes_sample${sample_id}.tsv"

  subset-bam \
    --bam "$bam" \
    --cell-barcodes "$work_dir/selected_barcodes_sample${sample_id}.tsv" \
    --out-bam "$work_dir/selected_sample${sample_id}.bam" \
    --cores "$cores"

  awk -v p="$sample_id" '{print p $0}' "$work_dir/selected_barcodes_sample${sample_id}.tsv" > "$work_dir/selected_barcodes_sample${sample_id}_prefix.tsv"

  python3 - <<EOF
import pysam
bam_in = pysam.AlignmentFile("${work_dir}/selected_sample${sample_id}.bam", "rb")
bam_out = pysam.AlignmentFile("${work_dir}/selected_sample${sample_id}_prefixed.bam", "wb", template=bam_in)

for read in bam_in:
    if read.has_tag("CB"):
        read.set_tag("CB", "${sample_id}" + read.get_tag("CB"), value_type='Z')
    if read.has_tag("RG"):
        read.tags = [tag for tag in read.tags if tag[0] != "RG"]
    bam_out.write(read)

bam_in.close()
bam_out.close()
EOF
  
  merged_bams+=("$work_dir/selected_sample${sample_id}_prefixed.bam")
  merged_barcodes+=("$work_dir/selected_barcodes_sample${sample_id}_prefix.tsv")
done

bam_merged_output="$work_dir/${id}.bam"
samtools merge "$bam_merged_output" "${merged_bams[@]}"
cat "${merged_barcodes[@]}" > "$work_dir/${id}_cellbarcode.tsv"

samtools view -H "$bam_merged_output" | grep -v "^@RG" > "$work_dir/tmp_header.sam"
samtools reheader "$work_dir/tmp_header.sam" "$bam_merged_output" > "$work_dir/${id}.noRG.bam"

samtools sort -o "$work_dir/${id}.noRG.sort.bam" "$work_dir/${id}.noRG.bam"
samtools index "$work_dir/${id}.noRG.sort.bam"

# rm $work_dir/selected_*
# rm $work_dir/tmp_header.sam
# rm $work_dir/${id}.bam
# rm $work_dir/${id}.noRG.bam