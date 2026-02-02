import subprocess
import argparse
import itertools
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.io import mmread
from sklearn.metrics import adjusted_rand_score
from .models.dropbucket_clustering import WeightedFuzzyMixtureModel as wfmm
from .models.dropbucket_doublet import is_doublet


def run_one_seed(
    seed: int,
    num_cell: int,
    k: int,
    max_iter: int,
    epsilon: float,
    filtered_alt_matrix,
    filtered_ref_matrix,
    alt_counts: np.ndarray,
    total_counts: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
):
    model = wfmm(
        num_cell=num_cell,
        k=k,
        max_iter=max_iter,
        epsilon=epsilon,
        filtered_alt_matrix=filtered_alt_matrix,
        filtered_ref_matrix=filtered_ref_matrix,
        alt_counts=alt_counts,
        total_counts=total_counts,
        rows=rows,
        cols=cols,
    )
    return model.run_model(seed)


def main(argv=None):
    t0_total = time.perf_counter()
    parser = argparse.ArgumentParser(description="set parameters for demultiplexing")
    parser.add_argument("-r", "--reference", type=str, required=True, help="reference fasta file")
    parser.add_argument("-b", "--bam", type=str, required=True, help="possorted_genome_bam file")
    parser.add_argument("-c", "--cellbarcode", type=str, required=True, help="cell barcode tsv file")
    parser.add_argument("-k", "--clusters", type=int, required=True, help="number of clusters")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="directory of output file")
    args = parser.parse_args(argv)

    cellranger_bam = args.bam
    reference = args.reference
    cellbarcodes = pd.read_csv(args.cellbarcode, sep="\t", header=None)

    MAX_ITER = 1000
    K = args.clusters
    EPSILON = 1e-05

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    chr = [x for i in range(1, 23) for x in ("--region", f"chr{i}")]

    t0 = time.perf_counter()

    freebayes_cmd = [
        "freebayes",
        "--bam", cellranger_bam,
        "--fasta-reference", reference,
        "--pooled-continuous",
        "-iXu",
        "-C", "10",
        "-q", "30",
        "-n", "1",
        "-m", "30",
        "--min-coverage", "20",
        "--skip-coverage", "100000",
        *chr
    ]

    output_vcf = f"{output_dir}/variants.vcf"

    with open(output_vcf, "w") as vcf_out:
        subprocess.run(freebayes_cmd, stdout=vcf_out, check=True)

    print(f"[TIME] freebayes : {int((e:=time.perf_counter()-t0)//3600):02d}:{int(e%3600//60):02d}:{int(e%60):02d}")

    t0 = time.perf_counter()

    vartrix_cmd = [
        "vartrix",
        "--bam", cellranger_bam,
        "--cell-barcodes", args.cellbarcode,
        "--vcf", output_vcf,
        "--fasta", reference,
        "--mapq", "30",
        "--scoring-method", "coverage",
        "--out-matrix", f"{output_dir}/alt.mtx",
        "--ref-matrix", f"{output_dir}/ref.mtx",
        "--threads", "4",
        "--umi",
    ]

    subprocess.run(vartrix_cmd, check=True)

    print(f"[TIME] vartrix : {int((e:=time.perf_counter()-t0)//3600):02d}:{int(e%3600//60):02d}:{int(e%60):02d}")

    ref_matrix = mmread(f"{output_dir}/ref.mtx").tocsr().T
    alt_matrix = mmread(f"{output_dir}/alt.mtx").tocsr().T
    total_matrix = ref_matrix + alt_matrix

    variant_cell_counts = (total_matrix > 0).sum(axis=0).A1
    keep_variant_idx = np.where(variant_cell_counts > 10)[0]
    filtered_total_matrix = total_matrix[:, keep_variant_idx]
    filtered_ref_matrix = ref_matrix[:, keep_variant_idx]
    filtered_alt_matrix = alt_matrix[:, keep_variant_idx]
    NUM_CELL, NUM_VARIANT = filtered_total_matrix.shape
    total_coo = filtered_total_matrix.tocoo()
    rows, cols, total_counts = total_coo.row, total_coo.col, total_coo.data
    alt_counts = filtered_alt_matrix[rows, cols].A1

    t0 = time.perf_counter()

    random_attempts = 50
    max_workers = 10
    seeds = range(random_attempts)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                run_one_seed,
                seeds,
                itertools.repeat(NUM_CELL),
                itertools.repeat(K),
                itertools.repeat(MAX_ITER),
                itertools.repeat(EPSILON),
                itertools.repeat(filtered_alt_matrix),
                itertools.repeat(filtered_ref_matrix),
                itertools.repeat(alt_counts),
                itertools.repeat(total_counts),
                itertools.repeat(rows),
                itertools.repeat(cols),
            )
        )

    print(f"[TIME] clustering ({random_attempts} seeds) : {int((e:=time.perf_counter()-t0)//3600):02d}:{int(e%3600//60):02d}:{int(e%60):02d}")


    label_results = [res[0] for res in results]
    membership_results = [res[1] for res in results]
    nll_results = [res[2] for res in results]

    label_tuples = [tuple(labels) for labels in label_results]
    label_counter = Counter(label_tuples)
    most_common_label, _ = label_counter.most_common(1)[0]
    best_idx = label_tuples.index(most_common_label)
    best_membership = membership_results[best_idx]
    best_nll = nll_results[best_idx]
    print("The best negative log likelihood : ", best_nll)

    ari_scores = []
    for i, j in combinations(range(random_attempts), 2):
        ari = adjusted_rand_score(label_results[i], label_results[j])
        ari_scores.append(ari)
    mean_ari = sum(ari_scores) / len(ari_scores)
    print("The mean of ARI : ", mean_ari)

    best_labels = np.argmax(best_membership, axis=1)
    ACTUAL_K = best_membership.shape[1]

    t0 = time.perf_counter()

    if ACTUAL_K == 1:
        result_df = pd.DataFrame({
            "cellbarcode": cellbarcodes.iloc[:, 0],
            "label": np.zeros(len(cellbarcodes), dtype=int),
        })
    else:
        doublet_membership_matrix, singlet_label, doublet_combo, curr = is_doublet(
            best_labels,
            best_membership,
            NUM_CELL,
            MAX_ITER,
            filtered_alt_matrix,
            filtered_ref_matrix,
            ACTUAL_K,
            alt_counts,
            total_counts,
            rows,
            cols,
            )
        print("Doubelt detection curr : ", curr)
        second_col = np.where(singlet_label == 1, best_labels, doublet_combo)
        result_df = pd.DataFrame({
            "cellbarcode": cellbarcodes.iloc[:, 0],
            "label": second_col,
        })

    print(f"[TIME] doublet detection : {int((e:=time.perf_counter()-t0)//3600):02d}:{int(e%3600//60):02d}:{int(e%60):02d}")

    save_path = os.path.join(output_dir, "result.tsv")
    result_df.to_csv(save_path, sep="\t", index=False, header=True)
    print(f"[TIME] TOTAL : {int((e:=time.perf_counter()-t0_total)//3600):02d}:{int(e%3600//60):02d}:{int(e%60):02d}")


if __name__ == "__main__":
    main()
