import argparse
import csv
import itertools
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.stats import betabinom
from sklearn.metrics import adjusted_rand_score

parser = argparse.ArgumentParser(description="Set parameters for clustering.")
parser.add_argument("--mtx_path", type=str, required=True, help="Path to the mtx file")
parser.add_argument("--cellbarcode_path", type=str, required=True, help="Path to the cell barcode file")
parser.add_argument("--k", type=int, required=True, help="Number of clusters")
parser.add_argument("--output_dir", type=str, required=True, help="Directory of output file")
args = parser.parse_args()

##### 1. Data loading
### 1) argumnets
mtx_path = args.mtx_path
ref_matrix = mmread(f"{mtx_path}/ref.mtx").tocsr().T
alt_matrix = mmread(f"{mtx_path}/alt.mtx").tocsr().T
total_matrix = ref_matrix + alt_matrix

### 2) filter matrix (SNV > 10)
variant_cell_counts = (total_matrix > 0).sum(axis=0).A1
keep_variant_idx = np.where(variant_cell_counts > 10)[0]
filtered_total_matrix = total_matrix[:, keep_variant_idx]
filtered_ref_matrix = ref_matrix[:, keep_variant_idx]
filtered_alt_matrix = alt_matrix[:, keep_variant_idx]
NUM_CELL, NUM_VARIANT = filtered_total_matrix.shape
total_coo = filtered_total_matrix.tocoo()
rows, cols, total_counts = total_coo.row, total_coo.col, total_coo.data  
alt_counts = filtered_alt_matrix[rows, cols].A1
cellbarcodes = pd.read_csv(args.cellbarcode_path, sep="\t", header=None)

MAX_ITER = 1000
K = args.k
EPSILON = 1e-05

output_dir=args.output_dir
os.makedirs(output_dir, exist_ok=True)

##### 2. Weighted Fuzzy Mixture Model (WFMM)
### 1) initialize the membership matrix with random values
def initialize_membership_matrix():
    membership_matrix = np.random.rand(NUM_CELL, K)
    membership_matrix /= np.sum(membership_matrix, axis=1, keepdims=True)
    return membership_matrix

### 2) Step 1: calculate the paramter of beta-binomial distribution
def calculate_beta_bino_parameter(membership_matrix,M):
    membership_matrix_m = membership_matrix ** M
    weighted_alt = membership_matrix_m.T @ filtered_alt_matrix
    weighted_ref = membership_matrix_m.T @ filtered_ref_matrix
    return np.stack([weighted_alt + 1, weighted_ref + 1], axis=-1)

### 3) Step 2: update the membership values using the cluster centers
def update_membership_value(membership_matrix, beta_bino_parameter, P):
    prior_prob = np.mean(membership_matrix, axis=0)
    prior_prob = np.clip(prior_prob, 0.001, 1.0)
    distances = np.empty((NUM_CELL, K), dtype=np.float64)
    alpha = beta_bino_parameter[:, :, 0]
    beta = beta_bino_parameter[:, :, 1]
    vaf = alpha / (alpha + beta)
    vaf_category = np.full_like(vaf, -1, dtype=int)
    vaf_category[vaf <= 0.2] = 0
    vaf_category[(vaf > 0.2) & (vaf < 0.8)] = 1
    vaf_category[vaf >= 0.8] = 2
    same_category_mask = np.all(vaf_category == vaf_category[0, :], axis=0)
    w = np.where(same_category_mask, 0.01, 1.0)
    for cluster in range(K):
        alpha_c = beta_bino_parameter[cluster, :, 0]
        beta_c = beta_bino_parameter[cluster, :, 1]
        likelihood = betabinom.pmf(alt_counts, total_counts, alpha_c[cols], beta_c[cols])
        likelihood = np.clip(likelihood, 1e-12, None)
        nll = -np.log(likelihood) * w[cols]
        sum_per_cell = np.bincount(rows, weights=nll, minlength=NUM_CELL)
        distances[:, cluster] = sum_per_cell - np.log(prior_prob[cluster])
    membership_matrix = np.empty((NUM_CELL, K), dtype=np.float64)
    for cluster in range(K):
        ratio = np.power(distances[:, [cluster]] / distances, P)
        ratio_sum = np.sum(ratio, axis=1)
        membership_matrix[:, cluster] = 1.0 / ratio_sum
    return membership_matrix

### 4) total likelihood
def calculate_log_likelihood(membership_matrix, beta_bino_parameter):
    nlls = np.empty((NUM_CELL, K), dtype=np.float64)
    for cluster in range(K):
        alpha_c = beta_bino_parameter[cluster, :, 0]
        beta_c = beta_bino_parameter[cluster, :, 1]
        likelihood = betabinom.pmf(alt_counts, total_counts, alpha_c[cols], beta_c[cols])
        likelihood = np.clip(likelihood, 1e-12, None)
        nll = -np.log(likelihood)
        nlls[:,cluster] = np.bincount(rows, weights=nll, minlength=NUM_CELL)
    weighted_nlls = membership_matrix * nlls
    return np.sum(weighted_nlls) 

### 5) the final function
def WFMM(seed_offset):
    random.seed(seed_offset)
    np.random.seed(seed_offset)
    # (1) Random Initialization of Membership Matrix
    membership_matrix = initialize_membership_matrix()
    # (2) Start iteration
    M_start = 2.0
    M_end = 1.05
    prev_beta_bino_prameter = None
    for curr in range(1, MAX_ITER + 1):
        M = max(M_end, M_start - (curr / 50) * (M_start - M_end))
        P = 2.0 / (M - 1)
        # (3) Step 1
        beta_bino_parameter = calculate_beta_bino_parameter(membership_matrix,M)
        # (4) Step 2
        membership_matrix = update_membership_value(membership_matrix,beta_bino_parameter,P)
        # (5) Cluster Reallocation based on Step 2
        cluster_label = membership_matrix.argmax(axis=1)
        # (6) Check with previous cluster center
        if prev_beta_bino_prameter is not None:
            shift = 0.0
            for i in range(K):
                shift += np.linalg.norm(beta_bino_parameter[i,:,0] - prev_beta_bino_prameter[i,:,0])
            if shift < EPSILON:
                break
        prev_beta_bino_prameter = beta_bino_parameter
    total_nll = calculate_log_likelihood(membership_matrix, beta_bino_parameter)
    return membership_matrix, cluster_label, curr, total_nll

##### 3. 50 Execution and Select best performance
def canonicalize_label(label):
    freq = Counter(label)
    sorted_groups = sorted(freq.items(), key=lambda x: (x[1], x[0]))
    mapping = {orig: new for new, (orig, _) in enumerate(sorted_groups)}
    new_label = tuple(mapping[l] for l in label)
    return new_label, mapping

def run_model(seed_offset):
    membership, label, crr, total_nll = WFMM(seed_offset)
    canonical_label, mapping = canonicalize_label(label)
    old_to_new = sorted(mapping.items(), key=lambda x: x[1])
    old_order = [old for old, new in old_to_new]
    reordered_membership = membership[:, old_order]
    print(f"curr: {crr}, nll: {total_nll:.4f}")
    return canonical_label, reordered_membership, total_nll

random_attempts = 50
max_workers = 10
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    results = list(executor.map(run_model, range(random_attempts)))

label_results = [res[0] for res in results]          
membership_results = [res[1] for res in results]     
nll_results = [res[2] for res in results]  

label_tuples = [tuple(labels) for labels in label_results]
label_counter = Counter(label_tuples)
most_common_label, _ = label_counter.most_common(1)[0]
best_idx = label_tuples.index(most_common_label)
best_labels = label_results[best_idx]
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

##### 4. doublet detection
def doublet_detection(best_labels, best_membership, singlet_label):
    selected_cells = np.where(singlet_label == 1)[0]
    selected_best_membership = best_membership[selected_cells]
    filtered_alt_matrix_selected = filtered_alt_matrix[selected_cells]
    filtered_ref_matrix_selected = filtered_ref_matrix[selected_cells]
    selected_best_membership_m = selected_best_membership ** 2
    weighted_alt = selected_best_membership_m.T @ filtered_alt_matrix_selected
    weighted_ref = selected_best_membership_m.T @ filtered_ref_matrix_selected
    beta_bino_parameter = np.stack([weighted_alt + 1, weighted_ref + 1], axis=-1)
    alpha = beta_bino_parameter[:, :, 0]
    beta = beta_bino_parameter[:, :, 1]
    vaf = alpha / (alpha + beta)
    vaf_category = np.full_like(vaf, -1, dtype=int)
    vaf_category[vaf <= 0.2] = 0
    vaf_category[(vaf > 0.2) & (vaf < 0.8)] = 1
    vaf_category[vaf >= 0.8] = 2
    same_category_mask = np.all(vaf_category == vaf_category[0, :], axis=0)
    w = np.where(same_category_mask, 0.01, 1.0)
    # (1) Singlet distance
    prior_prob = np.mean(selected_best_membership, axis=0)
    prior_prob = np.clip(prior_prob, 0.0001, 1.0)
    distances = np.empty((NUM_CELL, ACTUAL_K), dtype=np.float64)
    for cluster in range(ACTUAL_K):
        alpha_c = beta_bino_parameter[cluster, :, 0]
        beta_c = beta_bino_parameter[cluster, :, 1]
        likelihood = betabinom.pmf(alt_counts, total_counts, alpha_c[cols], beta_c[cols])
        likelihood = np.clip(likelihood, 1e-12, None)
        nll = -np.log(likelihood) * w[cols]
        sum_per_cell = np.bincount(rows, weights=nll, minlength=NUM_CELL)
        distances[:, cluster] = sum_per_cell - np.log(prior_prob[cluster])
    singlet_distance = distances[np.arange(NUM_CELL), best_labels]
    # (2) Doublet distance
    comb_list = list(itertools.combinations(range(ACTUAL_K), 2))
    doublet_distances = np.empty((NUM_CELL, len(comb_list)), dtype=np.float64)
    for idx, (cluster_a, cluster_b) in enumerate(comb_list):
        alpha_a = beta_bino_parameter[cluster_a, :, 0]
        beta_a  = beta_bino_parameter[cluster_a, :, 1]
        alpha_b = beta_bino_parameter[cluster_b, :, 0]
        beta_b  = beta_bino_parameter[cluster_b, :, 1]
        doublet_vaf = (alpha_a + alpha_b) / (alpha_a + alpha_b + beta_a + beta_b)
        cell_depth = alpha_a + beta_a
        doublet_alpha = doublet_vaf * cell_depth
        doublet_beta = cell_depth - doublet_alpha
        doublet_likelihood =  betabinom.pmf(alt_counts, total_counts, doublet_alpha[cols], doublet_beta[cols])
        doublet_likelihood = np.clip(doublet_likelihood, 1e-12, None)
        doublet_nll = -np.log(doublet_likelihood) * w[cols]
        doublet_sum_per_cell = np.bincount(rows, weights=doublet_nll, minlength=NUM_CELL)
        doublet_distances[:, idx] = doublet_sum_per_cell - np.log(0.1)
    doublet_best_idx = np.argmin(doublet_distances, axis=1)
    doublet_distance = doublet_distances[np.arange(NUM_CELL), doublet_best_idx]
    doublet_pairs = [f"{a}/{b}" for a, b in comb_list]
    doublet_labels = np.array([doublet_pairs[i] for i in doublet_best_idx])
    # (3) Total distance
    total_distance = np.stack([singlet_distance, doublet_distance], axis=1)
    ratio = (total_distance[:, :, None] / total_distance[:, None, :]) ** 2
    doublet_membership_matrix = 1.0 / ratio.sum(axis=2)
    singlet_label = (doublet_membership_matrix[:, 0] > doublet_membership_matrix[:, 1]).astype(float)
    doublet_combo = np.where(singlet_label == 1, "", doublet_labels)
    return doublet_membership_matrix, singlet_label, doublet_combo

def is_doublet():
    singlet_label = np.ones(NUM_CELL)
    prev_singlet_label = None
    doublet_combo = None
    for curr in range(1, MAX_ITER + 1):
        doublet_membership_matrix, singlet_label, doublet_combo = doublet_detection(best_labels, best_membership, singlet_label)
        if prev_singlet_label is not None:
            shift = np.linalg.norm(prev_singlet_label - singlet_label)
            if shift == 0:
                break
        prev_singlet_label = singlet_label
    return doublet_membership_matrix, singlet_label, doublet_combo, curr

if ACTUAL_K == 1:
    result_df = pd.DataFrame({
        'cellbarcode': cellbarcodes.iloc[:, 0],
        'label': np.zeros(len(cellbarcodes), dtype=int)
    })
else:
    doublet_membership_matrix, singlet_label, doublet_combo, curr = is_doublet()
    print("Doubelt detection curr : ", curr)
    second_col = np.where(singlet_label == 1, best_labels, doublet_combo)
    result_df = pd.DataFrame({
        'cellbarcode': cellbarcodes.iloc[:, 0],
        'label': second_col
    })

save_path = os.path.join(output_dir, "result.tsv")
result_df.to_csv(save_path, sep="\t", index=False, header=True)
