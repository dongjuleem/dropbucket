import itertools
import numpy as np
from scipy.stats import betabinom


def doublet_detection(
    best_labels,
    best_membership,
    singlet_label,
    filtered_alt_matrix,
    filtered_ref_matrix,
    NUM_CELL,
    ACTUAL_K,
    alt_counts,
    total_counts,
    rows,
    cols,
):
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
        beta_a = beta_bino_parameter[cluster_a, :, 1]
        alpha_b = beta_bino_parameter[cluster_b, :, 0]
        beta_b = beta_bino_parameter[cluster_b, :, 1]
        doublet_vaf = (alpha_a + alpha_b) / (alpha_a + alpha_b + beta_a + beta_b)
        cell_depth = alpha_a + beta_a
        doublet_alpha = doublet_vaf * cell_depth
        doublet_beta = cell_depth - doublet_alpha
        doublet_likelihood = betabinom.pmf(alt_counts, total_counts, doublet_alpha[cols], doublet_beta[cols])
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


def is_doublet(
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
):
    singlet_label = np.ones(NUM_CELL)
    prev_singlet_label = None
    doublet_combo = None
    for curr in range(1, MAX_ITER + 1):
        doublet_membership_matrix, singlet_label, doublet_combo = doublet_detection(
            best_labels,
            best_membership,
            singlet_label,
            filtered_alt_matrix,
            filtered_ref_matrix,
            NUM_CELL,
            ACTUAL_K,
            alt_counts,
            total_counts,
            rows,
            cols,
        )
        if prev_singlet_label is not None:
            shift = np.linalg.norm(prev_singlet_label - singlet_label)
            if shift == 0:
                break

        prev_singlet_label = singlet_label

    return doublet_membership_matrix, singlet_label, doublet_combo, curr
