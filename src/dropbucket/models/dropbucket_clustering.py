import random
from collections import Counter
import numpy as np
from scipy.stats import betabinom


class WeightedFuzzyMixtureModel:
    def __init__(
        self,
        num_cell: int,
        k: int,
        max_iter: int,
        epsilon: float,
        filtered_alt_matrix: np.ndarray,
        filtered_ref_matrix: np.ndarray,
        alt_counts: np.ndarray,
        total_counts: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
    ):
        self.NUM_CELL = num_cell
        self.K = k
        self.MAX_ITER = max_iter
        self.EPSILON = epsilon

        self.FILTERED_ALT_MATRIX = filtered_alt_matrix
        self.FILTERED_REF_MATRIX = filtered_ref_matrix

        self.ALT_COUNTS = alt_counts
        self.TOTAL_COUNTS = total_counts
        self.ROWS = rows
        self.COLS = cols

    def initialize_membership_matrix(self) -> np.ndarray:
        membership_matrix = np.random.rand(self.NUM_CELL, self.K)
        membership_matrix /= np.sum(membership_matrix, axis=1, keepdims=True)
        return membership_matrix

    def calculate_beta_bino_parameter(self, membership_matrix: np.ndarray, M: float) -> np.ndarray:
        membership_matrix_m = membership_matrix ** M
        weighted_alt = membership_matrix_m.T @ self.FILTERED_ALT_MATRIX
        weighted_ref = membership_matrix_m.T @ self.FILTERED_REF_MATRIX
        return np.stack([weighted_alt + 1, weighted_ref + 1], axis=-1)

    def update_membership_value(
        self,
        membership_matrix: np.ndarray,
        beta_bino_parameter: np.ndarray,
        P: float,
    ) -> np.ndarray:
        prior_prob = np.mean(membership_matrix, axis=0)
        prior_prob = np.clip(prior_prob, 0.001, 1.0)

        distances = np.empty((self.NUM_CELL, self.K), dtype=np.float64)

        alpha = beta_bino_parameter[:, :, 0]
        beta = beta_bino_parameter[:, :, 1]
        vaf = alpha / (alpha + beta)

        vaf_category = np.full_like(vaf, -1, dtype=int)
        vaf_category[vaf <= 0.2] = 0
        vaf_category[(vaf > 0.2) & (vaf < 0.8)] = 1
        vaf_category[vaf >= 0.8] = 2

        same_category_mask = np.all(vaf_category == vaf_category[0, :], axis=0)
        w = np.where(same_category_mask, 0.01, 1.0)

        for cluster in range(self.K):
            alpha_c = beta_bino_parameter[cluster, :, 0]
            beta_c = beta_bino_parameter[cluster, :, 1]

            likelihood = betabinom.pmf(
                self.ALT_COUNTS,
                self.TOTAL_COUNTS,
                alpha_c[self.COLS],
                beta_c[self.COLS],
            )
            likelihood = np.clip(likelihood, 1e-12, None)

            nll = -np.log(likelihood) * w[self.COLS]
            sum_per_cell = np.bincount(self.ROWS, weights=nll, minlength=self.NUM_CELL)
            distances[:, cluster] = sum_per_cell - np.log(prior_prob[cluster])

        new_membership = np.empty((self.NUM_CELL, self.K), dtype=np.float64)
        for cluster in range(self.K):
            ratio = np.power(distances[:, [cluster]] / distances, P)
            ratio_sum = np.sum(ratio, axis=1)
            new_membership[:, cluster] = 1.0 / ratio_sum

        return new_membership

    def calculate_log_likelihood(
        self,
        membership_matrix: np.ndarray,
        beta_bino_parameter: np.ndarray,
    ) -> float:
        nlls = np.empty((self.NUM_CELL, self.K), dtype=np.float64)

        for cluster in range(self.K):
            alpha_c = beta_bino_parameter[cluster, :, 0]
            beta_c = beta_bino_parameter[cluster, :, 1]

            likelihood = betabinom.pmf(
                self.ALT_COUNTS,
                self.TOTAL_COUNTS,
                alpha_c[self.COLS],
                beta_c[self.COLS],
            )
            likelihood = np.clip(likelihood, 1e-12, None)

            nll = -np.log(likelihood)
            nlls[:, cluster] = np.bincount(self.ROWS, weights=nll, minlength=self.NUM_CELL)

        weighted_nlls = membership_matrix * nlls
        return float(np.sum(weighted_nlls))

    def fit_once(self, seed_offset: int):
        random.seed(seed_offset)
        np.random.seed(seed_offset)

        membership_matrix = self.initialize_membership_matrix()

        M_start = 2.0
        M_end = 1.05
        prev_beta_bino_parameter = None

        for curr in range(1, self.MAX_ITER + 1):
            M = max(M_end, M_start - (curr / 50) * (M_start - M_end))
            P = 2.0 / (M - 1)

            beta_bino_parameter = self.calculate_beta_bino_parameter(membership_matrix, M)
            membership_matrix = self.update_membership_value(membership_matrix, beta_bino_parameter, P)

            if prev_beta_bino_parameter is not None:
                shift = 0.0
                for i in range(self.K):
                    shift += np.linalg.norm(
                        beta_bino_parameter[i, :, 0] - prev_beta_bino_parameter[i, :, 0]
                    )
                if shift < self.EPSILON:
                    break

            prev_beta_bino_parameter = beta_bino_parameter

        cluster_label = membership_matrix.argmax(axis=1)
        total_nll = self.calculate_log_likelihood(membership_matrix, beta_bino_parameter)

        return membership_matrix, cluster_label, curr, total_nll
    
    @staticmethod
    def canonicalize_label(label):
        freq = Counter(label)
        sorted_groups = sorted(freq.items(), key=lambda x: (x[1], x[0]))
        mapping = {orig: new for new, (orig, _) in enumerate(sorted_groups)}
        new_label = tuple(mapping[l] for l in label)
        return new_label, mapping

    def run_model(self, seed_offset: int):
        membership, label, crr, total_nll = self.fit_once(seed_offset)

        canonical_label, mapping = self.canonicalize_label(label)
        old_to_new = sorted(mapping.items(), key=lambda x: x[1])
        old_order = [old for old, _new in old_to_new]

        reordered_membership = membership[:, old_order]
        print(f"curr: {crr}, nll: {total_nll:.4f}")

        return canonical_label, reordered_membership, total_nll
