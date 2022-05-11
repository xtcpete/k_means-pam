import numpy as np
import math
import matplotlib.pyplot as plt


class clustering_eval_metrics:
    def __init__(self, labels: list, true_labels=None):  # label must be between 0 to number_of_labels - 1
        """
        labels: list of labels that are from the clustering algorithm, for example, from the get_labels method in
                the above k_means class
        true_labels: the ground truth of the labels
        """
        self.labels = np.array(labels)
        self.true_labels = true_labels
        self.cmat = None
        self.ars = None

    def set_true_labels(self, true_labels):
        self.true_labels = np.array(true_labels)

    def contingency_matrix(self):
        """
        return a contingency matrix
        """

        classes, class_idx = np.unique(self.true_labels, return_inverse=True)
        clusters, cluster_idx = np.unique(self.labels, return_inverse=True)
        n_classes = classes.shape[0]
        n_clusters = clusters.shape[0]

        ctable = np.zeros(n_classes ** 2, dtype=np.dtype(int))
        reindex = np.dot(np.stack((self.labels, self.true_labels)).transpose(),
                         np.array([n_classes, 1]))
        idx, count = np.unique(reindex, return_counts=True)

        for i, c in zip(idx, count):
            ctable[int(i)] = c

        self.cmat = ctable.reshape(n_classes, n_clusters)

        return self.cmat

    def adjusted_rand_score(self):
        """
        return the adjusted_rand_score
        """
        index = 0
        sum_row = 0
        sum_col = 0
        n = sum([sum(subrow) for subrow in self.cmat])

        for i in range(len(self.cmat)):
            col_ = 0
            for j in range(len(self.cmat)):
                index += math.comb(self.cmat[i][j], 2)
                col_ += self.cmat[j][i]

            sum_row += math.comb(sum(self.cmat[i]), 2)
            sum_col += math.comb(col_, 2)

        expected_index = (sum_row * sum_col)
        expected_index = expected_index / math.comb(n, 2)
        max_index = (sum_row + sum_col)
        max_index = max_index / 2

        self.ars = (index - expected_index) / (max_index - expected_index)

        return self.ars