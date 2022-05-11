import numpy as np
import math
import matplotlib.pyplot as plt
import k_means


class pam(k_means):
    def __init__(self, data: np.ndarray, d: int, k: int, tol: float, max_iter: int, p: float):
        """
        p is the parameter for the L_p norm
        """
        self.k_means = k_means(data, d=d, k=k, tol=tol, max_iter=max_iter)
        self.p = p
        self.medoid_cost = np.zeros(k)

    def search(self):
        distance = np.zeros((self.k_means.n, self.k_means.k))
        for k in range(self.k_means.k):
            distance[:, k] = np.linalg.norm(self.k_means.data - self.k_means.centers[k, :], self.p, axis=1)
        self.medoid_cost = np.sum(distance, axis=0)
        classsifications = np.argmin(distance, axis=1)
        self.k_means.next_centers = np.zeros((self.k_means.k, self.k_means.d))
        for k in np.unique(classsifications):
            self.k_means.partitions[k] = np.where(classsifications == k)[0]
            original_cost = self.medoid_cost[k]
            for j in self.k_means.partitions[k]:
                current_cost = 0
                for d in self.k_means.partitions[k]:
                    current_cost += np.linalg.norm(self.k_means.data[j] - self.k_means.data[d], self.p)
                if current_cost < original_cost:
                    self.k_means.next_centers[k, :] = self.k_means.data[j]
                    original_cost = current_cost

    def fit_model(self):
        """
        function to fit the k-means algorithms using the above functions
        """
        self.k_means.initialize_centers(1)
        self.counter = 0
        for i in range(self.k_means.max_iter):
            self.search()
            if self.k_means.is_updated():
                if i == self.k_means.max_iter:
                    print("Maximum Number of Iteration Reached!")
            else:
                print("Convergence Reached! Number of Iterations: {}".format(i))
                break
        self.get_labels()

    def get_labels(self):
        return self.k_means.get_labels()

    def get_medoids(self):
        return self.k_means.get_centers()

    def get_clusters(self):
        return self.k_means.partitions

    def plot_clusters(self):
        self.k_means.plot_clusters()