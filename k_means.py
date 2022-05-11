import numpy as np
import math
import matplotlib.pyplot as plt


class k_means:

    def __init__(self, data: np.ndarray, d: int, k: int, tol: float, max_iter: int, seed=20):
        """
        data: data to cluster
        d:dimension of the data
        k: prespecified number of clusters
        tol: convergence criterion
        max_iter: maximum number of iterations allowed
        """

        self.partitions = {i: [] for i in range(k)}
        self.centers = np.zeros((k, d))
        self.next_centers = np.zeros((k, d))
        self.labels = []
        self.d = d
        self.n = data.shape[0]
        self.counter = 0
        self.seed = seed
        self.max_iter = max_iter
        self.tol = tol
        self.k = k
        self.data = data
        self.cost = 0

    def initialize_centers(self, method: int):
        """
        method = 0:
        pick the first k points from the data as centers

        method = 1:
        randomly pick k points from the data as centers
        """
        if method == 0:
            self.centers = self.data[:self.k, :]

        elif method == 1:
            np.random.seed(20)
            self.centers = self.data[np.random.choice(len(self.data), self.k, replace=False)]

    def search(self):
        """
        update the patitions and then calculate the next centers;
        here we use centroids for k-means method
        """
        self.partitions = {i: [] for i in range(self.k)}
        self.next_centers = np.array([])

        distance = np.zeros((self.n, self.k))

        for k in range(self.k):
            # distance[:, k] = np.square(np.linalg.norm(self.data - self.centers[k, :], axis = 1))
            distance[:, k] = np.linalg.norm(self.data - self.centers[k, :], axis=1)

        classsifications = np.argmin(distance, axis=1)

        self.next_centers = np.zeros((self.k, self.d))

        for k in np.unique(classsifications):
            self.partitions[k] = np.where(classsifications == k)[0]

            self.next_centers[k, :] = np.mean(self.data[self.partitions[k]], axis=0)

    def is_updated(self):
        """
        return True if update is completed, namely, the algorithm has not yet converged;
        return False otherwise;
        The convergence criterion is the sum of absolute relative differences between
        self.centers and self.next_centers smaller than tol;
        """

        loss = np.absolute(np.sum((self.next_centers - self.centers) / self.centers * 100.0))
        if loss > self.tol:
            self.centers = self.next_centers
            return True
        else:
            False

    def fit_model(self):
        """
        function to fit the k-means algorithms using the above functions
        """
        self.initialize_centers(1)
        self.counter = 0

        for i in range(self.max_iter):
            self.search()

            if self.is_updated():

                if i == self.max_iter:
                    print("Maximum Number of Iteration Reached!")

            else:
                print("Convergence Reached! Number of Iterations: {}".format(i))
                break

        self.get_labels()

    def set_k(self, k):
        self.k = k

    def predict(self, pt):
        distances = [np.linalg.norm(pt - c) for c in self.centers]
        cluster_label = distances.index(min(distances))
        return cluster_label

    def get_labels(self):
        """
        :return: The label of each data point
        """

        self.labels = [0 for i in range(self.n)]

        for k in range(self.k):
            for i in self.partitions[k]:
                self.labels[int(i)] = k

        return self.labels

    def get_centers(self):
        return self.centers

    def get_clusters(self):
        return self.partitions

    def get_cost(self):
        """
        Here we use within cluster sum of squares as cost
        """

        cost = []
        for k in range(self.k):
            idxs = self.partitions[k]
            cost.append(np.sum(np.square(np.linalg.norm(self.data[idxs] - self.next_centers[k], axis=1))))
        self.cost = sum(cost)

        return self.cost

    def plot_clusters(self):
        if self.d > 2:
            print("Dimension too large!")
            return
        if self.labels == []:
            self.fit_model()
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, s=3)
        plt.scatter(np.array(self.centers)[:, 0], np.array(self.centers)[:, 1], marker='*', c=list(range(self.k)),
                    s=300)