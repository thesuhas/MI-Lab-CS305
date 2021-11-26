import numpy as np
import sys

class KMeansClustering:
    """
    K-Means Clustering Model

    Args:
        n_clusters: Number of clusters(int)
    """

    def __init__(self, n_clusters, n_init=10, max_iter=1000, delta=0.001):

        self.n_cluster = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.delta = delta

    def init_centroids(self, data):
        idx = np.random.choice(
            data.shape[0], size=self.n_cluster, replace=False)
        self.centroids = np.copy(data[idx, :])

    def fit(self, data):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix(M data points with D attributes each)(numpy float)
        Returns:
            The object itself
        """
        if data.shape[0] < self.n_cluster:
            raise ValueError(
                'Number of clusters is grater than number of datapoints')

        best_centroids = None
        m_score = float('inf')

        for _ in range(self.n_init):
            self.init_centroids(data)

            for _ in range(self.max_iter):
                cluster_assign = self.e_step(data)
                old_centroid = np.copy(self.centroids)
                self.m_step(data, cluster_assign)

                if np.abs(old_centroid - self.centroids).sum() < self.delta:
                    break

            cur_score = self.evaluate(data)

            if cur_score < m_score:
                m_score = cur_score
                best_centroids = np.copy(self.centroids)

        self.centroids = best_centroids

        return self

    def e_step(self, data):
        """
        Expectation Step.
        Finding the cluster assignments of all the points in the data passed
        based on the current centroids
        Args:
            data: M x D Matrix (M training samples with D attributes each)(numpy float)
        Returns:
            Cluster assignment of all the samples in the training data
            (M) Vector (M number of samples in the train dataset)(numpy int)
        """
        cluster = np.zeros(data.shape[0])
        centroid = -1
        for i in range(data.shape[0]):
            min = sys.maxsize
            for k in range(self.n_cluster):
                diff = np.subtract(data[i], self.centroids[k])
                square_diff = np.square(diff)
                sum_square_diff = np.sum(square_diff)
                dist = sum_square_diff ** 0.5
                if dist < min:
                    min = dist
                    centroid = k
            cluster[i] = centroid
        return cluster

    def m_step(self, data, cluster_assgn):
        """
        Maximization Step.
        Compute the centroids
        Args:
            data: M x D Matrix(M training samples with D attributes each)(numpy float)
        Change self.centroids
        """
        assign =  {new_list: [] for new_list in range(self.n_cluster)}
        for i in range(data.shape[0]):
            assign[cluster_assgn[i]].append(i)
        k = 0
        for samples in assign.values():
            x = np.zeros(data.shape[1])
            for sample in samples:
                x = np.add(data[sample], x)
            x = np.divide(x, len(samples))
            self.centroids[k] = x
            k += 1
        

    def evaluate(self, data, cluster_assign):
        """
        K-Means Objective
        Args:
            data: Test data (M x D) matrix (numpy float)
        Returns:
            metric : (float.)
        """
        error = 0
        for i in range(data.shape[0]):
            k = int(cluster_assign[i])
            for j in range(data.shape[1]):
                error += (data[i][j] - self.centroids[k][j]) ** 2
        return error