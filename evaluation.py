"""
In this module an algorithm for evaluation and comparison of different clustering algorithms is implemented.
"""

import numpy as np

from sklearn.datasets import make_moons

import matplotlib.pyplot as plt


class ClusterEvaluation:
    """Evaluation of cluster algorithms.

    Runs different algorithms on same data and compares their results in form of text and plots.
    """

    def __init__(self, algorithms, names, metric):
        """Intializes cluster evaluation.

        Args:
            algorithms (list of ClusterMixin): clustering algorithms
            names (list of str): names of the algorithms
            metric (callable): metric for evaluation
        """
        self.algorithms = algorithms
        self.names = names
        self.metric = metric
        self.points = None
        self.labels = None

    def create_data(self, cluster_size, scale):
        """creates data on which algorithms should be evaluated

        Args:
            cluster_size (int): number of samples per cluster
            scale (float): noise parameter
        """
        # create moon data
        X_1, _ = make_moons(n_samples=cluster_size, noise=scale, random_state=666)
        X_2, _ = make_moons(n_samples=cluster_size, noise=scale, random_state=666)
        X_3, _ = make_moons(n_samples=cluster_size, noise=scale, random_state=999)
        X_4, _ = make_moons(n_samples=cluster_size, noise=scale, random_state=999)

        # shift in y-direction
        X_3[:, 1] += 2
        X_4[:, 1] += 2

        # assign true labels to data points
        labeled_points = np.zeros((4 * cluster_size, 3))
        labeled_points[:, :2] = np.vstack((X_1, X_2, X_3, X_4))
        for i in range(4):
            labeled_points[i * cluster_size: (i + 1) * cluster_size, 2] = i

        np.random.shuffle(labeled_points)

        self.points = labeled_points[:, :2]
        self.labels = labeled_points[:, 2]

    def evaluate(self):
        """Performs the evaluation of the algorithms and outputs results as text and as plot.
        """
        n = len(self.algorithms)
        plt.figure(figsize=(16, 8))
        for i, (algo, name) in enumerate(zip(self.algorithms, self.names)):
            print('Performing clustering....')
            predicted = algo.fit_predict(self.points)
            score = self.metric(self.labels, predicted)

            score_text = f'Algorithm: {name}\nnoise: {scale}\nscore: {score:.4f}'

            plt.subplot(1, n, i + 1)
            plt.title(score_text)
            plt.scatter(self.points[:, 0], self.points[:, 1], c=predicted)

            print(score_text)

            print('-' * 16)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import mutual_info_score

    from dbscan import GeneralizedDensityBasedScan

    kmeans = KMeans(n_clusters=4, algorithm='full', n_init=1, random_state=0)
    dbscan_euclidean = GeneralizedDensityBasedScan(radius=0.2, min_density=5)
    dbscan_manhattan = GeneralizedDensityBasedScan(radius=0.2, min_density=5, nb_metric='manhattan')
    dbscan_sklearn = DBSCAN(eps=0.2, min_samples=5)
    agglo = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')

    algorithms = [kmeans, dbscan_euclidean, dbscan_manhattan, dbscan_sklearn, agglo]
    names = ['kmeans', 'dbscan euclidean', 'dbscan manhattan', 'sklearn.dbscan', 'agglo']

    ce = ClusterEvaluation(
        algorithms,
        names,
        mutual_info_score
    )

    for scale in [0.1, 0.05]:
        print(f'Running 4 samples with scale: {scale}')
        print('Wait a moment until the plots show up....')

        ce.create_data(cluster_size=199, scale=scale)

        for _ in range(4):
            ce.evaluate()
