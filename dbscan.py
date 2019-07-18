"""
In this module a custom generalized DBSCAN algorithm is implemented.
"""
import numpy as np

from scipy.spatial.distance import euclidean, cityblock

from sklearn.base import ClusterMixin


class ClusterPoint:
    """Represents a cluster point."""

    def __init__(self, point, point_id):
        """Initializes the cluster point

        Args:
            point ((2,)-ndarray): 2d coordinates of the point
            point_id (int): a unique id of the point
        """
        self.coords = point
        self.point_index = point_id
        self.visited = False
        self.cluster = None
        self.is_noise = False


class NeighborHoodCalculator:
    """Calculates a neighborhood of a given point w.r.t. a specified norm.

    The neighborhood calculating method get_neighbors is assigned during the initialization based on given norm.
    """

    def __init__(self, cluster_points, neighborhood_type='euclidean'):
        """Initializes neighborhood calculator.

        Args:
            cluster_points (list of ClusterPoint): cluster points
            neighborhood_type (str, 'euclidean' or 'manhattan'): type of distance to use for neighborhood calculation
        """
        if neighborhood_type not in ('euclidean', 'manhattan'):
            raise ValueError('neighborhood_type should be euclidean or manhattan')
        self.cluster_points = cluster_points

        if neighborhood_type == 'euclidean':
            self.get_neighbors = self._get_neighbors_euclidean
        if neighborhood_type == 'manhattan':
            self.get_neighbors = self._get_neighbors_manhattan

    def _get_neighbors_euclidean(self, point, radius):
        """Returns cluster points whose coordinates are within radius to the given point according to Euclidean distance

        Args:
            point ((2,)-ndarray): the point
            radius (positive float): search radius

        Returns:
            list of ClusterPoint
        """
        return self._filter_by_distance(point, radius, euclidean)

    def _get_neighbors_manhattan(self, point, radius):
        """Returns cluster points within radius to the given point according to Manhattan distance

        Args:
            point ((2,)-ndarray): the point
            radius (positive float): search radius

        Returns:
            list of ClusterPoint
        """
        return self._filter_by_distance(point, radius, cityblock)

    def _filter_by_distance(self, point, radius, distance):
        """Returns cluster points within radius to the given point where distance is calculated according to passed norm

        Args:
            point ((2,)-ndarray): the point
            radius (positive float): search radius
            distance (callable): norm to use for distance

        Returns:
            list of ClusterPoint
        """
        return [cluster_point for cluster_point in self.cluster_points if
                distance(point.coords, cluster_point.coords) <= radius]


class DensityCheck:
    """Represents an algorithm to check density of some point set."""

    def __init__(self, min_density, density_type='number'):
        """Initializes density check.

        Args:
            min_density (int): minimal number of points in the neighborhood for it to be dense
            density_type (str): density check strategy, only 'number' supported by now
        """
        if density_type != 'number':
            raise ValueError('density type should be number')
        self.min_density = min_density
        self.density_type = density_type

        if density_type == 'number':
            self.check = self._check_number

    def _check_number(self, cluster_points):
        """Checks if cluster_points is dense.

        Args:
            cluster_points (list of ClusterPoint): neighborhood

        Returns:
            bool: True if the neighborhood is dense, False otherwise
        """
        return len(cluster_points) >= self.min_density


class Cluster:
    """Represents a cluster.

    count: tracks the number of generated clusters
    """
    count = 0

    def __init__(self):
        """Initializes the cluster.
        """
        self.cluster_id = Cluster.count
        self._points = []
        Cluster.count += 1

    def insert(self, point):
        """Inserts a new point to the cluster.

        Args:
            point ((2,)-ndarray): new point
        """
        self._points.append(point)
        point.cluster = self


class GeneralizedDensityBasedScan(ClusterMixin):
    """Represents the GDBSCAN algorithm.
    """

    def __init__(self, radius, min_density, nb_metric='euclidean', density_type='number'):
        """Initializes a GDBSCAN.

        Args:
            radius (positive float): search radius
            min_density (int): minimal density for density criterion
            nb_metric (str): metric to use for distance calculations
            density_type (str): the type of density criterion
        """
        self.radius = radius
        self.nb_metric = nb_metric
        self.density_type = density_type
        self.min_density = min_density
        self.cluster_points = None
        self.labels_ = None
        self.clusters = None
        self.nb_calculator = None
        self.density_check = None

    def fit(self, points):
        """Fits the algorithm to the points.

        Args:
            points ((n_samples, 2)-ndarray): the points
        """
        Cluster.count = 0

        self.cluster_points = [
            ClusterPoint(point, i) for i, point in enumerate(points)
        ]
        self.nb_calculator = NeighborHoodCalculator(self.cluster_points, self.nb_metric)
        self.density_check = DensityCheck(self.min_density, self.density_type)

        self.labels_ = -1 * np.ones(points.shape[0])

        for point in self.cluster_points:
            if not point.visited:
                point.visited = True
                neighbors = self.nb_calculator.get_neighbors(point, self.radius)

                if self.density_check.check(neighbors):
                    cluster = Cluster()
                    self.expand_cluster(point, neighbors, cluster)
                else:
                    point.is_noise = True

    def expand_cluster(self, point, neighbors, cluster):
        """Expands the cluster.

        Args:
            point ((2,)-ndarray):
            neighbors (list of LusterPoint):
            cluster (Cluster):
        """
        cluster.insert(point)
        self.labels_[point.point_index] = cluster.cluster_id

        for p in neighbors:
            if not p.visited:
                p.visited = True
                neighbors_ = self.nb_calculator.get_neighbors(p, self.radius)
                if self.density_check.check(neighbors_):
                    neighbors.extend(neighbors_)
            if not p.cluster:
                cluster.insert(p)
                self.labels_[p.point_index] = cluster.cluster_id
