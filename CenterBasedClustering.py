import numpy as np
import threading
import random

from NetworkSimplex import NetworkSimplex
from scipy.spatial.distance import cdist

class CenterBasedClustering:
    def __init__(self, X, c_true, init_method, seed, initial_centers=None):
        self.num = len(X)
        self.dim = len(X[0])
        self.c_true = c_true

        self.seed = seed
        self.init_method = init_method
        random.seed(seed)
        self.uniform_rand = lambda: random.randint(0, self.c_true - 1)

        self.X = np.array(X)

        self.Cen = np.zeros((self.c_true, self.dim)) if initial_centers is None else np.array(initial_centers)

        self.y = np.zeros(self.num, dtype=int)

        self.n = np.zeros(self.c_true, dtype=int)

    def Update_n(self):
        self.n = np.bincount(self.y, minlength=self.c_true)

    def Update_Cen(self):
        self.n = np.zeros(self.c_true, dtype=int)
        self.Cen = np.zeros((self.c_true, self.dim))

        for i in range(self.num):
            tmp_c = self.y[i]
            self.n[tmp_c] += 1
            self.Cen[tmp_c] += self.X[i]

        for k in range(self.c_true):
            if self.n[k] > 0:
                self.Cen[k] /= self.n[k]

    def GetSumSquaredError(self):
        dist_matrix = cdist(self.X, self.Cen, 'cosine')

        dist = np.zeros(self.num)

        for i in range(self.num):
            tmp_c = self.y[i]
            dist[i] = dist_matrix[i, tmp_c]

    def initial_y(self):
        if self.init_method == "random_y":
            self.InitWithRandomAssignment()
        else:
            self.initial_y_from_centers()

    def InitWithRandomAssignment(self):
        self.y = np.array([self.uniform_rand() for _ in range(self.num)])

    def initial_y_from_centers(self):
        # Calculate the cosine distance between each data point and each cluster center
        distances = cdist(self.X, self.Cen, 'cosine')

        # Assign each data point to the cluster center with the smallest distance
        self.y = np.argmin(distances, axis=1)


class RegularizedKMeans(CenterBasedClustering):
    def __init__(self, X, c_true, init_method, warm_start, n_jobs, seed, initial_centers=None):
        super().__init__(X, c_true, init_method, seed, initial_centers)
        self.warm_start = warm_start
        self.n_jobs = n_jobs if n_jobs != -1 else threading.cpu_count()
        self.costs = np.zeros((self.num, self.c_true))
        self.obj = []
        self.Y = []

    def opt(self, rep, type):
        self.Y = np.zeros((rep, self.num), dtype=int)
        self.obj = np.zeros(rep)

        for rep_i in range(rep):
            if type == "Hard":
                self.obj[rep_i] = self.SolveHard()
            elif type == "Soft":
                self.obj[rep_i] = self.Solve(0)
            self.Y[rep_i] = self.y.copy()

    def SolveHard(self):
        return self._SolveHard(self.num // self.c_true, (self.num + self.c_true - 1) // self.c_true)

    def _SolveHard(self, lower_bound, upper_bound):
        self.initial_y()
        self.UpdateCostMatrix()
        ns = NetworkSimplex()
        ns.build_hard(self.costs, self.c_true, lower_bound, upper_bound)
        return self._Solve(ns)

    def Solve(self, f_th):
        self.initial_y()
        self.UpdateCostMatrix()
        ns = NetworkSimplex()
        ns.build(self.costs, f_th)
        return self._Solve(ns)

    def _Solve(self, builder):
        old_assignments = np.zeros_like(self.y)
        ns_solver = builder
        ns_solver.simplex()
        self.y = ns_solver.get_assignments()

        while not np.array_equal(old_assignments, self.y):
            old_assignments = self.y.copy()
            self.Update_Cen()
            self.UpdateCostMatrix()
            if self.warm_start:
                ns_solver.update_costs(self.costs)
            else:
                ns_solver = builder
            ns_solver.simplex()
            self.y = ns_solver.get_assignments()

        return self.GetSumSquaredError()

    def UpdateCostMatrix(self):
        self.costs = cdist(self.X, self.Cen, 'cosine')
