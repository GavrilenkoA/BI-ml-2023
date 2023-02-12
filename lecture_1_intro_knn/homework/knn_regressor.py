import numpy as np


class KNNRegress:
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X):
        distances = self.compute_distances_one_loops(X)
        return self.predict_regressor(distances)

    def compute_distances_one_loops(self, X):
        num_test = len(X)
        num_train = len(self.train_X)
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i] = np.sum(np.abs(X[i] - self.train_X), axis=1)
        return dists

    def predict_regressor(self, distances):
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)
        k = self.k
        for n in range(n_test):
            dist = distances[n]  # раcтояния от текущего теста до всех объектов из трейна
            idx = np.argsort(dist)[:k]  # индексы топ k объектов из трейна до которых расстояние наименьшее
            candidates = self.train_y[idx]
            prediction[n] = candidates.mean()  # среднее от ближайших
        return prediction
