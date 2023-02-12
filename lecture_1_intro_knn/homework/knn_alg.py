import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """

    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, n_loops=1):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """

        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)

        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)

    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        # X - test выборка
        distances = []
        for i in X:
            dist_per_train = []
            for j in self.train_X:
                cur_dist = np.sum(np.abs(i - j))
                dist_per_train.append(cur_dist)
            distances.append(dist_per_train)
        distances = np.array(distances)
        return distances

    def compute_distances_one_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        num_test = len(X)
        num_train = len(self.train_X)
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i] = np.sum(np.abs(X[i] - self.train_X), axis=1)
        return dists

    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        dists = np.abs(X[:, None] - self.train_X).sum(-1)
        return dists

    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        prediction, np array of bool (num_test_samples) - binary predictions
           for every test sample
        """

        #n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)
        k = self.k
        for n in range(n_test):
            dist = distances[n]     # растояния от текущего теста до всех объектов из трейна
            idx = np.argsort(dist)[:k]  # индексы топ k объектов из трейна до которых расстояние наименьшее
            candidates = self.train_y[idx]
            values, counts = np.unique(candidates, return_counts=True)  # выбираем самый распространенный
            prediction[n] = values[counts.argmax()]
        return prediction


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        #n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)
        k = self.k
        for n in range(n_test):
            dist = distances[n]     # растояния от текущего теста до всех объектов из трейна
            idx = np.argsort(dist)[:k]  # индексы топ k объектов из трейна до которых расстояние наименьшее
            candidates = self.train_y[idx]
            values, counts = np.unique(candidates, return_counts=True)  # выбираем самый распространенный
            prediction[n] = values[counts.argmax()]
        return prediction

# %%
