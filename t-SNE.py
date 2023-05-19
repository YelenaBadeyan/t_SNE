import numpy as np


class t_SNE:
    def __init__(self, perplexity, n_dim, number_of_iter, learning_rate, momentum):
        self.perplexity = perplexity
        self.n_dim = n_dim
        self.number_of_iter = number_of_iter
        self.learning_rate = learning_rate
        self.momentum = momentum

    def sigma_searh(self, D, i, perplexity):
        differance = 0
        norm = np.linalg.norm(D, axis=1)
        std = np.std(norm)
        sigma = np.inf

        for i in np.linspace(0.001 * std, 10 * std, 100):
            p = np.exp(-norm ** 2 / (2 * i ** 2))
            p = p / np.sum(p)
            p = np.maximum(p, 10 ** (-7))

            H = -np.sum(p * np.log2(p))

            if np.abs(np.log(perplexity) - H * np.log(2)) < np.abs(differance):
                differance = np.log(perplexity) - H * np.log(2)
                sigma = i
        return sigma

    def pairwise_affinities(self, X):
        n = X.shape[0]
        p = np.zeros(shape=(n, n))
        for i in range(n):
            D = X[i] - X
            sigma = self.sigma_searh(D, i, self.perplexity)
            N = np.linalg.norm(D, axis=1)
            p[i, :] = np.exp(-N ** 2 / (2 * sigma ** 2))
            np.fill_diagonal(p, 0)
            p[i, :] = p[i, :] / np.sum(p[i, :])
            p = np.maximum(p, 10 ** (-7))
        return p

    def symmetric_p(self, p):
        n = len(p)
        p_sym = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(n):
                p_sym[i, j] = (p[i, j] + p[j, i]) / (2 * n)
        p_sym = np.maximum(p_sym, 10 ** (-7))

        return p_sym

    def generate_y(self, X):
        return np.random.normal(size=(len(X), self.n_dim))

    def low_dimensional_affinities(self, Y):
        q = np.zeros(shape=(len(Y), len(Y)))
        for i in range(len(Y)):
            D = Y[i] - Y
            N = np.linalg.norm(D, axis=1)
            q[i, :] = (1 + N ** 2) ** (-1)
        np.fill_diagonal(q, 0)
        q = q / np.sum(q)
        q = np.maximum(q, 10 ** (-7))
        return q

    def gradient(self, p, q, Y):
        n = len(p)
        gradient = np.zeros(shape=(n, self.n_dim))
        for i in range(n):
            diff = Y[i] - Y
            a = np.array([(p[i, :] - q[i, :])])
            b = np.array([(1 + np.linalg.norm(diff, axis=1)) ** (-1)])
            c = diff
            gradient[i] = 4 * np.sum((a * b).T * c, axis=0)
        return gradient

    def fit_transform(self, X):
        n = len(X)
        p = self.pairwise_affinities(X)
        p_sym = self.symmetric_p(p)
        Y = self.generate_y(X)
        q = self.low_dimensional_affinities(Y)
        Y_final = np.zeros(shape=(len(X), self.n_dim))
        for i in range(self.number_of_iter):
            print(Y[i])
            gradient = self.gradient(p_sym, q, Y)
            print(gradient)
            Y_final[i] = Y[i] + self.learning_rate * gradient + self.momentum * (np.array(Y[i]) - np.array(Y[i - 1]))
        return Y_final
