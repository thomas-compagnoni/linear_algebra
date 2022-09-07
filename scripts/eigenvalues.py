from decimal import *
import numpy as np


class Eig:
    """ Computes eigenvalues using QR factorization, where Q is the Gram-Schimdt decomposition of A """

    def __init__(self, precision=64):
        self.prec = precision
        self.threshold = float("1e-" + str(int(.9*self.prec)))

    def rref(self, A, I):

        R = np.concatenate([A.copy(), I], axis=1)
        m = A.shape[0]
        n = A.shape[1]

        j = 0
        pivot_cols = []

        for i in range(m):

            while j < n:
                if abs(R[i, j]) <= self.threshold:
                    idx = min(np.where(abs(R[i:, j]) > self.threshold)[0] + i, default=0)
                    if not idx:
                        j += 1
                    else:
                        R[[i, idx]] = R[[idx, i]]
                        break
                else:
                    break

            if np.all(abs(R[i:]) < self.threshold) or j >= n:
                break

            for k in range(i + 1, m):
                if abs(R[k, j]) >= self.threshold:
                    factor = R[k, j] / R[i, j]
                    R[k] = R[k] - R[i] * factor
            pivot_cols.append(j)
            j += 1

        for i, j in enumerate(pivot_cols):
            for k in range(i):
                if abs(R[k, j]) >= self.threshold:
                    factor = R[k, j] / R[i, j]
                    R[k] = R[k] - R[i] * factor

            pivot = R[i, j]
            if pivot != 1:
                R[i] /= pivot

        return R

    def inverse(self, A):
        I = np.vectorize(Decimal)(np.eye(A.shape[0]).astype(np.float64))
        IA = self.rref(A, I)
        return IA[:, A.shape[0]:]

    def projection_matrix(self, A):
        P = A @ self.inverse(A.T @ A) @ A.T
        return P

    def project_vector(self, A, b):
        P = self.projection_matrix(A)
        return np.dot(P, b)

    def gram_schmidt(self, A):
        Q = A.copy()

        for n in range(A.shape[1]):
            projection = Q[:, n]
            for j in range(n):
                projection -= self.project_vector(Q[:, [j]], Q[:, n])
            Q[:, n] = projection

        for n in range(A.shape[1]):
            if np.sqrt(np.inner(Q[:, n], Q[:, n])) == 0:
                continue
            else:
                Q[:, n] /= np.sqrt(np.inner(Q[:, n], Q[:, n]))

        return Q

    def eigenvalues(self, A):

        with localcontext() as ctx:
            ctx.prec = self.prec
            A = np.vectorize(Decimal)(A.astype(np.float64))

            for i in range(10000):
                Q = self.gram_schmidt(A)
                A = Q.T @ A @ Q
                if all(abs(A[np.tril_indices(A.shape[0], -1)]) <= self.threshold):
                    break

            if i == 9999:
                raise ValueError('The matrix is non-diagonalizable')

        return np.diagonal(A).astype(np.float64)


def eigenvalues(A):
    """ Computes eigenvalues using QR factorization, where Q is the Gram-Schimdt decomposition of A

        ! The algorithm does not work for repeated and complex eigenvalues  """

    return Eig().eigenvalues(A)
