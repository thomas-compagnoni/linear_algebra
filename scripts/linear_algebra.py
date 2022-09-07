import numpy as np
from numpy.linalg import eigvals

from tqdm import tqdm

def PLU(A):
    """ PLU Decomposition:
        P is the permutation matrix, L is the lower triangular, U is the upper triangular
        ! The function works only for square matrices, and it will raise an error if the matrix is singular."""

    if not A.shape[0] == A.shape[1]:
        raise ValueError('The matrix is not square')

    P = np.eye(A.shape[0])
    L = np.eye(A.shape[0])
    U = A.astype('float64')

    m = A.shape[0]
    for i in range(m):

        # exchange rows if necessary
        if abs(U[i, i]) < 1e-12:
            for k in range(i, m):
                if abs(U[k, i]) > 1e-12:
                    U[[i, k]] = U[[k, i]]
                    P[[i, k]] = P[[k, i]]
                    break
                elif k == m - 1:
                    raise ValueError('The matrix is singular')
        # elimination
        for k in range(i + 1, m):
            if abs(U[k, i]) >= 1e-12:
                factor = U[k, i] / U[i, i]
                U[k] = U[k] - factor * U[i]
                L[k, i] = factor

    return P, L, U


def rref(A, augmented=None, return_pivots=False):
    """ Reduced Row Echelon Form of a matrix, the simplest way to represent a matrix.
        This works for every matrix (also not square and singular).

        Some notes:
        1. The pivot columns are the independent columns of the matrix, which can be used also as a basis for C(A). Remember C(A) /= C(R).
        2. Non-zero rows forms the rows space of A, C(A').
        3. The parameter augmented can be used to perform the algorithm on an Augmented Matrix, \
           for example to find the inverse of A or the solutions to Ax=b.
        """

    R = A.astype('float64')
    m = A.shape[0]
    n = A.shape[1]

    if isinstance(augmented, np.ndarray):
        R = np.concatenate([R, augmented], axis=1)

    j = 0
    pivot_cols = []

    for i in range(m):

        while j < n:
            if abs(R[i, j]) <= 1e-12:
                idx = min(np.where(abs(R[i:, j]) > 1e-12)[0] + i, default=0)
                if not idx:
                    j += 1
                else:
                    R[[i, idx]] = R[[idx, i]]
                    break
            else:
                break

        if np.allclose(R[i:], 0) or j >= n:
            break

        for k in range(i + 1, m):
            if abs(R[k, j]) >= 1e-12:
                factor = R[k, j] / R[i, j]
                R[k] = R[k] - R[i] * factor
        pivot_cols.append(j)
        j += 1

    for i, j in enumerate(pivot_cols):
        # put all zeros in pivot column
        for k in range(i):
            if abs(R[k, j]) >= 1e-12:
                factor = R[k, j] / R[i, j]
                R[k] = R[k] - R[i] * factor

        # normalize pivot to 0
        pivot = R[i, j]
        if pivot != 1:
            R[i] /= pivot

    R[np.isclose(R, 1, atol=1e-09)] = 1.
    R[np.isclose(R, 0, atol=1e-09)] = 0.

    if return_pivots:
        return R, pivot_cols
    else:
        return R


def inverse(A):
    """ The function perform the Gauss-Jordan reduction for a matrix A, so it finds its inverse.
        Create the augmented matrix AI, which through elimination becomes I(A^-1).
        ! The function works only for square matrices, and it will raise an error if the matrix is singular."""

    if not A.shape[0] == A.shape[1]:
        raise ValueError('The matrix is not square')

    IA = rref(A, np.eye(A.shape[0]))

    if np.all(IA[:, :A.shape[0]] == np.eye(A.shape[0])):
        return IA[:, A.shape[0]:]
    else:
        raise ValueError('The matrix is not invertible')


def column_space(A):
    """ C(A) is formed by the vectors of A in the pivot columns which result from RREF """

    if isinstance(A, np.ndarray):
        R, pivots = rref(A, return_pivots=True)
    else:
        raise ValueError("Please insert A as np.ndarray")
    # pivots = np.where((np.isclose(R.sum(axis=0), 1)) & (np.count_nonzero(R, axis=0) == 1) == 1)[0].tolist()
    return A[:, pivots]


def nullspace(A):
    """ N(A) == [-F I]'
        F is composed by the free columns resulting from RREF
        The dimension of N(A) is n - rank(A). So, if there are no free variables and rank(A) = n, N(A) = {zero_vector}.  """

    # if isinstance(A, np.ndarray):
    R, pivots = rref(A, return_pivots=True)
    # pivots = np.where((np.isclose(R.sum(axis=0), 1)) & (np.count_nonzero(R, axis=0) == 1) == 1)[0].tolist()
    free = [f for f in range(R.shape[1]) if f not in pivots]

    if len(free) == 0:
        return np.zeros((A.shape[1], 1))
    else:
        N = np.zeros((A.shape[1], A.shape[1] - len(pivots)))
        R_non_zero = R[np.count_nonzero(R, axis=1) > 0, :]
        F = -R_non_zero[:, free]
        N[pivots, :] = F
        N[free, :] = np.eye(len(free))
        N[np.isclose(N, 0)] = 0
        return N


def row_space(A):
    """ There are two different methods for finding the row space:
        1. Row_space of A = C(A')
        2. Row_space of A = non-zero rows of RREF
        The function executes the second methods.
        Remember: the dimension of C(A) == rank(A) == C(A') """

    # if isinstance(A, np.ndarray):
    R, pivots = rref(A, return_pivots=True)
    # pivots = np.where((np.isclose(R.sum(axis=0), 1)) & (np.count_nonzero(R, axis=0) == 1) == 1)[0].tolist()
    if len(pivots) == R.shape[0]:
        return np.zeros((1, R.shape[0]))
    else:
        R_non_zero = R[np.count_nonzero(R, axis=1) > 0, :]
        return R_non_zero.T


def left_nullspace(A):
    """ The dimension of N(A') is m - rank(A).
        The function computes A = E*R
        The zero rows of R represent the nullspace in E """

    R = rref(A, augmented=np.eye(A.shape[0]))
    zero_rows = np.count_nonzero(R[:, :A.shape[1]], axis=1) == 0
    N = R[zero_rows, A.shape[1]:]
    return N.T


def solve(A, b):
    b = b.reshape(-1, 1)
    R, pivots = rref(A, augmented=b, return_pivots=True)
    # pivots = np.where((np.isclose(R.sum(axis=0), 1)) & (np.count_nonzero(R, axis=0) == 1) == 1)[0].tolist()
    R_non_zero = R[np.count_nonzero(~np.isclose(R, 0), axis=1) != 0]
    if R_non_zero.shape[0] == len(pivots):
        particular_solution = np.zeros((A.shape[1], 1))
        particular_solution[pivots] = R_non_zero[:, [-1]]
        return particular_solution
    else:
        raise ValueError("There are no solutions, b is not dependent on A")


def orthogonality(A, B):
    """ Condition: A.T * B = 0 """
    if np.allclose(np.dot(A.T, B), 0):
        return True
    else:
        return False


def projection_matrix(A):
    P = A @ inverse(A.T @ A) @ A.T
    return P


def project_vector(A, b):
    P = projection_matrix(A)
    return np.dot(P, b)


def gram_schmidt(A):
    """ Transform columns of a matrix A into orthonormal vectors
        - Orthogonality: vectors are orthogonal between each others
        - Normality: length of vector == 1
        (The column space remains invariant) """

    Q = np.asarray(A, dtype='float64')

    # ORTHOGONALITY
    for n in range(A.shape[1]):
        projection = Q[:, n]
        for j in range(n):
            projection -= project_vector(Q[:, [j]], Q[:, n])
        Q[:, n] = projection

    # NORMALITY
    for n in range(A.shape[1]):
        Q[:, n] /= np.sqrt(np.inner(Q[:, n], Q[:, n]))

    return Q


def determinant(A):
    """ det(A) = product of the diagonal of the reduced form U
        Property 2 of determinant: exchange rows -> reverse the sign of the determinant
            * if even exchanges: sign is the same
            * if odd exchanges: sign is the inverse
        This algorithm is much faster than the one with cofactor """

    P, L, U = PLU(A)
    det = np.prod(U.diagonal())

    count_row_exchanges = 0
    for j in range(P.shape[1]):
        if P[j, j] == 1:
            continue
        else:
            for i in range(j + 1, P.shape[1]):
                if P[i, j] == 1:
                    P[[j, i]] = P[[i, j]]
                    count_row_exchanges += 1
    if count_row_exchanges % 2 != 0:
        det = -det

    return det


def determinant_cofactor(A, progress_bar=True):
    """ The function computes the determinant using the Cofactor formula.
        det(A) = SUM of (± Aij * Cij) with j=0, 1, ..., n - 1

        Default: i = 0

        ! The function works only for square matrices"""

    if not A.shape[0] == A.shape[1]:
        raise ValueError('The matrix is not square')
    elif A.shape[0] == 2:
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        return det
    else:
        det = 0.
        A_crop = np.delete(A, 0, axis=0)
        sign = 1.

        bar = tqdm(total=A.shape[1]) if progress_bar else None

        for j in range(A.shape[1]):
            a = A[0, j]
            cofactor = determinant_cofactor(A=np.delete(A_crop, j, axis=1), progress_bar=False)
            det += sign * a * cofactor
            sign = -sign
            if progress_bar:
                bar.update()

        return det


def inverse_cofactor(A):
    detA = determinant_cofactor(A, False)
    if not np.isclose(detA, 0):
        C = np.zeros(A.shape)
        for i in range(C.shape[0]):
            A_crop = np.delete(A, i, axis=0)
            for j in range(C.shape[1]):
                sign = 1 if (i + j) % 2 == 0 else -1
                Cij = sign * determinant_cofactor(np.delete(A_crop, j, axis=1), False)
                C[i, j] = Cij
        return (1 / detA) * C.T
    else:
        raise ValueError('The matrix is singular')


def solve_cramer(A, b):
    A = A.astype('float64')
    x = np.zeros(b.shape)
    detA = determinant_cofactor(A, False)
    for j in range(A.shape[1]):
        Bj = A.copy()
        Bj[:, [j]] = b
        detBj = determinant_cofactor(Bj, False)
        x[j] = detBj / detA
    return x


def eigenvalues(A):
    return eigvals(A)


def eigenvectors(A, evalues):
    if not A.shape[0] == A.shape[1]:
        raise ValueError('The matrix is not square')

    x = np.zeros(A.shape)

    for i, labda in enumerate(evalues):
        x[:, [i]] = nullspace(A - labda * np.eye(A.shape[0]))

    return x


def matrix_power(A, power=1):
    """ Power of a matrix A using eigenvalues if A is diagonalizable
        return A^n """

    evalues = eigenvalues(A)
    evectors = eigenvectors(A, evalues)

    LAMBDA = np.eye(A.shape[0])
    LAMBDA[np.diag_indices(A.shape[0])] = np.power(evalues, power)

    return evectors @ LAMBDA @ inverse(evectors)


def matrix_exponential(A, t=1):
    """ Exponential of a matrix A using eigenvalues if A is diagonalizable
        return e^A """

    evalues = eigenvalues(A)
    evectors = eigenvectors(A, evalues)

    LAMBDA = np.eye(A.shape[0])
    LAMBDA[np.diag_indices(A.shape[0])] = np.exp(np.multiply(evalues, t))

    return evectors @ LAMBDA @ inverse(evectors)
