import numpy as np
import sys


def solve(c: np.array, b: np.array, a: np.matrix, eps: np.double, alpha: np.double, starting_point: np.array):
    first_solution = iteration(c=c, b=b, a=a, alpha=alpha, starting_point=starting_point)
    second_solution = iteration(c=c, b=b, a=a, alpha=alpha, starting_point=first_solution)
    while np.all(np.linalg.norm(first_solution - second_solution, ord=2) >= eps):
        second_solution = first_solution
        first_solution = iteration(c=c, b=b, a=a, alpha=alpha, starting_point=second_solution)
    return np.round(second_solution, -int(np.floor(np.log10(eps))))


def iteration(c: np.array, b: np.array, a: np.matrix, alpha: np.double, starting_point: np.array) -> np.array:
    row_starting_point = np.asarray(starting_point.reshape(1, -1))[0]

    diag = np.diag(row_starting_point)
    a_tilda = np.dot(a, diag)
    c_tilda = np.dot(diag, c)

    i = np.identity(len(row_starting_point))

    mul1 = np.dot(a_tilda, np.transpose(a_tilda))
    if np.linalg.det(mul1) == 0:
        print("The problem does not have solution")
        sys.exit(0)
    inv = np.linalg.inv(mul1)
    a_tilda_t = np.transpose(a_tilda)

    mul2 = np.dot(a_tilda_t, inv)
    mul3 = np.dot(mul2, a_tilda)
    p = np.subtract(i, mul3)

    c_p = np.dot(p, c_tilda)
    v = np.abs(np.min(c_p))

    ones = np.ones(len(c_p))
    mul4 = np.divide(alpha, v)
    mul5 = c_p * mul4
    x_tilda = mul5 + ones.reshape(-1, 1)

    if np.allclose(np.dot(a, x_tilda), b):
        print("The method is not applicable")
        sys.exit(0)

    return np.matmul(diag, x_tilda)


c = np.array([1, 1, 0, 0], np.double).reshape(-1, 1)
a = np.matrix([[2, 4, 1, 0], [1, 3, 0, -1]], np.double)
b = np.array([16, 9], np.double).reshape(-1, 1)
alpha = np.double(0.5)
eps = np.double(0.01)
starting_point = np.array([0.5, 3.5, 1, 2], np.double).reshape(-1, 1)

solution = solve(c=c, b=b, a=a, eps=eps, alpha=alpha, starting_point=starting_point)
print(solution)
