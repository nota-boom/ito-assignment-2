import numpy as np
from typing import List
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


def simplex(objectiveFunction, constraintCoefficients, constraintValues,
            eps) -> tuple:
    constraints = []
    objectiveFunction = list(map(lambda x: -x, objectiveFunction))
    objectiveFunction.append(0)
    for i in range(len(constraintCoefficients)):
        constraintCoefficients[i].append(constraintValues[i])
    constraints.append(objectiveFunction)
    for constraint in constraintCoefficients:
        constraints.append(constraint)

    variables = []
    basic_variables = ["Maximum function value"]
    for i in range(1, len(objectiveFunction)):
        variables.append(str(i))
    for i in range(1, len(constraintValues) + 1):
        basic_variables.append("s" + str(i))
        variables.append("s" + str(i))

    while min(constraints[0]) < 0:
        pivot_column = constraints[0].index(min(constraints[0]))
        ratios = [-1]
        for i in range(1, len(constraints)):
            if constraints[i][pivot_column] == 0:
                ratios.append(-1)
                continue
            ratios.append(constraints[i][-1] / constraints[i][pivot_column])
        if len(list(filter(lambda x: x >= 0, ratios))) == 0:
            return ["Method is not applicable"]
        pivot_row = ratios.index(min(filter(lambda x: x >= 0, ratios)))

        enter = variables[pivot_column]
        basic_variables[pivot_row] = enter
        pivot = constraints[pivot_row][pivot_column]
        constraints[pivot_row] = list(map(lambda x: x / pivot, constraints[pivot_row]))

        enter_row = constraints[pivot_row]
        enter_row_index = constraints.index(enter_row)
        constraints.remove(enter_row)
        for i in range(0, len(constraints)):
            current_pivot = constraints[i][pivot_column]
            for j in range(len(constraints[i])):
                constraints[i][j] -= current_pivot * enter_row[j]
        constraints.insert(enter_row_index, enter_row)

    for i in range(len(basic_variables)):
        if "s" in basic_variables[i]:
            constraints.pop(i)
    basic_variables = list(filter(lambda x: "s" not in x, basic_variables))
    constraints = list(map(lambda x: x[-1], constraints))

    x = [0] * (len(objectiveFunction) - 1)
    maximumFunctionValue = constraints.pop(0)
    basic_variables.pop(0)
    for i in range(len(basic_variables)):
        x[int(basic_variables[i]) - 1] = constraints[i]

    x = list(map(lambda x: round(x, eps), x))
    maximumFunctionValue = round(maximumFunctionValue, eps)

    return (x, maximumFunctionValue)


c = np.array([1, 1, 0, 0], np.double).reshape(-1, 1)
a = np.matrix([[2, 4, 1, 0], [1, 3, 0, -1]], np.double)
b = np.array([20, 19], np.double).reshape(-1, 1)
eps = np.double(0.01)
starting_point = np.array([0.5, 3.5, 1, 2], np.double).reshape(-1, 1)

solution05 = solve(c=c, b=b, a=a, eps=eps, alpha=np.double(0.5), starting_point=starting_point)
solution09 = solve(c=c, b=b, a=a, eps=eps, alpha=np.double(0.9), starting_point=starting_point)

print("Interior-point:")
print("alpha = 0.5, z_max =", np.dot(np.transpose(solution05), c), "solution:", solution05.tolist())
print("alpha = 0.9, z_max =", np.dot(np.transpose(solution09), c), "solution:", solution09.tolist())
print("\nSimplex:")
print(simplex(c.reshape(1, -1).tolist()[0], a.tolist(), b.reshape(1, -1).tolist()[0],4))