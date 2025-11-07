from src.utils.OQUBO import DocplexModeltoQUBO, QUBOtoIsingModel
import numpy as np

def solution(model):
    docplex_sol = model.solve() # Solves the docplex model using cplex
    solution = ''
    for i in model.iter_binary_vars():
        solution += str(int(docplex_sol[i]))
    return solution

def cont_solution(model):
    solution = model.solve()# Solves the docplex model using cplex
    rlxd_sol = []
    for i in model.iter_continuous_vars():
        rlxd_sol.append(round(solution[i], 6))
    return rlxd_sol

def check_feasibility(docplexModel, solution):
    # Make a temporary copy of the model
    mtest = docplexModel.clone()

    # Fix variables to candidate values
    for v, s in zip(mtest.iter_variables(), solution):
        mtest.add_constraint(v == int(s))

    # Solve feasibility
    feasible = mtest.solve() is not None

    return feasible

def WS_parameters(rlxd_solution, epsilon=0.001):
    ws_params = []
    for i in rlxd_solution:
        if i <= 1 - epsilon and i >= epsilon:
            param = round(2*np.arcsin(np.sqrt(round(i, 6))), 6)
        elif i < epsilon:
            param = round(2*np.arcsin(np.sqrt(round(epsilon, 6))), 6)
        elif i > 1 - epsilon:
            param = round(2*np.arcsin(np.sqrt(round(1 - epsilon, 6))), 6)
    # return [round(2*np.arcsin(np.sqrt(round(i, 9))), 9) for i in rlxd_solution]
        ws_params.append(param)
    return ws_params