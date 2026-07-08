from src.utils.OQUBO import DocplexModeltoQUBO, QUBOtoIsingModel
import numpy as np

def solution(model):
    """
    Solve a DOcplex model and return the optimal binary assignment.

    The model is solved using the configured DOcplex solver (typically CPLEX),
    and the values of the binary decision variables are returned as a bitstring
    ordered according to ``model.iter_binary_vars()``.

    Parameters
    ----------
    model : docplex.mp.model.Model
        DOcplex optimization model containing binary decision variables.

    Returns
    -------
    str
        Bitstring representing the optimal binary solution.

    Raises
    ------
    RuntimeError
        If the optimization problem has no feasible solution.
    """
    
    docplex_sol = model.solve() # Solves the docplex model using cplex
    solution = ''
    for i in model.iter_binary_vars():
        solution += str(int(docplex_sol[i]))
    return solution

def cont_solution(model):
    """
    Solve a DOcplex model and return the optimal continuous variable values.

    The model is solved using the configured DOcplex solver (typically CPLEX),
    and the values of the continuous decision variables are returned in the
    order provided by ``model.iter_continuous_vars()``.

    Parameters
    ----------
    model : docplex.mp.model.Model
        DOcplex optimization model containing continuous decision variables.

    Returns
    -------
    list[float]
        Optimal values of the continuous variables, rounded to six decimal
        places.

    Raises
    ------
    RuntimeError
        If the optimization problem has no feasible solution.
    """
    
    solution = model.solve()# Solves the docplex model using cplex
    rlxd_sol = []
    for i in model.iter_continuous_vars():
        rlxd_sol.append(round(solution[i], 6))
    return rlxd_sol

def check_feasibility(docplexModel, solution):
    """
    Check whether a candidate assignment satisfies all model constraints.

    A copy of the original DOcplex model is created, the decision variables are
    fixed to the values specified in ``solution``, and the resulting model is
    solved as a feasibility problem.

    Parameters
    ----------
    docplexModel : docplex.mp.model.Model
        DOcplex optimization model.

    solution : str or iterable of int
        Candidate assignment for the model variables. The values are assumed
        to follow the same ordering as ``docplexModel.iter_variables()``.

    Returns
    -------
    bool
        ``True`` if the assignment is feasible, otherwise ``False``.
    """
    
    # Make a temporary copy of the model
    mtest = docplexModel.clone()

    # Fix variables to candidate values
    for v, s in zip(mtest.iter_variables(), solution):
        mtest.add_constraint(v == int(s))

    # Solve feasibility
    feasible = mtest.solve() is not None

    return feasible

def WS_parameters(rlxd_solution, epsilon=0.001):
    """
    Compute Warm-Start (WS) parameters from a relaxed solution.
    
    The relaxed variable values are converted into rotation angles according
    to
    
    .. math::
    
        \\theta_i = 2\\arcsin(\\sqrt{x_i}),
    
    where :math:`x_i` is the relaxed value of the corresponding binary
    variable. Values within ``epsilon`` of 0 or 1 are clipped to avoid
    exactly zero or :math:`\\pi` rotations.
    
    Parameters
    ----------
    rlxd_solution : iterable of float
        Relaxed solution values, typically obtained by solving the continuous
        relaxation of the optimization model.
    
    epsilon : float, default=0.001
        Clipping threshold used to bound the relaxed values away from 0 and 1.
    
    Returns
    -------
    list[float]
        Rotation angles (in radians) corresponding to the relaxed solution,
        rounded to six decimal places.
    
    Notes
    -----
    These parameters are commonly used to initialize variational quantum
    algorithms through warm-start strategies.
    
    References
    ----------
    .. [1] D. J. Egger, J. Mareček, and S. Woerner,
           *Warm-starting quantum optimization*,
           Quantum **5**, 479 (2021).
           DOI: 10.22331/q-2021-06-17-479
    """
    ws_params = []
    for i in rlxd_solution:
        if i <= 1 - epsilon and i >= epsilon:
            param = round(2*np.arcsin(np.sqrt(round(i, 6))), 6)
        elif i < epsilon:
            param = round(2*np.arcsin(np.sqrt(round(epsilon, 6))), 6)
        elif i > 1 - epsilon:
            param = round(2*np.arcsin(np.sqrt(round(1 - epsilon, 6))), 6)
        ws_params.append(param)
    return ws_params