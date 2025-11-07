import numpy as np
from collections import defaultdict
import copy
import matplotlib.pyplot as plt

def DocplexModeltoQUBO(docplexModel, 
                       include='Total', 
                       normalize=False, 
                       obj_normalize=False, 
                       pen_normalize=False,
                       UP=False,
                       alpha=1.0, 
                       gamma=1.0, 
                       EqC=2.0,
                       precision=20):
    """
    Convert a DOcplex optimization model into a Quadratic Unconstrained Binary Optimization (QUBO) formulation.

    Parameters
    ----------
    docplexModel : docplex.mp.model.Model
        A DOcplex model object containing the objective and constraints.

    include : {'Total', 'Objective', 'Penalization'}, default='Total'
        Determines which terms are included in the QUBO.

    exponential : bool, default=True
        If True, applies exponential (λ1, λ2 terms).

    softplus : bool, default=False
        If True, uses a quadratic approximation of softplus as penalization.

    alpha : float
        Scale factor for softplus penalization.

    gamma : float
        Smoothness parameter for softplus.

    approx_range : tuple, default=(-3, 3)
        Interval of u-values where the quadratic approximation of softplus is fitted.

    approx_points : int, default=200
        Number of sample points for fitting the quadratic approximation.

    Returns
    -------
    QUBO : dict
        Dictionary representation of the QUBO with keys as `(i, j)` variable index pairs.

    constant : float
        Constant offset term in the QUBO formulation.
    """
    var_assign = {var.name: num for num, var in enumerate(docplexModel.iter_variables())}
    QUBO_obj = {}
    try:
        constant_obj = docplexModel.objective_expr.constant
    except:
        constant_obj = 0
    
    # Objective
    for v1, v2, coeff in docplexModel.objective_expr.iter_quad_triplets():
        key = tuple(sorted([var_assign[v1.name], var_assign[v2.name]]))
        QUBO_obj[key] = QUBO_obj.get(key, 0) + coeff
    for v, coeff in docplexModel.objective_expr.iter_terms():
        idx = var_assign[v.name]
        QUBO_obj[(idx, idx)] = QUBO_obj.get((idx, idx), 0) + coeff
    
    # Penalization
    QUBO_total_pen = {}
    constant_total_pen = 0
    norm_const = 0
    
    for cstr in docplexModel.iter_constraints():  
        QUBO_pen = {}
        constant_pen = 0.0 
        lhs = copy.copy(cstr.lhs)
        rhs = copy.copy(cstr.rhs)
        if not rhs.is_constant():
            lhs = lhs - rhs
            rhs = 0
        else:
            rhs = rhs.constant
        constr_dict = {var_assign[v.name]: coeff for v, coeff in lhs.iter_terms()}

        if cstr.sense.name == 'EQ':
            a = EqC
            b = 0
            c = 0
        else:
            if UP: 
                a = alpha
                b = alpha
                c = gamma
        sign = 1 if cstr.sense.name == 'LE' else -1
        rhs_signed = sign * rhs
        constr_dict = {k: sign * v for k, v in constr_dict.items()}
        
        # Quadratic terms
        keys = list(constr_dict.keys())
        for i in keys:
            vi = constr_dict[i]
            QUBO_pen[(i, i)] = QUBO_pen.get((i, i), 0.0) + a * vi**2 + b * vi - 2*a*rhs_signed*vi 
            norm_const += a * vi**2 + b * vi - 2*a*rhs_signed*vi 
            # off-diagonal
            for j in keys:
                if j <= i:
                    continue
                vj = constr_dict[j]
                QUBO_pen[(i, j)] = QUBO_pen.get((i, j), 0.0) + 2*a*vi*vj
                norm_const += 2*a*vi*vj
        # Constant part 
        constant_pen += a * rhs_signed**2 - b * rhs_signed + c
        norm_const += a * rhs_signed**2 - b * rhs_signed + c

        # accumulate
        QUBO_total_pen = {k: QUBO_total_pen.get(k, 0) + QUBO_pen.get(k, 0)
                          for k in set(QUBO_total_pen) | set(QUBO_pen)}
        constant_total_pen += constant_pen 
    
    # Normalizaciones
    if pen_normalize:
        max_pen = max([abs(val) for val in QUBO_total_pen.values()], default=0)
        if max_pen != 0:
            QUBO_total_pen = {key: val/max_pen for key, val in QUBO_total_pen.items()}
            constant_total_pen = constant_total_pen/max_pen

    if obj_normalize:
        max_obj = max([abs(val) for val in QUBO_obj.values()], default=0)
        if max_obj != 0:
            QUBO_obj = {key: val/max_obj for key, val in QUBO_obj.items()}
            constant_obj = constant_obj/max_obj 
    
    if include == 'Total':
        QUBO = {key: QUBO_obj.get(key, 0) + QUBO_total_pen.get(key, 0) 
                for key in (set(QUBO_obj) | set(QUBO_total_pen))}
        constant = constant_obj + constant_total_pen
    elif include == 'Objective':
        QUBO = QUBO_obj
        constant = constant_obj
    elif include == 'Penalization':
        QUBO = QUBO_total_pen
        constant = constant_total_pen
    else:
        raise ValueError("Invalid option for `include`")

    if normalize:
        max_qubo = max([abs(val) for val in QUBO.values()], default=0)
        #max_qubo = norm_const
        if max_qubo != 0: 
            QUBO = {key: round(val/max_qubo, precision) for key, val in QUBO.items()}
            constant = round(constant / max_qubo, precision)
    
    return QUBO, constant

def QUBOtoIsingModel(docplexModel, 
                       include='Total', 
                       normalize=False, 
                       obj_normalize=False, 
                       pen_normalize=False,
                       UP=False,
                       alpha=1.0, 
                       gamma=1.0, 
                       EqC=2.0,
                       precision=20):
    """
    Convert a DOcplex optimization model into an Ising model formulation 
    (from its QUBO representation).

    Parameters
    ----------
    docplexModel : docplex.mp.model.Model
        A DOcplex model object containing the objective and constraints.

    UPcoefficients : list of float, default=[0.5, 0.5]
        Coefficients for unbalanced penalization (see `DocplexModeltoQUBO`).

    include : {'Total', 'Objective', 'Penalization'}, default='Total'
        Determines which terms are included in the model conversion:
        - 'Total': Objective + penalization terms.
        - 'Objective': Only the original objective terms.
        - 'Penalization': Only the penalization terms.

    normalize : bool, default=False
        If True, normalizes the entire QUBO coefficients before conversion.

    obj_normalize : bool, default=False
        If True, normalizes only the objective coefficients before conversion.

    pen_normalize : bool, default=False
        If True, normalizes only the penalization coefficients before conversion.

    exponential : bool, default=True
        If True, applies exponential when constructing the QUBO.

    Returns
    -------
    ising_model : dict
        Dictionary containing the Ising model coefficients. 
        Keys are `(i, i)` for local fields (h) and `(i, j)` for couplings (J).

    offset : float
        Constant offset term in the Ising Hamiltonian.

    Notes
    -----
    - The transformation is based on the standard mapping between binary variables 
      (0/1 in QUBO) and spin variables (-1/+1 in Ising).
    - The QUBO diagonal terms contribute to both the local fields and the offset.
    - Off-diagonal QUBO terms contribute to couplings (J) and redistribute into local fields.
    """
    QUBO, cnt = DocplexModeltoQUBO(docplexModel, 
                       include=include, 
                       normalize=normalize, 
                       obj_normalize=obj_normalize, 
                       pen_normalize=pen_normalize,
                       UP=UP,
                       alpha=alpha, 
                       gamma=gamma, 
                       EqC=EqC,
                       precision=precision)
    n = docplexModel.number_of_variables
    J = defaultdict(float)
    h = defaultdict(float)
    offset = cnt
    
    for i in range(n):
        qii = QUBO.get((i, i), 0)
        h[(i, i)] -= qii / 2 
        offset += qii / 2
        for j in range(i + 1, n):
            qij = QUBO.get((i, j), 0)
            if qij != 0:
                h[(i, i)] -= qij / 4
                h[(j, j)] -= qij / 4
                J[(i, j)] += qij / 4 
                offset += qij / 4
                
    ising_model = {key: J.get(key, 0) + h.get(key, 0) for key in (set(J) | set(h))}
        
    return ising_model, offset