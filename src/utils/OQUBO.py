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
    Convert a DOcplex optimization model into a Quadratic Unconstrained Binary
    Optimization (QUBO) formulation.
    
    The objective function and constraints of the input optimization model are
    translated into a QUBO representation. Equality constraints are enforced
    through quadratic penalties, while inequality constraints can be handled
    using the Unbalanced Penalization (UP) method.
    
    Parameters
    ----------
    docplexModel : docplex.mp.model.Model
        DOcplex optimization model containing binary decision variables,
        an objective function, and optional constraints.
    
    include : {"Total", "Objective", "Penalization"}, default="Total"
        Specifies which terms are included in the generated QUBO:
    
        - ``"Total"``: objective and penalty terms.
        - ``"Objective"``: objective function only.
        - ``"Penalization"``: penalty terms only.
    
    normalize : bool, default=False
        If True, normalize the final QUBO coefficients by the largest absolute
        coefficient.
    
    obj_normalize : bool, default=False
        If True, normalize only the objective contribution before combining it
        with the penalty terms.
    
    pen_normalize : bool, default=False
        If True, normalize only the penalty contribution before combining it
        with the objective.
    
    UP : bool, default=False
        If True, inequality constraints are converted using the Unbalanced
        Penalization (UP) method. If False, only equality constraints are
        supported.
    
    alpha : float, default=1.0
        Quadratic penalty coefficient used during constraint encoding.
    
    gamma : float, default=1.0
        Constant penalty coefficient used by the UP formulation.
    
    EqC : float, default=2.0
        Penalty coefficient applied to equality constraints.
    
    precision : int, default=20
        Decimal precision used when rounding normalized coefficients.
    
    Returns
    -------
    QUBO : dict[tuple[int, int], float]
        Dictionary representing the QUBO matrix. Keys correspond to variable
        index pairs ``(i, j)`` and values are the associated coefficients.
    
    constant : float
        Constant offset of the QUBO objective function.
    
    References
    ----------
    .. [1] Hernández et al., *Unbalanced penalization: A new approach to encode
           inequality constraints for quantum optimization*,
           Quantum Science and Technology, 2024.
           DOI: 10.1088/2058-9565/ad35e4
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
            # UP == False is not allowed since the other way is to use Slack Variables and it transforms the inquealities to inequalities
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
    Convert a DOcplex optimization model into its equivalent Ising Hamiltonian.
    
    The optimization model is first transformed into a QUBO formulation using
    :func:`DocplexModeltoQUBO`, after which the standard binary-to-spin mapping
    
    .. math::
    
        x_i = \\frac{1 - s_i}{2},
    
    is applied to obtain the corresponding Ising Hamiltonian.
    
    Parameters
    ----------
    docplexModel : docplex.mp.model.Model
        DOcplex optimization model containing binary decision variables.
    
    include : {"Total", "Objective", "Penalization"}, default="Total"
        Specifies which terms are included in the generated Hamiltonian.
    
    normalize : bool, default=False
        If True, normalize the complete QUBO before the Ising conversion.
    
    obj_normalize : bool, default=False
        If True, normalize only the objective contribution.
    
    pen_normalize : bool, default=False
        If True, normalize only the penalty contribution.
    
    UP : bool, default=False
        If True, inequality constraints are encoded using the Unbalanced
        Penalization (UP) method.
    
    alpha : float, default=1.0
        Quadratic penalty coefficient.
    
    gamma : float, default=1.0
        Constant penalty coefficient used by the UP formulation.
    
    EqC : float, default=2.0
        Penalty coefficient for equality constraints.
    
    precision : int, default=20
        Decimal precision used when rounding normalized coefficients.
    
    Returns
    -------
    ising_model : dict[tuple[int, int], float]
        Dictionary containing the Ising Hamiltonian coefficients. Diagonal keys
        ``(i, i)`` correspond to local fields :math:`h_i`, while off-diagonal
        keys ``(i, j)`` correspond to spin-spin couplings :math:`J_{ij}`.
    
    offset : float
        Constant energy offset of the Ising Hamiltonian.
    
    Notes
    -----
    The conversion uses the standard correspondence between binary variables
    and Ising spins,
    
    .. math::
    
        x_i = \\frac{1 - s_i}{2},
    
    where :math:`x_i \\in \\{0,1\\}` and :math:`s_i \\in \\{-1,+1\\}`.
    
    References
    ----------
    .. [1] Hernández et al., *Unbalanced penalization: A new approach to encode
           inequality constraints for quantum optimization*,
           Quantum Science and Technology, 2024.
           DOI: 10.1088/2058-9565/ad35e4
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