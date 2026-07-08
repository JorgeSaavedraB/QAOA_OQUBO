import numpy as np
from docplex.mp.model import Model
import cplex
from collections import defaultdict

def get_neighbors(rows: int, 
                  cols: int, 
                  J: list, 
                  S: list):
    """
    Compute the coverage neighborhood of each demand node.

    The grid is assumed to be arranged in row-major order. For every demand
    node, the function identifies all eligible facility locations within a
    Euclidean distance not exceeding ``S``.

    Parameters
    ----------
    rows : int
        Number of rows in the grid.

    cols : int
        Number of columns in the grid.

    J : list[int]
        Binary indicator of eligible facility locations. An entry equal to
        ``1`` denotes that the corresponding grid cell can host a facility.

    S : int or float
        Coverage radius measured in grid units.

    Returns
    -------
    collections.defaultdict[list]
        Dictionary mapping each demand node to the indices of the eligible
        facilities capable of covering it.
    """
    
    potential_sites = [idx for idx, val in enumerate(J) if val==1]
    
    sites = {(i): (i // cols, i % cols) for i in range(rows * cols)}
    
    N = defaultdict(list)
    for i in range(rows * cols):
        for j in potential_sites:
            d = (sites[i][0] - sites[j][0]) ** 2 + (sites[i][1] - sites[j][1]) ** 2
            if d <= S**2:
                N[i].append(potential_sites.index(j))
    return N

def second_factor(n):  
    """
    Return the largest factor of ``n`` smaller than ``n``.

    The function is used to infer a rectangular grid from the total number
    of nodes by computing a non-trivial divisor.

    Parameters
    ----------
    n : int
        Positive integer.

    Returns
    -------
    int
        Largest divisor of ``n`` strictly smaller than ``n``. If ``n`` is
        prime, ``n`` itself is returned.
    """
    
    for i in range(2, n):  
        if n % i == 0: 
           return n // i  
    return n 

def MCLP(J: np.ndarray, 
         I: np.ndarray,
         p: int,
         S: int,
         version: str = 'COP',
         slack_vars:bool = False) -> Model:
    """
    Build a DOcplex model for the Maximum Covering Location Problem (MCLP).

    The objective is to maximize the total covered demand by selecting
    exactly ``p`` facility locations from a set of eligible sites. Demand
    nodes are considered covered if at least one selected facility lies
    within the specified coverage radius.

    Parameters
    ----------
    J : numpy.ndarray
        Binary array indicating the eligible facility locations. Entries
        equal to ``1`` correspond to candidate facility sites.

    I : numpy.ndarray
        Demand weight associated with each node.

    p : int
        Number of facilities to locate.

    S : int or float
        Coverage radius measured in grid units.

    version : {"COP", "CONT"}, default="COP"
        Type of optimization model to construct.

        - ``"COP"``: binary combinatorial optimization model.
        - ``"CONT"``: continuous relaxation with variables in
          :math:`[0,1]`.

    slack_vars : bool, default=False
        If True, each coverage inequality is reformulated as an equality
        using binary slack variables. Otherwise, the coverage constraints
        are enforced directly as inequalities.

    Returns
    -------
    docplex.mp.model.Model
        DOcplex optimization model representing the Maximum Covering
        Location Problem.

    Notes
    -----
    The optimization model minimizes the negative covered demand, which is
    equivalent to maximizing the total covered demand.
    """
    
    n = len(I) # number of demand nodes
    m = sum(J) # number of eligible facilities
        
    cols = int(second_factor(n))
    rows = int(n // cols)
    
    model = Model(name = 'MCLP')

    if version == 'COP':
        # Binary variables 
        y = np.array(model.binary_var_list(n, name='y')) # covered demand
        x = np.array(model.binary_var_list(m, name='x')) # open sites
    elif version == 'CONT':
        # Continuous variables 
        y = np.array(model.continuous_var_list(n, lb=0, ub=1, name='y')) # covered demand
        x = np.array(model.continuous_var_list(m, lb=0, ub=1, name='x')) # open sites
    
    # Objective function
    objective_function = model.sum(- 1* I[i] * y[i] for i in range(n))
    model.minimize(objective_function)
    
    # Constraints
    neighbors = get_neighbors(rows, cols, J, S)
    
    s = {}
    for i in range(n):
        if neighbors[i]:
            if slack_vars==False:
                model.add_constraint(y[i] <= model.sum(x[j] for j in neighbors[i]), ctname=f'coverage_{i}')
            else: 
                N_bits_i = int(np.ceil(np.log2(len(neighbors[i])) + 1))
                if version == 'COP':
                    s[i] = np.array(model.binary_var_list(N_bits_i, name=f'slack{i}'))
                elif version == 'CONT':
                    s[i] = np.array(model.continuous_var_list(N_bits_i, lb=0, ub=1, name=f'slack{i}'))
                S_i = model.sum(2**j * s[i][j] for j in range(N_bits_i))
                model.add_constraint(y[i] + S_i == model.sum(x[j] for j in neighbors[i]) , ctname=f'coverage_{i}')
        else:
            model.add_constraint(y[i] == 0, ctname=f'coverage_{i}')

    model.add_constraint(model.sum(x[i] for i in range(m)) == p, ctname='numspots')
 
    
    return model