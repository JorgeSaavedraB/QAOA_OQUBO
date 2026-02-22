import numpy as np
from docplex.mp.model import Model
import cplex
from collections import defaultdict

def get_neighbors(rows: int, 
                  cols: int, 
                  J: list, 
                  S: list):
    potential_sites = [idx for idx, val in enumerate(J) if val==1]
    
    sites = {(i): (i//cols, i % cols) for i in range(rows*cols)}
    
    N = defaultdict(list)
    for i in range(rows * cols):
        for j in [i for i, e in enumerate(J) if e == 1]:
            distance = np.sqrt((sites[i][0] - sites[j][0]) ** 2 + (sites[i][1] - sites[j][1]) ** 2)
            if distance <= S:
                N[i].append(potential_sites.index(j))
    return N

def MCLP(J: np.ndarray, 
         I: np.ndarray,
         p: int,
         S: int,
         version: str = 'COP',
         slack_vars:bool = False) -> Model:
    """
    Args:
    
    Returns:
      docplex.mp.model.Model 
    """
    n = len(I)
    m = sum(J) # numero de sitios potenciales
        
    rows = cols = int(len(I) ** (1/2))
    
    model = Model(name = 'MCLP')

    if version == 'COP':
        # Binary variables that represent the stocks
        y = np.array(model.binary_var_list(n, name='y')) # covered demand
        x = np.array(model.binary_var_list(m, name='x')) # open sites
    elif version == 'CONT':
        # Continuous variables that represent the stocks
        y = np.array(model.continuous_var_list(n, lb=0, ub=1, name='y')) # covered demand
        x = np.array(model.continuous_var_list(m, lb=0, ub=1, name='x')) # open sites
    
    # Objective function
    max_pop = max(I)
    objective_function = sum(- 1* I[i] * y[i] for i in range(n))
    model.minimize(objective_function)
    
    # Constraints
    neighbors = get_neighbors(rows, cols, J, S)
    
    s = {}
    for i in range(n):
        if neighbors[i]:
            if slack_vars==False:
                model.add_constraint(y[i] <= sum((x[j] for j in neighbors[i])), ctname=f'coverage_{i}')
            else: 
                N_bits_i = int(np.ceil(np.log2(len(neighbors[i])) + 1))
                if version == 'COP':
                    s[i] = np.array(model.binary_var_list(N_bits_i, name=f'slack{i}'))
                elif version == 'CONT':
                    s[i] = np.array(model.continuous_var_list(N_bits_i, lb=0, ub=1, name=f'slack{i}'))
                S_i = model.sum(2**j * s[i][j] for j in range(N_bits_i))
                model.add_constraint(y[i] + S_i == sum((x[j] for j in neighbors[i])) , ctname=f'coverage_{i}')
        else:
            model.add_constraint(y[i] == 0, ctname=f'coverage_{i}')

    model.add_constraint(sum((x[i] for i in range(m))) == p, ctname='numspots')
 
    
    return model