from docplex.mp.model import Model
import numpy as np

def portfolio_optimization(mu: np.ndarray,
                           sigma: np.ndarray, 
                           risk: float, 
                           budget: int = None,
                           slack_vars:bool = False,
                           version: str = 'COP') -> Model:
    """
    Construye el modelo de programación binaria para optimizar portafolio.
    
    Args:
      mu: vector de retornos esperados (n,) 
      sigma: matriz de covarianza (n,n)
      risk: factor de riesgo
      budget: número máximo de activos
    
    Returns:
      docplex.mp.model.Model optimizado
    """
    N_stocks = len(mu)
    
    model = Model(name = 'Portfolio_Optimization')
    
    # Binary variables that represent the stocks
    if version == 'COP':
        x = np.array(model.binary_var_list(N_stocks, name='stock'))
    elif version == 'CONT':
        x = np.array(model.continuous_var_list(N_stocks, lb=0, ub=1, name='stock'))
    
    # Objective function
    diag = np.diag(sigma)
    sum_var = np.sum(diag)
    mean_var = np.mean(diag)
    
    risk_term = sum((sigma[i, j] * x[i] * x[j]) / (2)
                          for i in range(N_stocks) for j in range(N_stocks))
    return_term = sum((mu[i]) * x[i] for i in range(N_stocks))
    model.minimize(10*(risk * risk_term - return_term))
    
    # Budget constraint
    if slack_vars==False:
        model.add_constraint(model.sum(x) <= budget, ctname='budget')
    else:
        N = int(np.ceil(np.log2(budget + 1)))
        if version == 'COP':
            s = np.array(model.binary_var_list(N, name='slack'))
        elif version == 'CONT':
            s = np.array(model.continuous_var_list(N, lb=0, ub=1, name='slack'))
        S = np.array([2**n * s[n] for n in range(N)])
        model.add_constraint(model.sum(x) + model.sum(S) == budget, ctname='budget')
    return model

def data_generator(df):
    dates = df.index
    total_days = len(dates)
    
    df_ret = df.pct_change().dropna()
    
    sigma = df_ret.cov() * total_days   # annualized covariance
    mu = df_ret.mean() * total_days     # annualized expected return
    
    budget = len(mu) // 3
    return mu.to_numpy(), sigma.to_numpy(), budget