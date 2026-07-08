from docplex.mp.model import Model
import numpy as np

def portfolio_optimization(mu: np.ndarray,
                           sigma: np.ndarray, 
                           risk: float, 
                           budget: int = None,
                           slack_vars:bool = False,
                           version: str = 'COP') -> Model:
    """
    Build a portfolio optimization model as a DOcplex optimization problem.
    
    The model minimizes the mean-variance objective
    
    .. math::
    
        \min_x \; \lambda x^\top \Sigma x - \mu^\top x,
    
    subject to a budget constraint on the number of selected assets. Binary
    decision variables define the combinatorial optimization problem, while
    continuous variables can be used to obtain its continuous relaxation.
    
    Parameters
    ----------
    mu : numpy.ndarray
        One-dimensional array containing the expected return of each asset.
    
    sigma : numpy.ndarray
        Covariance matrix of asset returns.
    
    risk : float
        Risk-aversion coefficient controlling the trade-off between expected
        return and portfolio variance.
    
    budget : int, optional
        Maximum number of assets that can be selected.
    
    slack_vars : bool, default=False
        If True, the budget inequality is reformulated as an equality using
        binary slack variables. Otherwise, the budget is imposed directly as
        an inequality constraint.
    
    version : {"COP", "CONT"}, default="COP"
        Type of optimization model to construct.
    
        - ``"COP"``: binary combinatorial optimization model.
        - ``"CONT"``: continuous relaxation with variables in :math:`[0,1]`.
    
    Returns
    -------
    docplex.mp.model.Model
        DOcplex model representing the portfolio optimization problem.
    
    Notes
    -----
    The objective follows the classical mean-variance portfolio optimization
    framework, balancing expected return against portfolio risk through the
    ``risk`` parameter.
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
    # The objective function is multiplied by a constant scaling factor (10)
    
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
    """
    Generate portfolio optimization inputs from historical price data.
    
    The function computes annualized expected returns and the annualized
    covariance matrix from a time series of asset prices. A default investment
    budget equal to one third of the available assets is also returned.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing historical asset prices. Rows correspond to
        observation dates and columns correspond to individual assets.
    
    Returns
    -------
    mu : numpy.ndarray
        Annualized expected returns.
    
    sigma : numpy.ndarray
        Annualized covariance matrix of asset returns.
    
    budget : int
        Default investment budget, computed as one third of the number of
        available assets.
    
    Notes
    -----
    Expected returns are computed as the mean of the daily percentage returns,
    while the covariance matrix is obtained from the daily return covariance.
    Both quantities are annualized by multiplying by the number of observations
    in the input data.
    """
    
    dates = df.index
    total_days = len(dates)
    
    df_ret = df.pct_change().dropna()
    
    sigma = df_ret.cov() * total_days   # annualized covariance
    mu = df_ret.mean() * total_days     # annualized expected return
    
    budget = len(mu) // 3
    return mu.to_numpy(), sigma.to_numpy(), budget