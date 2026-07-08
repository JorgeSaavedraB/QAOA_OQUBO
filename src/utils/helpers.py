from src.utils.OQUBO import DocplexModeltoQUBO, QUBOtoIsingModel
import numpy as np

def compute_expect(counts, i, j):
    """
    Compute the expectation value of a single-qubit or two-qubit Pauli-Z operator.

    Given a measurement outcome distribution, this function evaluates either
    the expectation value of :math:`Z_i` (when ``i == j``) or the correlation
    :math:`Z_i Z_j` (when ``i != j``). Bit values are mapped according to the
    computational basis convention:

    - ``0 -> +1``
    - ``1 -> -1``

    Parameters
    ----------
    counts : dict[str, int]
        Dictionary mapping measured bitstrings to their corresponding counts.
    i : int
        Index of the first qubit.
    j : int
        Index of the second qubit. If ``i == j``, the function computes
        :math:`\\langle Z_i \\rangle`; otherwise it computes
        :math:`\\langle Z_i Z_j \\rangle`.

    Returns
    -------
    float
        Expectation value of the specified Pauli-Z observable.
    """
    
    total = sum(counts.values())
    expect = 0
    for bitstring, count in counts.items():
        bit_i = int(bitstring[i])  
        bit_j = int(bitstring[j])
        # Compute <Z_i>, e.g. Z_0
        if i == j:
            value = 1 if bit_i == 0 else -1
        # Compute <Z_i Z_j>, e.g. Z_0 Z_1
        else:
            value_i = 1 if bit_i == 0 else -1
            value_j = 1 if bit_j == 0 else -1
            value = value_i * value_j
        expect += value * count / total
    return expect

def compute_cost(solution, IsingModel: tuple):
    """
    Evaluate the energy of a measurement distribution with respect to an Ising Hamiltonian.

    Parameters
    ----------
    solution : dict[str, int]
        Dictionary mapping measured bitstrings to their corresponding counts.
    IsingModel : tuple
        Tuple ``(hamiltonian, constant)`` where:

        - ``hamiltonian`` is a dictionary mapping qubit index pairs
          ``(i, j)`` to their Ising coefficients.
        - ``constant`` is the Hamiltonian energy offset.

    Returns
    -------
    float
        Energy expectation value of the provided measurement distribution.
    """
    
    IsingHamiltonian, constant = IsingModel
    exp_val = 0
    for k, v in IsingHamiltonian.items():
        i = k[0]
        j = k[1]
        exp_val += v * compute_expect(solution, i, j)
    exp_val += constant
    return exp_val

def BruteForceSolutions(model, 
                        alpha, 
                        gamma,
                        EqC=2,
                        include='Total', 
                        normalize=False, 
                        obj_normalize=False, 
                        pen_normalize=False, 
                        UP=False,
                        precision=20):
    """
    Compute the exact energy of every computational basis state.

    This function converts the provided QUBO model into its corresponding
    Ising Hamiltonian and exhaustively evaluates the energy of all
    :math:`2^n` possible bitstrings, where ``n`` is the number of binary
    variables in the model.

    The resulting dictionary is sorted in ascending order of energy, allowing
    the exact ground state and excited states to be identified.

    Parameters
    ----------
    model
        Optimization model containing binary decision variables.
    alpha : float
        Scaling factor applied to the objective function during the
        QUBO-to-Ising conversion.
    gamma : float
        Scaling factor applied to the penalty terms.
    EqC : int, default=2
        Equality constraint coefficient used during model conversion.
    include : {"Total", "Objective", "Penalty"}, default="Total"
        Specifies which Hamiltonian terms are included in the conversion.
    normalize : bool, default=False
        Whether to normalize the complete Hamiltonian.
    obj_normalize : bool, default=False
        Whether to normalize only the objective contribution.
    pen_normalize : bool, default=False
        Whether to normalize only the penalty contribution.
    UP : bool, default=False
        Enables the UP normalization strategy during conversion.
    precision : int, default=20
        Numerical precision used during the QUBO-to-Ising transformation.

    Returns
    -------
    dict[str, float]
        Dictionary mapping every computational basis bitstring to its
        corresponding Ising energy, sorted in ascending order.
    """
    
    Ising_model = QUBOtoIsingModel(model,
                                    include=include, 
                                    normalize=normalize, 
                                    obj_normalize=obj_normalize, 
                                    pen_normalize=pen_normalize, 
                                    UP=UP,
                                    alpha=alpha, 
                                    gamma=gamma,
                                    EqC=EqC,
                                    precision=precision
                                   )
    energies_dict = {}
    n_qubits = len(list(model.iter_binary_vars()))
    for case_i in range(2**n_qubits):
        bit_string = str(np.binary_repr(case_i, n_qubits))
        sol_cost = compute_cost({bit_string: 1}, Ising_model)
        energies_dict[bit_string] = sol_cost
    sort_sol = sorted(energies_dict, key=energies_dict.get, reverse=False)
    sol_dict = {key: energies_dict[key] for key in sort_sol}
    return sol_dict