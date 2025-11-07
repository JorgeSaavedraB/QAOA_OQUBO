from src.utils.OQUBO import DocplexModeltoQUBO, QUBOtoIsingModel
import numpy as np

def compute_expect(counts, i, j):
    total = sum(counts.values())
    expect = 0
    for bitstring, count in counts.items():
        bit_i = int(bitstring[i])  
        bit_j = int(bitstring[j])
        # Calcular <Z_i>, por ejemplo Z_0
        if i == j:
            value = 1 if bit_i == 0 else -1
        # Calcular <Z_i Z_j>, por ejemplo Z_0 Z_1
        else:
            value_i = 1 if bit_i == 0 else -1
            value_j = 1 if bit_j == 0 else -1
            value = value_i * value_j
        expect += value * count / total
    return expect

def compute_cost(solution, IsingModel: tuple):
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
                        include='Total', 
                        normalize=False, 
                        obj_normalize=False, 
                        pen_normalize=False, 
                        UP=False,
                        precision=20):
    Ising_model = QUBOtoIsingModel(model,
                                    include=include, 
                                    normalize=normalize, 
                                    obj_normalize=obj_normalize, 
                                    pen_normalize=pen_normalize, 
                                    UP=UP,
                                    alpha=alpha, 
                                    gamma=gamma, 
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