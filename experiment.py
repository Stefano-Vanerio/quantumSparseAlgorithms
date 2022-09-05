import numpy as np
from sklearn.datasets import make_sparse_coded_signal
from QOMP import OMP
from QMPa import MP
from QMPb import MP_2
from QMPc import MP_3


# Function that is called from create_data to obtain the data
def qMP_experiment(n, m, L, q_err, kind, version, n_times, error_type, output, delta, seed=1234, L_thresh=0, ):
    # Create a synthetic dataset
    s, D, x = make_sparse_coded_signal(n_samples=n_times,
                                       n_components=m,
                                       n_features=n,
                                       n_nonzero_coefs=L,
                                       random_state=seed)
    # Creation of synthetic dataset with "n_times"-elements, "m"-number of atoms,
    # "n"-size of the signals, "L"-sparsity, "seed"-to replicate experiment

    # Make the signals a bit noisy and compute an error tolerance
    s_noisy = s + 0.05 * np.random.randn(s.shape[0], s.shape[1])

    # Definition of the tolerance as 0.05*sqrt(n), more components, bigger tolerance
    tolerance = 0.05 * np.sqrt(n)

    # Variables used to store the solutions
    class_sol = []
    quant_sol = []
    class_k = []
    quant_k = []
    class_residuals = []
    quant_residuals = []
    class_residuals_full = []
    quant_residuals_full = []
    resc = []
    resq = []
    xc = 0
    xq = 0
    kc = 0
    kq = 0

    # For each sample that I want to analyse do
    for i in range(n_times):

        if kind == 'QOMP':
            xc, kc, resc, xc_a, A_c = OMP(s_noisy[:, i], D, tolerance, error_type, L_thresh, info='all')
            xq, kq, resq, xq_a, A_q = OMP(s_noisy[:, i], D, tolerance, error_type, L_thresh, error=q_err, info='all')
        if kind == 'QMPa':
            xc, kc, resc = MP(s_noisy[:, i], D, tolerance, L_thresh, version=version, info='all')
            xq, kq, resq = MP(s_noisy[:, i], D, tolerance, L_thresh, quantum_error=q_err, version=version, info='all')
        if kind == 'QMPb':
            xc, kc, resc = MP_2(s_noisy[:, i], D, tolerance, error_type, L_thresh, info='all')
            xq, kq, resq = MP_2(s_noisy[:, i], D, tolerance, error_type, L_thresh, error=q_err, info='all')
        if kind == 'QMPc':
            xc, kc, resc = MP_3(s_noisy[:, i], D, tolerance, error_type, L_thresh, info='all')
            xq, kq, resq = MP_3(s_noisy[:, i], D, tolerance, error_type, L_thresh, error=q_err, info='all')

        class_sol.append(xc)            # Classical solutions together
        quant_sol.append(xq)            # Quantum solutions together

        class_k.append(kc)              # Sparsity obtained classically together
        quant_k.append(kq)              # Sparsity obtained quantumly together

        if kind == 'QOMP':
            class_residuals.append(np.linalg.norm(s[:, i] - A_c @ xc_a))  # Residual left classically
            quant_residuals.append(np.linalg.norm(s[:, i] - A_q @ xq_a))  # Residual left quantumly
        else:
            class_residuals.append(np.linalg.norm(s[:, i] - D @ xc))    # Residual left classically
            quant_residuals.append(np.linalg.norm(s[:, i] - D @ xq))    # Residual left quantumly
        
        class_residuals_full.append(resc)               # All residuals scaled classically
        quant_residuals_full.append(resq)               # All residuals scaled quantumly

    class_perc = np.sum(class_residuals <= tolerance)/n_times       # Percentage of blocked for threshold clas.
    quant_perc = np.sum(quant_residuals <= tolerance)/n_times       # Percentage of blocked for threshold quant.
    quantum_run = [quantum_runtime(quant_k[i], n, m, q_err, delta, kind, D, s_noisy[:, i]) for i in range(0, len(quant_k), 1)]

    # Definition of the output, with all you can find in the file
    if output == "all":
        return class_sol, quant_sol, class_k, quant_k, class_residuals, quant_residuals, class_residuals_full, quant_residuals_full, class_perc, quant_perc, quantum_run
    elif output == "sol":
        return class_sol, quant_sol
    elif output == "iter":
        return class_k, quant_k
    elif output == "res_last":
        return class_residuals, quant_residuals
    elif output == "res":
        return class_residuals_full, quant_residuals_full
    elif output == "ressol":
        return class_sol, quant_sol, class_residuals_full, quant_residuals_full
    elif output == "res_all":
        return class_residuals, quant_residuals, class_residuals_full, quant_residuals_full
    elif output == "perc":
        return class_perc, quant_perc

###################################################################### RUNTIME COMPUTATION ######################################################################


def quantum_runtime(k, n, m, xi, delta, kind, D, s):
    if kind == 'QMPa':
        return k*n*np.log(n) + k*np.sqrt(m)*(1/xi)*np.log(3*k*m/delta)*np.log(n*m)
    if kind == 'QMPb':
        return k * n + k * linear_search(D) * np.sqrt(m)*(1/xi) * np.linalg.norm(s) * np.log(6*k*m/delta)**2 * np.log(1/delta) ** 2 * np.log(n*m)
        # k * best_mu(D) * np.linalg.cond(D) * (1/np.linalg.norm(D, 2)) * np.sqrt(m)*(1/xi)  #* np.log(3*k*m/delta)*np.log(n*m) RUNTIME WITH NORM OF D
    if kind == 'QMPc':
        return k * linear_search(D) * np.sqrt(m)*(1/xi) * np.linalg.norm(s) * np.log(6*k*m/delta)**2 * np.log(1/delta) ** 2 * np.log(n*m)
    if kind == 'QOMP':
        return k * linear_search(D) * np.sqrt(m)*(1/xi) * np.linalg.norm(s) * np.log(7*k*m/delta)**2 * np.log(1/delta) ** 3 * np.log(n*m)


def linear_search(matrix, start=0.0, end=1.0, step=0.05):
    domain = [i for i in np.arange(start, end, step)] + [end]
    values = [__mu(i, matrix) for i in domain]
    return min(values)


def __mu(p, matrix):
    def s(p, A):
        if p == 0:
            result = np.max([np.count_nonzero(A[i]) for i in range(len(A))])
        else:
            norms = np.sum(np.power(np.abs(A), p), axis=1)
            result = max(norms)
            del norms
        return result

    s1 = s(2 * p, matrix)
    s2 = s(2 * (1 - p), matrix.T)
    mu = np.sqrt(s1 * s2)

    return mu




