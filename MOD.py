from sklearn.datasets import make_sparse_coded_signal
from QOMP import OMP
import numpy as np


def MOD_experiment(n, m, L, q_err, n_times, tol, seed=1234, L_thresh=0, upper=1000):  ###Function that is called from create_data to obtain the data
    # Create a synthetic dataset
    s, D, x = make_sparse_coded_signal(n_samples=n_times,
                                       n_components=m,
                                       n_features=n,
                                       n_nonzero_coefs=L,
                                       random_state=seed)
    ### Creation of synthetic dataset with "n_times"-elements, "m"-number of atoms, "n"-size of the signals, "L"-sparsity, "seed"-to replicate esperiment

    # Make the signals a bit noisy and compute an error tolerance
    s_noisy = s + 0.05 * np.random.randn(s.shape[0], s.shape[1])

    ### Noise vector has the same lenght and width of the signal vector (or matrix), at least two samples for each dimension
    tolerance = 0.05 * np.sqrt(n)

    ### Variables used to store the solutions
    count = 0
    cond_c = True
    cond_q = True

    D_q = D + 0.05 * np.random.randn(D.shape[0], D.shape[1])
    D_c = D_q
    # devo cambiare il dizionario iniziale

    while not(count < upper & cond_c & cond_q):
        class_sol = list()
        quant_sol = list()

        ### For each sample that I want to analyse do
        for i in range(n_times):
            xc = OMP(s_noisy[:, i], D_c, tolerance, L_thresh)
            xq = OMP(s_noisy[:, i], D_q, tolerance, L_thresh, error=q_err)

            class_sol.append(xc)  ### Classical solutions together
            quant_sol.append(xq)  ### Quantum solutions together

        count = count + 1

        old_D_c = D_c
        old_D_q = D_q

        X_c = np.array(class_sol)
        X_q = np.array(quant_sol)
        print(len(X_c))
        print(len(X_c[0]))

        D_c = np.dot(s, np.linalg.pinv(X_c.transpose()))
        print(D_c)
        for j in range(0, len(D_c[0]), 1):
            if not(np.linalg.norm(D_c[:, j]) == 0):
                D_c[:, j] = D_c[:, j]/np.linalg.norm(D_c[:, j])

        D_q = np.dot(s, np.linalg.pinv(X_q.transpose()))
        for j in range(0, len(D_q[0]), 1):
            if not(np.linalg.norm(D_q[:, j]) == 0):
                D_q[:, j] = D_q[:, j]/np.linalg.norm(D_q[:, j])

        if np.linalg.norm(D_c-old_D_c) < tol:
            cond_c = False

        if np.linalg.norm(D_q-old_D_q) < tol:
            cond_q = False
        print(count)
        print(np.linalg.norm(D_c-old_D_c))
        print(np.linalg.norm(D_q-old_D_q))

    return D_c, D_q, np.linalg.norm(D_c-old_D_c), np.linalg.norm(D_q-old_D_q), count

print(MOD_experiment(10, 20, 2, 0.01, 10, 0.01))
