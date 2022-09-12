from sklearn.datasets import make_sparse_coded_signal
from QOMP import OMP
import numpy as np


def MOD_experiment(n, m, L, L_thresh, q_err, n_times, tol, seed=1234, upper=1000):  ###Function that is called from create_data to obtain the data
    # Create a synthetic dataset
    s, D, x = make_sparse_coded_signal(n_samples=n_times,
                                       n_components=m,
                                       n_features=n,
                                       n_nonzero_coefs=L,
                                       random_state=seed)
    ### Creation of synthetic dataset with "n_times"-elements, "m"-number of atoms, "n"-size of the signals, "L"-sparsity, "seed"-to replicate esperiment

    # Make the signals a bit noisy and compute an error tolerance
    s_noisy = s + 0.05 * np.random.randn(s.shape[0], s.shape[1])
    for j in range(0, len(s_noisy[0]), 1):
        if not (np.linalg.norm(s_noisy[:, j]) == 0):
            s_noisy[:, j] = s_noisy[:, j] / np.linalg.norm(s_noisy[:, j])

    ### Noise vector has the same lenght and width of the signal vector (or matrix), at least two samples for each dimension
    tolerance = 0.05 * np.sqrt(n)

    ### Variables used to store the solutions
    count = 0
    cond_c = True
    cond_q = True

    D_c = D + np.random.randn(D.shape[0], D.shape[1])
    for j in range(0, len(D_c[0]), 1):
        if not (np.linalg.norm(D_c[:, j]) == 0):
            D_c[:, j] = D_c[:, j] / np.linalg.norm(D_c[:, j])
    D_q= D_c
    # devo cambiare il dizionario iniziale

    while (count < upper and (cond_c or cond_q)):
        class_sol = list()
        quant_sol = list()
        class_k = list()
        quant_k = list()

        ### For each sample that I want to analyse do
        for i in range(n_times):
            xc, kc = OMP(s_noisy[:, i], D_c, tolerance, '', L_thresh=L_thresh, info='it')
            xq, kq = OMP(s_noisy[:, i], D_q, tolerance, '', L_thresh=L_thresh, error=q_err, info='it')

            class_k.append(kc)
            quant_k.append(kq)
            class_sol.append(xc)  ### Classical solutions together
            quant_sol.append(xq)  ### Quantum solutions together

        count = count + 1

        old_D_c = D_c
        old_D_q = D_q

        X_c = np.array(class_sol)
        X_q = np.array(quant_sol)

        D_c = np.dot(s_noisy, np.linalg.pinv(X_c.transpose()))

        for j in range(0, len(D_c[0]), 1):
            if not(np.linalg.norm(D_c[:, j]) == 0):
                D_c[:, j] = D_c[:, j]/np.linalg.norm(D_c[:, j])

        D_q = np.dot(s_noisy, np.linalg.pinv(X_q.transpose()))
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

    return D_c, D_q, count, class_k, quant_k

print(MOD_experiment(n=100, m=200, L=20, L_thresh=80, q_err=0.01, n_times=29, tol=0.1))
