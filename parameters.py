from sklearn.datasets import make_sparse_coded_signal
from QOMP import OMP
import numpy as np

def mu_kappa(n, m, L, q_err, n_times, seed=1234):  ###Function that is called from create_data to obtain the data
    # Create a synthetic dataset
    s, D, x = make_sparse_coded_signal(n_samples=n_times,
                                       n_components=m,
                                       n_features=n,
                                       n_nonzero_coefs=L,
                                       random_state=seed)
    ### Creation of synthetic dataset with "n_times"-elements, "m"-number of atoms, "n"-size of the signals, "L"-sparsity, "seed"-to replicate esperiment
    print(np.linalg.cond(D))
    # Make the signals a bit noisy and compute an error tolerance
    s_noisy = s + 0.05 * np.random.randn(s.shape[0])

    ### Noise vector has the same lenght and width of the signal vector (or matrix), at least two samples for each dimension
    tolerance = 0.05 * np.sqrt(n)

    n, mu_c, kappa_c = OMP(s_noisy, D, tolerance, '', info='param')

    return mu_c, kappa_c, n

n= 1000
m = int(n * 2)
L = int(n / 5)
ite, mu, ka = mu_kappa(n, m, L, q_err=0.00, n_times=1)
print(ite)
print(mu)
print(ka)


