from scipy.stats import truncnorm
import numpy as np


# Define the opposite of the stopping condition: stop if ||x||_0 > L or ||r||_2 <= \epsilon
def continuing_criterium(x, r, tolerance, L_thresh):
    # if L == 0 use only ||r||_2 to consider the next iteration
    if L_thresh != 0:
        sparsity_check = np.linalg.norm(x, ord=0) <= L_thresh
    else:
        sparsity_check = True

    return sparsity_check and np.linalg.norm(r) > tolerance


# Matching Pursuit algorithm
# s is the dense vector;
# D is the dictionary;
# tolerance is \epsilon, the reconstruction error;
# L is the sparsity threshold;
# info specifies the output type;
# quantum_error is the IPE error \xi;
# version can be "both" or "single", see the implementation and the paper for more info.
def MP(s, D, tolerance, L=0, info='', quantum_error=0, version=''):
    N, M = D.shape
    # Initialize the solution vector to zeros
    x = np.zeros(M)

    # Initialize the residual
    # i.e., components not yet captured by the sparse signal
    r = s

    # init the number of iterations k
    num_iterations = 1
    residuals = [np.linalg.norm(r)]

    while continuing_criterium(x, r, tolerance, L):
        z = np.zeros(M)

        # Compute all the inner products
        for j in range(M):
            z[j] = D[:, j].dot(r)

        # qMP's V1 introduces error both on j* and on z
        # the error is scaled by the residual's norm: \xi * ||r||_2
        if version == 'both' and quantum_error != 0:
            z = z + truncnorm.rvs(-quantum_error, quantum_error, size=M) * np.linalg.norm(r)

        # to find the minimum e(j) we just need to search for the maximum |dj^T r|
        jStar = np.argmax(np.abs(z))

        # qMP's V2 introduces error both on j* and on z.
        # i.e., |z' - z| <= quantum_error*||r||
        if version == 'single' and quantum_error != 0:
            jStar = np.argmax(np.abs(z + truncnorm.rvs(-quantum_error, quantum_error, size=M) * np.linalg.norm(r)))

        # update the solution
        x[jStar] = x[jStar] + z[jStar]

        # update the residual
        r = r - z[jStar] * D[:, jStar]

        residuals.append(np.linalg.norm(r))

        # update the number of iterations so far
        num_iterations += 1

    if info == 'it':
        return x, num_iterations
    if info == 'residual':
        return x, residuals
    if info == 'all':
        return x, num_iterations, residuals
    else:
        return x
