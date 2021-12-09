from sklearn.datasets import make_sparse_coded_signal
from scipy.stats import truncnorm
import numpy as np

# Define the opposite of the stopping condition: stop if ||x||_0 > L or ||r||_2 <= \epsilon
def continuing_criterium(x, r, tolerance, L):
    # if L == 0 use only ||r||_2 to consider the next iteration
    if L!=0:
        sparsity_check = np.linalg.norm(x, ord=0) <= L
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
def MP(s, D, tolerance, L=0, info = '', quantum_error=0, version=''):
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
            z[j] = D[:,j].dot(r)
            
        # qMP's V1 introduces error both on j* and on z
        # the error is scaled by the residual's norm: \xi * ||r||_2
        if version=='both' and quantum_error != 0:
            z = z + truncnorm.rvs(-quantum_error,quantum_error, size=M)*np.linalg.norm(r)
        
        # to find the minimum e(j) we just need to search for the maximum |dj^T r|
        jStar = np.argmax(np.abs(z))
    
        # qMP's V2 introduces error both on j* and on z. 
        # i.e., |z' - z| <= quantum_error*||r||
        if version=='single' and quantum_error != 0:
            jStar = np.argmax(np.abs(z + truncnorm.rvs(-quantum_error,quantum_error, size=M)*np.linalg.norm(r)))
        
        # update the solution
        x[jStar] = x[jStar] + z[jStar]

        # update the residual
        r = r - z[jStar]*D[:,jStar]

        residuals.append(np.linalg.norm(r))
       
        # update the number of iterations so far
        num_iterations += 1

    if info=='it':
        return x, num_iterations
    if info=='residual':
        return x, residuals
    if info=='all':
        return x, num_iterations, residuals
    else:
        return x

def qMP_experiment(n, m, L, q_err, version, n_times, output, seed=1234, L_thresh=0):
    # Create a synthetic dataset
    s, D, x = make_sparse_coded_signal(n_samples=n_times,
                                   n_components=m,
                                   n_features=n,
                                   n_nonzero_coefs=L,
                                   random_state=seed)

    
    # Make the signals a bit noisy and compute an error tolerance
    s_noisy = s + 0.05 * np.random.randn(s.shape[0], s.shape[1])
    tolerance = 0.05 * np.sqrt(n)
    
    class_sol = []
    quant_sol = []
    class_k = []
    quant_k = []
    class_residuals = []
    quant_residuals = []
    class_residuals_full = []
    quant_residuals_full = []


    
    for i in range(n_times):
        x1, kc, resc = MP(s_noisy[:,i], D, tolerance, L_thresh, info='all')
        xq, kq, resq = MP(s_noisy[:,i], D, tolerance, L_thresh, quantum_error=q_err, version=version, info='all')
            
        class_sol.append(x1)
        quant_sol.append(xq)

        class_k.append(kc)
        quant_k.append(kq)

        class_residuals.append(np.linalg.norm(s[:,i]-D@x1))
        quant_residuals.append(np.linalg.norm(s[:,i]-D@xq))
        
        class_residuals_full.append(resc)
        quant_residuals_full.append(resq)

    class_perc = np.sum(class_residuals <= tolerance)/n_times
    quant_perc = np.sum(quant_residuals <= tolerance)/n_times

    if output == "all":
        return class_sol, quant_sol, class_k, quant_k, class_residuals, quant_residuals, class_residuals_full, quant_residuals_full, class_perc, quant_perc
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