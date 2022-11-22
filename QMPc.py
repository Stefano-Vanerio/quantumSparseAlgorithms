import numpy as np


def continuing_criterium_3(x, s, c, tolerance, L_thresh):
    # if L == 0 use only ||r||_2 to consider the next iteration
    if L_thresh != 0:
        sparsity_check = np.linalg.norm(x, ord=0) <= L_thresh
    else:
        sparsity_check = True
    return sparsity_check and (s-c > tolerance)


def MP_3(s, D, tolerance, error_type="", L_thresh=0, info='', error=0):
    N, M = D.shape
    # Initialize the solution vector to zeros
    x = np.zeros(M)
    phi = np.zeros(N)

    # Initialize the residual
    # i.e., components not yet captured by the sparse signal
    r = s
    residuals = [np.linalg.norm(r)]


    # init the number of iterations k
    num_iterations = 1

    # initialization of parameters for exit condition
    error_tolerance = tolerance**2
    s_norm_squared = np.linalg.norm(s)**2
    c = 0

    # ERRORS
    # error for the inner product routine
    ipe_error = error/4

    # error for the value of phi
    phi_error = error/4

    # vector of inner product just between signal and atoms
    z_1 = np.zeros(M)

    for j in range(M):
        z_1[j] = np.dot(D[:, j], s)

    if ipe_error != 0:
        if error_type == "U":
            z_1 = z_1 + np.random.uniform(-ipe_error*2, ipe_error*2, size=M)
        else:
            eps = np.random.normal(0, ipe_error*2/3, size=M)
            while max(eps) > (ipe_error*2) or min(eps) < (-2*ipe_error):
                eps = np.random.normal(0, ipe_error*2/3, size=M)
            z_1 = z_1 + eps

    while continuing_criterium_3(x, s_norm_squared, c, error_tolerance, L_thresh):
        z = np.zeros(M)
        z_2 = np.zeros(M)

        # Compute all the inner products
        for j in range(M):
            z_2[j] = np.dot(D[:, j], phi)

        if ipe_error != 0:
            if error_type == "U":
                z_2 = z_2 + np.random.uniform(-ipe_error, ipe_error, size=M)
            else:
                eps = np.random.normal(0, ipe_error / 3, size=M)
                while max(eps) > ipe_error or min(eps) < -ipe_error:
                    eps = np.random.normal(0, ipe_error / 3, size=M)
                z_2 = z_2 + eps

        for j in range(M):
            z[j] = z_1[j] - z_2[j]

        # to find the minimum e(j) we just need to search for the maximum |dj^T r|
        jStar = np.argmax(np.abs(z))

        # update the solution
        x[jStar] = x[jStar] + z[jStar]

        # Approximated solution
        phi = D @ x

        # error added due to application of block encoding
        if phi_error != 0:
            if error_type == "U":
                vector = np.random.uniform(-1, 1, size=phi.shape[0])
                phi = phi + phi_error * vector / np.linalg.norm(vector)
            else:
                eps = np.random.normal(0, 1 / 3, size=phi.shape[0])
                while max(eps) > 1 or min(eps) < -1:
                    eps = np.random.normal(0, 1 / 3, size=phi.shape[0])
                phi = phi + phi_error * eps / np.linalg.norm(eps)

        # update the residual - not used for exit condition, just check for correct execution
        r = r - z[jStar] * D[:, jStar]
        residuals.append(np.linalg.norm(r))

        # Update the energy for stop condition, the error with the stop condition changes in a different way
        c = c + z[jStar]**2

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
