import numpy as np
import statsmodels.api as sm


def continuing_criterium_omp(t, r, tolerance, L_thresh):
    if L_thresh != 0:
        sparsity_check = t <= L_thresh
    else:
        sparsity_check = True
    return sparsity_check and r > tolerance


def OMP(s, D, tolerance, error_type, L_thresh=0,  info='', error=0):
    cond_A = list()
    mu_A = list()
    N, M = D.shape
    jStar = list()
    P = D

    # Initialize the solution vector to zeros
    x = np.zeros(M)
    phi = np.zeros(N)

    # Initialize the residual
    # i.e., components not yet captured by the sparse signal
    r = np.linalg.norm(s)
    residuals = [r]

    # init the number of iterations k
    num_iterations = 1

    # ERRORS
    # the same for both the inner products
    ipe_error = error/4

    # error for the exit condition computation
    dis_error = error/4

    # vector of error for the vector phi
    phi_error = error/4

    while continuing_criterium_omp(num_iterations-1, r, tolerance, L_thresh):
        z = np.zeros(P.shape[1])
        z_1 = np.zeros(P.shape[1])
        z_2 = np.zeros(P.shape[1])

        # Compute all the inner products
        for j in range(P.shape[1]):
            z_1[j] = np.dot(P[:, j], s)

        if ipe_error != 0:
            if error_type == "U":
                z_1 = z_1 + np.random.uniform(-ipe_error*2, ipe_error*2, size=P.shape[1])
            else:
                eps = np.random.normal(0, ipe_error*2/3, size=P.shape[1])
                while max(eps) > ipe_error or min(eps) < -ipe_error:
                    eps = np.random.normal(0, ipe_error*2/3, size=P.shape[1])
                z_1 = z_1 + eps

        if num_iterations == 1:
            z = z_1
        else:
            for j in range(P.shape[1]):
                z_2[j] = np.dot(P[:, j], phi.reshape(N, 1))

            if ipe_error != 0:
                if error_type == "U":
                    z_2 = z_2 + np.random.uniform(-ipe_error, ipe_error, size=P.shape[1])
                else:
                    eps = np.random.normal(0, ipe_error / 3, size=P.shape[1])
                    while max(eps) > ipe_error or min(eps) < -ipe_error:
                        eps = np.random.normal(0, ipe_error / 3, size=P.shape[1])
                    z_2 = z_2 + eps

            for j in range(P.shape[1]):
                z[j] = z_1[j] - z_2[j]

        # to find the minimum e(j) we just need to search for the maximum |dj^T r|
        jStar.append(np.argmax(np.abs(z)))
        jS = int(jStar[num_iterations-1])

        if num_iterations == 1:
            A = np.transpose(np.matrix(P[:, jS]))
        else:
            A = np.c_[A, P[:, jS]]

        P = np.delete(P, jS, 1)

        # update the solution
        x = sm.OLS(s, A)
        x = x.fit()
        x = np.array(x.params)

        # Approximated solution
        phi = A @ x

        if phi_error != 0:
            if error_type == "U":
                vector = np.random.uniform(-1, 1, size=phi.shape[0])
                phi = phi + phi_error * vector / np.linalg.norm(vector)
            else:
                eps = np.random.normal(0, 1 / 3, size=phi.shape[0])
                while max(eps) > 1 or min(eps) < -1:
                    eps = np.random.normal(0, 1 / 3, size=phi.shape[0])
                phi = phi + phi_error * eps / np.linalg.norm(eps)

        r = np.linalg.norm(s - phi)

        # update the residual
        if dis_error != 0:
            if error_type == "U":
                r = r + np.random.uniform(-dis_error, dis_error, size=1)
            else:
                eps = np.random.normal(0, dis_error / 3, size=1)
                while max(eps) > dis_error or min(eps) < -dis_error:
                    eps = np.random.normal(0, dis_error / 3, size=1)
                r = r + eps

        residuals.append(r)

        # update the number of iterations so far
        num_iterations += 1
        if info == 'param':
            cond_A.append(np.linalg.cond(A))
            mu_A.append(linear_search(A))

    x_fin = np.zeros(M)
    el = list()
    count = 0

    for a in A.transpose():
        if a in D.transpose():
            el.append((np.where((D.transpose() == a).all(axis=1))[0])[0])   #np.where((source == target).all(axis=1))[0]

    for i in range(0, M, 1):
        if i in el:
            x_fin[i] = x[count]
            count = count+1
        else:
            x_fin[i] = 0

    if info == 'it':
        return x_fin, num_iterations
    if info == 'residual':
        return x_fin, residuals
    if info == 'param':
        return num_iterations, mu_A, cond_A
    if info == 'all':
        return x_fin, num_iterations, residuals, x, A
    if info == 'full':
        return x_fin, num_iterations, mu_A, cond_A
    else:
        return x_fin


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