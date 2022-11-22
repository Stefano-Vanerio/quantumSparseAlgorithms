#!/usr/bin/env python
# coding: utf-8

#### Uses quantumMatchPurs to collect data and store them in a file: dump.json
import os

from experiment import qMP_experiment
import numpy as np
import json

# Storing the results
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def classical_runtime(k, n, m): ###Computes the classical runtime
    return k*n*m

# Parameters to modify
delta = 0.01        ###Probability of failure
xi = 0.005           ###Epsilon
n_times = 100       ###Number of sample to generate for each iteration, to have a bunch of signals with almost same characteristics
step = 0.005
ns = range(50, 1050, 50)
kind = 'QOMP'
error_type = "U"
iterations = 10

run_t = []              ###Median and standard deviation of classical runtime
qrun_t = []             ###Median and standard deviation of quantum runtime
expected_class = []     ###These are explained below
expected_quant = []     ###...
class_sol_list = []
quant_sol_list = []
kcs = []
kqs = []
resc = []
resq = []
i = 0


for j in range(0, iterations, 1):
    for n in ns:
        m = int(n * 2)              ###Number of elements in the dictionary (100/200/300/...)
        L = int(n/5)                ###Sparsity required of the analysed signals
        class_sol, quant_sol, class_k, quant_k, class_residuals, quant_residuals, class_residuals_full, quant_residuals_full, class_perc, quant_perc, quant_runtime = qMP_experiment(n, m, L, xi, kind, "both", n_times, error_type, "all", delta, 1234+i)
        ###Class/Quant sol: matrix of solution vectors
        ###Class/Quant k: number of iterations
        ###Class/Quant residuals: residuals when algorithm stopped
        ###Class/Quant residuals full: residuals of each iteration of the algorithm for each signal
        ###Class/Quant perc: percentage of how many times the algorithm had stopped for threshold on sparsity and not on number of iterations
        class_runtime = [classical_runtime(k, n, m) for k in class_k]               ###Here computes the vector of runtime for each classical computation (only k changes)
        class_sol_list.append(class_sol)                    ###List of all classical solutions for each "n" considered during for
        quant_sol_list.append(quant_sol)                    ###List of all quantum solutions for each "n" considered during for
        kcs.append(class_k)                                 ###List of number of iterations classically
        kqs.append(quant_k)                                 ###List of number of iterations quantumly
        resc.append(class_residuals_full)                   ###List of all the residuals of all the iterations classical
        resq.append(quant_residuals_full)                   ###List of all the residuals of all the iterations quantum
        run_t.append((np.mean(class_runtime), np.std(class_runtime)))               ###Median and standard deviation of classical runtime
        qrun_t.append((np.mean(quant_runtime), np.std(quant_runtime)))              ###Median and standard deviation of quantum runtime
        i += 1                                              ###Changes the seed for the generation of different signals
        print(n, m, L)                                      ###Prints batch of iteration

    dump_dict = {}                                          ###Dumps everything
    dump_dict['class_sol'] = class_sol_list
    dump_dict['quant_sol'] = quant_sol_list
    dump_dict['class_k'] = kcs
    dump_dict['quant_k'] = kqs
    dump_dict['class_residuals'] = class_residuals
    dump_dict['quant_residuals'] = quant_residuals
    dump_dict['class_residuals_full'] = resc
    dump_dict['quant_residuals_full'] = resq
    dump_dict['class_runtime'] = run_t
    dump_dict['quant_runtime'] = qrun_t
    dump_dict['class_perc'] = class_perc
    dump_dict['quant_perc'] = quant_perc

    dumped = json.dumps(dump_dict, cls=NumpyEncoder)        ###Puts in .json
    save_path = 'Results'

    name = 'data_dump_' + "{:.3f}".format(xi) + '_' + kind + error_type + '_R.json'
    completeName = os.path.join(save_path, name)
    with open(completeName, 'w') as f:
        f.write(dumped + '\n')
    xi = xi + step

    # Refresh variables for multiple iterations
    run_t = []
    qrun_t = []
    expected_class = []
    expected_quant = []
    class_sol_list = []
    quant_sol_list = []
    kcs = []
    kqs = []
    resc = []
    resq = []
    i = 0
