#!/usr/bin/env python
# coding: utf-8

#### Uses quantumMatchPurs to collect data and store them in a file: sintetici_dump.json

from quantumMatchPurs import MP, qMP_experiment
from scipy import stats
import numpy as np
import json

## Storing the results
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

def classical_runtime(k, n, m):
    return k*n*m

def quantum_runtime(k, n, m, xi, delta):
    return k*n*np.log(n) + k*np.sqrt(m)*(1/xi)*np.log(3*k*m/delta)*np.log(n*m)

delta = 0.01
xi = 0.01
n_times = 100

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
ns = range(50, 2050, 50)
i=0
for n in ns:
    m = int(n * 2)
    L = int(n/5)
    class_sol, quant_sol, class_k, quant_k, class_residuals, quant_residuals, class_residuals_full, quant_residuals_full, class_perc, quant_perc = qMP_experiment(n, m, L, xi, "both", n_times, "all", 1234+i)
    class_runtime = [classical_runtime(k, n, m) for k in class_k]
    quant_runtime = [quantum_runtime(k, n, m, xi, delta) for k in quant_k]
    class_sol_list.append(class_sol)
    quant_sol_list.append(quant_sol)
    kcs.append(class_k)
    kqs.append(quant_k)
    resc.append(class_residuals_full)
    resq.append(quant_residuals_full)
    run_t.append((np.mean(class_runtime), np.std(class_runtime)))
    qrun_t.append((np.mean(quant_runtime), np.std(quant_runtime)))
    i += 1
    print(n)

dump_dict = {}
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

dumped = json.dumps(dump_dict, cls=NumpyEncoder)
with open('data_dump.json', 'a') as f:
    f.write(dumped + '\n')