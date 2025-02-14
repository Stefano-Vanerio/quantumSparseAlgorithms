# Quantum matching pursuit

This repo contains the core code of the experiments of the quantum matching pursuit algorithm and quantum orthogonal matching pursuit.

`create_data.py` is a script that can be used to collect the data of multiple instances of the experiment, here is possible to modify the parameters for run.
- **delta** is the probability of failure
- **xi** is the starting error added
- **step** is the incremental step in the error variable
- **iterations** is the number of step of error increasing 
- **n_times** is the number of sample per iteration 
- **ns** number of signal components 
- **kind** algorithm chosen, can be *QOMP*, *QMPa*, *QMPb*, *QMPc*
- **error_type** type of error *""* for Gaussian or *"U"* for uniform

`create_data.py` is a script that can be used to collect the data of multiple instances of the experiment.
`experiment.py` contains the code that starts each different basic algorithm.

Using the data generated from this code it is possible to reproduce the analysis of our work.
`QOMP.py`cointains the code for Quantum Orthogonal Matching Pursuit

`QMPa.py`cointains the code for Quantum Matching Pursuit from Bellante's work

`QMPb.py`cointains the code for Quantum Matching Pursuit from my work with single error version

`QMPc.py`cointains the code for Quantum Matching Pursuit from my work with double error version

Using the data generated from this code it is possible to reproduce the analysis of our work.
The notebooks `Error Variation.ipynb` and `Utility.ipynb` where used to obtain some graphs to visualize the results.

In the folder 'N-grams' it is present the code used to test the QOMP algorithm on samples described by N-grams. The dataset had been obtained using the code of this [repository](https://github.com/Arun152k/Malware-Detection-using-N-Gram-Frequency). 
