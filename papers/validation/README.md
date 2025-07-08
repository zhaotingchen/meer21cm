# Validation of the `meer21cm` pipeline
This page holds the validation pipeline for the `meer21cm` package.
Each validation test is numbered by the two digits and described below.
For each test, the `func_xx.py` holds the script for the validation calculations, and `run_xx.ipynb` holds the script to running and visualising the validations.

## 00: Mock simulation test
In a given cubic box, a mock tracer density field and a discrete tracer catalogue are generated. They are then used to calculate the simulated power spectrum and compared against the input.
