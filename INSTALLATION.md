# Installation

Unfortunately, `meer21cm` uses [`pmesh`](https://github.com/rainwoodman/pmesh) for dependencies. This has caused severe difficulty in testing and enabling a simple `pip` installation, as `pmesh` is not actively maintained and needs to be installed in a painstaking way. The installation is tested locally on arm64 MacOS system as well as x86_64 Linux system, on python 3.9 and 3.10. Here is some hopefully useful guide to installing `pfft-python` which `pmesh` uses as dependency.

If you have a working installation of `mpi4py` on your system you may skip the first part and go straight to [`pfft-python`](#pfft)

## Installing `mpi4py`
A clean `conda` environment is highly recommended.

First, install an MPI implementation. For MacOS (no need to use Rosetta as well), simply
```
brew install mpi4py
```

Although `mpi4py` should already be installed, for linking reasons we find that it is better if you do
```
pip install mpi4py
```
again.

For Linux PC, depending on your system you should be able to install `openmpi` or `mpich`. For `ubuntu`, the most secure way in our tests seems to be

```
sudo apt install libopenmpi-dev
conda install conda-forge::openmpi
conda install openmpi-mpicc
pip install mpi4py
```

Note that technically only `openmpi-mpicc` is really needed. Try only installing `openmpi-mpicc` and then `mpi4py` first. There is always a possibility of compiling issues, so the extra steps are just trying to avoid problems as much as possible.

If you are on a cluster, most likely you already have some MPI implementation available. You can check its availability by
```
module avail
```

Find the MPI you want to use and then do
```
module load MPI_MODULE
```
and replace `MPI_MODULE` above with whatever you find in `module avail`, for example `module load openmpi`. Then simply `pip install mpi4py`.

If `pip install mpi4py` still failed with an MPI issue, it may be fixed by specifying `mpicc` path. Try to find your `mpicc` path by entering
```
which mpicc
```
in your terminal and do
```
env MPICC=path/to/mpicc pip install mpi4py
```
instead.

## Manual installation of `pfft-python` dependencies
<a name="pfft"></a>
Sadly this can not just a `pip install`. First, make sure `cython` is installed with an older version instead of "cython3"
```
pip install 'cython<3.0'
```
Then, install numpy manually
```
pip install "numpy<2.0"
```

This prepares the environment so `setup.py` for `pfft-python` and `pmesh` will not fail.

## Install `meer21cm`
Finally, clone the repo for `meer21cm`
```
git clone https://github.com/zhaotingchen/meer21cm
```

And run
```
pip install .
```

If you want to run tests, instead of the installation above do
```
pip install ".[test]"
```

and run
```
pytest tests/
```
do see if the installation is successful.

If you want to develop `meer21cm`, the installation step should be
```
pip install -e ".[full]"
```

Similarly, if there is an MPI compilation error, you can try fixing it by specifying

```
env MPICC=path/to/mpicc pip install ".[test]"
```
