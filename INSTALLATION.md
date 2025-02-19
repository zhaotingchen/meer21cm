# Installation

The installation is tested locally on arm64 MacOS system as well as x86_64 Linux system, on python 3.9 and 3.10. Here is some hopefully useful guide to installing the pacakge.

Most of the extra efforts before installing the package itself is to install `pfft-python` and `pmesh` which `meer21cm` depends on.

If you are on ilifu jump straight to [`Installing on ilifu`](#ilifu).

## Installing dependencies
A clean `conda` environment on python 3.9 or 3.10 is highly recommended.

### MacOS
In the conda environment, do
```
conda install -c anaconda 'cython<3.0'
pip install "numpy<2.0"
brew install openmpi
brew install hdf5
brew install c-blosc
export HDF5_DIR=/opt/homebrew/opt/hdf5
export BLOSC_DIR=/opt/homebrew/opt/c-blosc
pip install "mpi4py<4.0"
git clone https://github.com/zhaotingchen/MP-sort.git
cd MP-sort
pip install -e .
cd ..
pip install git+https://github.com/rainwoodman/pfft-python.git
```
This should install some tricky yet required dependencies for `meer21cm`.

### Linux PC
For Linux PC, depending on your system you should be able to install `openmpi` through `conda` or `apt`. For `ubuntu`, the most secure way in our tests seems to be

```
conda install -c anaconda 'cython<3.0'
pip install "numpy<2.0"
sudo apt install libopenmpi-dev
rm /path/to/your/conda/env/compiler_compat/ld
conda install -c conda-forge mpi4py openmpi
pip install git+https://github.com/rainwoodman/pfft-python.git
```
Note that you need to replace `/path/to/your/conda/env` with the actual path to your conda environment.
You can also try skipping the `rm` step and see if it works.

### HPC
If you are on a cluster, most likely you already have some MPI implementation available. You can check its availability by
```
module avail
```
You can first find that anaconda module to use. Typically this will be called `anaconda` or `anaconda3`.
Then do
```
module load anaconda3
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

Sometimes you already have a version of `cython` installed that causes a compilation error. In that case you can override the `cython` in your conda environment by installing it again
```
conda install "cython<3.0"
```

## Install `meer21cm`
Finally, clone the repo for `meer21cm`
```
git clone git@github.com:zhaotingchen/meer21cm.git
```

Go to the directory
```
cd meer21cm
```

And run
```
pip install -e .
```

If you want to run tests, instead of the installation above do
```
pip install -e ".[test]"
```

and run
```
pytest tests/
```
to see if the installation is successful.

If you want to develop `meer21cm`, the installation step should be
```
pip install -e ".[full]"
```

Similarly, if there is an MPI compilation error, you can try fixing it by specifying

```
env MPICC=path/to/mpicc pip install -e ".[test]"
```

Note that development install `-e` is needed, as this package is in early stage and will not have a stable version before the official release.

## Installing on ilifu
<a name="ilifu"></a>
If you are on ilifu, the installation has been tested so you can follow the exact steps listed here.

```
git clone git@github.com:zhaotingchen/meer21cm.git
cd meer21cm
module load anaconda3
conda create -n meer21cm python=3.10
conda activate meer21cm
conda install "cython<2"
module load openmpi
pip install mpi4py
pip install "numpy<2"
pip install -e ".[full]"
```

Optionally, to use the environment as a jupyter kernel run
```
pip install ipykernel
python -m ipykernel install --name "meer21cm" --user
```
