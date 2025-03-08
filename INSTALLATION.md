# Installation

The installation is tested on arm64 MacOS system as well as x86_64 Linux system, on python 3.9 and 3.10.

If you are on ilifu jump straight to [`Installing on ilifu`](#ilifu).

## 1. Installing dependencies
A clean `conda` environment on python 3.9 or 3.10 is highly recommended.
You can create a new conda environment by
```
conda create -n meer21cm python=3.10
conda activate meer21cm
```

If you are not on your own machine, check below for how to activate the conda environment on HPC.

### MacOS
For MacOS, you can install hdf5 through brew.
```
brew install hdf5
brew install c-blosc
export HDF5_DIR=/opt/homebrew/opt/hdf5
export BLOSC_DIR=/opt/homebrew/opt/c-blosc
```

### Linux PC
For Linux PC, most of the time you can install the h5df dependency through `conda`.

```
conda install hdf5
```

### HPC
If you are on a cluster, most likely you already have some conda module available. You can check its availability by
```
module avail
```
Typically this will be called `conda`, `anaconda` or `anaconda3`.
Then do
```
module load anaconda3
```

If the HPC does not have conda, you can also install your own miniconda (check the [official guide](https://docs.conda.io/en/latest/miniconda.html) for more details). One thing to note is that you need to change the path to which conda is installed when using the installation prompt as you typically do not have access to the default system path.


Then create a new conda environment and activate it. Install the dependencies
```
conda install hdf5
```

## 2. Install `meer21cm`
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
to see if the current installation passes all the tests.

If you want to develop `meer21cm`, the installation step should be
```
pip install -e ".[full]"
```

Note that development install `-e` is needed, as this package is in early stage and will not have a stable version before the official release.

## 3. Check if installation is successful
In the conda environment, do
```
python -c "import meer21cm; print(meer21cm.__file__)"
```
If you see the output, then the installation is successful.


## A. Installing on ilifu
<a name="ilifu"></a>
If you are on ilifu, the installation has been tested so you can follow the exact steps listed here.

```
git clone git@github.com:zhaotingchen/meer21cm.git
module load anaconda3
conda create -n meer21cm python=3.10
conda activate meer21cm
conda install hdf5
pip install -e ".[full]"
```

Optionally, to use the environment as a jupyter kernel run
```
pip install ipykernel
python -m ipykernel install --name "meer21cm" --user
```
