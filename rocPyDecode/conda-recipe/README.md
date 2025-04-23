# rocPyDecode conda package

## Prerequisites
* conda or mini-conda installed
* rocPyDecode wheel

If you do not have conda or mini-conda installed you can use following steps to install it:

```bash
    # if inside docker you might not have wget
    sudo apt install wget -y 
    # download latest minimal conda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    # to automate use '-b'
    bash Miniconda3-latest-Linux-x86_64.sh -b
    source ~/miniconda3/etc/profile.d/conda.sh
    # to speed up installations
    conda config --add channels conda-forge
    conda config --set channel_priority strict
    # update to ensure latest
    conda update conda -y
    # finally the build pkg
    conda install conda-build -y
```
To create the required rocPyDecode wheel:
```bash
    # change directory to rocPyDecode main folder, then execute this command
    # if using docker (not bare-metal) do not use 'sudo'
    sudo python3 setup.py bdist_wheel   # this will create the .whl file under 'dist' subfolder
```

## Building conda package

Change directory to rocPyDecode subfolder 'conda-recipe', then execute the conda build command:

```bash
    conda build .   # assumes the meta.yaml file on same folder
```
## Location of rocPyDecode conda package

After successful build from the previous step the generated conda package will be on the following path:

```bash
    /root/miniconda3/conda-bld/linux-64/rocpydecode-0.2.0-py310_0.tar.bz2
 ```

## Installing rocPyDecode conda package

To install the generated package use the following command:

```bash
    # default
    conda install --use-local rocpydecode

    # or target the exact bz2 package
    conda install --use-local /root/miniconda3/conda-bld/linux-64/rocpydecode-0.2.0-py310_0.tar.bz2
```

## Installing rocPyDecode package on conda virtual environment

To install rocPyDecode conda package on virtual environment use the following commands:
```bash
    # create conda virtual env
    conda create -n test_env_10 python=3.10  # Specify Python version
    conda activate test_env_10  # activate the env

    # when inside the env
    conda install --use-local rocpydecode
    pip3 install numpy # needed to test sample

    # test using provided samples (specify the location of the .py and .mp4 files)
    python3 videodecode.py -i AMD_driving_virtual_20-H265.mp4

    # when env not in use, deactivate it
    conda deactivate
```




