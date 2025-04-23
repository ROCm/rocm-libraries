# rocAL Python Binding

rocAL Python Binding allows you to call functions and pass data from Python to rocAL C/C++ libraries,
letting you take advantage of the rocAL functionality in both languages.

rocal_pybind.so is a wrapper library that bridge python and C/C++, so that a rocAL functionality
written primarily in C/C++ language can be used effectively in Python.

## Prerequisites

* [rocAL C/C++ Library](../rocAL/README.md#prerequisites)
* CMake Version 3.10 or higher
* Python 3
* PIP3 - `sudo apt install python3-pip`
* [dlpack](https://github.com/dmlc/dlpack)

## Install

rocAL_pybind installs during [rocAL build](https://github.com/ROCm/rocAL#build-instructions)

### Prerequisites

* Install pip packages

````shell
pip3 install numpy opencv-python torch pillow
````

### Run Test Scripts

* Test scripts and instructions to run them can be found [here](../tests/python_api/)
* Examples using python APIs can be found [here](../docs/examples/)
