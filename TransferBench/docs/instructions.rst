-------------
Requirements
-------------

1. ROCm stack installed on the system (HIP runtime)
2. libnuma installed on system

-------------
Building
-------------

To build TransferBench using Makefile:
::

    $ make

To build TransferBench using cmake:
::

    $ mkdir build
    $ cd build
    $ CXX=/opt/rocm/bin/hipcc cmake ..
    $ make

If ROCm is installed in a folder other than `/opt/rocm/`, set ROCM_PATH appropriately

--------------------------
NVIDIA platform support
--------------------------

TransferBench may also be built to run on NVIDIA platforms via HIP, but requires a HIP-compatible CUDA version installed (e.g. CUDA 11.5)

To build:
::
    
   CUDA_PATH=<path_to_CUDA> HIP_PLATFORM=nvidia make`
