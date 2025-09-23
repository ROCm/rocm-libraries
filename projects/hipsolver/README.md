# hipSOLVER

hipSOLVER is a LAPACK marshalling library, with multiple supported backends.  It sits between the application and a 'worker' LAPACK library, marshalling inputs into the backend library and marshalling results back to the application.  hipSOLVER exports an interface that does not require the client to change, regardless of the chosen backend.  Currently, hipSOLVER supports [rocSOLVER](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsolver) and [cuSOLVER](https://developer.nvidia.com/cusolver) as backends.

## Documentation

> [!NOTE]
> The published hipSOLVER documentation is available at [hipSOLVER](https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the hipSOLVER/docs folder of this repository. As with all ROCm projects, the documentation is open source. For more information, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).


### How to build documentation

Run the steps below to build documentation locally.

```shell
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Quickstart Build

To download the hipSOLVER source code, use either a sparse checkout or a full clone of the rocm-libraries repository.

To limit your local checkout to only the hipSOLVER project, configure ``sparse-checkout`` before cloning.
This uses the Git partial clone feature (``--filter=blob:none``) to reduce how much data is downloaded.
Use the following commands for a sparse checkout:

```bash
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/hipsolver # add projects/rocsolver projects/rocblas projects/rocsparse to include dependencies
git checkout develop # or use the branch you want to work with
```

To clone the entire rocm-libraries repository, use the following commands. This process takes more time,
but is recommended if you want to work with a large number of libraries.

```bash
git clone https://github.com/ROCm/rocm-libraries.git
```

hipSOLVER requires either cuSOLVER or rocSOLVER + SuiteSparse to be installed on the system. Once these are installed, the following commands will build hipSOLVER and install to `/opt/rocm`:

```bash
cd rocm-libraries/projects/hipsolver
./install.sh -i
```

Once installed, hipSOLVER can be used just like any other library with a C API. The header file will need to be included in the user code, and the hipSOLVER library will become a link-time and run-time dependency for the user application.

For more information on building and installing hipSOLVER, see the [hipSOLVER install guide](https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/installation/install.html).

### Building hipSOLVER for the NVIDIA CUDA platform

For the purpose of porting an application from NVIDIA CUDA to ROCm, it's possible to build hipSOLVER to run on NVIDIA hardware.
To build the library on a NVIDIA CUDA-enabled machine, with CUDA cuSOLVER as the backend, run the following install command.
The NVIDIA CUDA backend has a dependency on cuSOLVER. Consult the [hipSOLVER install guide](https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/installation/install.html) for more information.

    ./install.sh --cuda

> [!NOTE]
Running hipSOLVER on NVIDIA hardware is for development purposes only.

## Using the hipSOLVER Interface

The hipSOLVER interface is compatible with the rocSOLVER and cuSOLVER-v11 APIs. Porting a CUDA application that originally calls the cuSOLVER API to an application calling the hipSOLVER API should be fairly straightforward (see [porting a cuSOLVER application to hipSOLVER](https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/howto/usage.html#porting-cusolver-applications-to-hipsolver)). For example, the hipSOLVER SGEQRF interface is

```c
hipsolverStatus_t
hipsolverSgeqrf_bufferSize(hipsolverHandle_t handle,
                           int m,
                           int n,
                           float* A,
                           int lda,
                           int* lwork);
```

```c
hipsolverStatus_t
hipsolverSgeqrf(hipsolverHandle_t handle,
                int               m,
                int               n,
                float*            A,
                int               lda,
                float*            tau,
                float*            work,
                int               lwork,
                int*              devInfo);
```

## Supported Functionality

For a complete listing of all supported functions, see the [hipSOLVER user guide](https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/howto/usage.html) and/or [API documentation](https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/reference/index.html).
