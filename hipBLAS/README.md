# hipBLAS

hipBLAS is a Basic Linear Algebra Subprograms (BLAS) marshalling library with multiple supported
backends. It sits between your application and a 'worker' BLAS library, where it marshals inputs to the
backend library and marshals results to your application.  hipBLAS exports an interface that doesn't
require the client to change, regardless of the chosen backend. Currently, hipBLAS supports rocBLAS
and cuBLAS backends.

To use hipBLAS, you must first install rocBLAS, rocSPARSE, and rocSOLVER or cuBLAS.

## Documentation

> [!NOTE]
> The published hipBLAS documentation is available at [hipBLAS](https://rocm.docs.amd.com/projects/hipBLAS/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the hipBLAS/docs folder of this repository. As with all ROCm projects, the documentation is open source. For more information, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).


To build our documentation locally, use the following code:

```bash
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

Alternatively, build with CMake:

```bash
cmake -DBUILD_DOCS=ON ...
```


## Build and install

1. Download the hipBLAS source code (clone this repository):

    ```bash
        git clone https://github.com/ROCmSoftwarePlatform/hipBLAS.git
    ```

    ```note
        hipBLAS requires specific versions of rocBLAS and rocSOLVER. Refer to
        [CMakeLists.txt](https://github.com/ROCmSoftwarePlatform/hipBLAS/blob/develop/library/CMakeLists.txt)
        for details.
    ```

2. Build hipBLAS and install it into `/opt/rocm/hipblas`:

    ```bash
        cd hipblas
        ./install.sh -i
    ```

## Interface examples

The hipBLAS interface is compatible with rocBLAS and cuBLAS-v2 APIs. Porting a CUDA application
that originally calls the cuBLAS API to an application that calls the hipBLAS API is relatively
straightforward. For example, the hipBLAS SGEMV interface is:

### GEMV API

```c
hipblasStatus_t
hipblasSgemv( hipblasHandle_t handle,
              hipblasOperation_t trans,
              int m, int n, const float *alpha,
              const float *A, int lda,
              const float *x, int incx, const float *beta,
              float *y, int incy );
```

### Batched and strided GEMM API

hipBLAS GEMM can process matrices in batches with regular strides by using the strided-batched
version of the API:

```c
hipblasStatus_t
hipblasSgemmStridedBatched( hipblasHandle_t handle,
              hipblasOperation_t transa, hipblasOperation_t transb,
              int m, int n, int k, const float *alpha,
              const float *A, int lda, long long bsa,
              const float *B, int ldb, long long bsb, const float *beta,
              float *C, int ldc, long long bsc,
              int batchCount);
```

hipBLAS assumes matrix A and vectors x, y are allocated in GPU memory space filled with data. You
are responsible for copying data to and from the host and device memory.
