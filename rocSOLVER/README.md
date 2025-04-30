# rocSOLVER

rocSOLVER is a work-in-progress implementation of a subset of [LAPACK][1]
functionality on the [ROCm platform][2].

## Documentation

> [!NOTE]
> The published rocSOLVER documentation is available at [rocSOLVER](https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the rocSOLVER/docs folder of this repository. As with all ROCm projects, the documentation is open source. For more information, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

### How to build documentation

Please follow the instructions below to build the documentation.

```
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Building rocSOLVER

To download the rocSOLVER source code, clone this repository with the command:

    git clone https://github.com/ROCmSoftwarePlatform/rocSOLVER.git

rocSOLVER requires rocBLAS as a companion GPU BLAS implementation. For
more information about rocBLAS and how to install it, see the
[rocBLAS documentation][4].

After a standard installation of rocBLAS, the following commands will build
rocSOLVER and install to `/opt/rocm`:

    cd rocSOLVER
    ./install.sh -i

Once installed, rocSOLVER can be used just like any other library with a C API.
The header file will need to be included in the user code, and both the rocBLAS
and rocSOLVER shared libraries will become link-time and run-time dependencies
for the user application.

If you are a developer contributing to rocSOLVER, you may wish to run
`./scripts/install-hooks` to install the git hooks for autoformatting.
You may also want to take a look at the [contributing guidelines][7]

## Using rocSOLVER

The following code snippet shows how to compute the QR factorization of a
general m-by-n real matrix in double precision using rocSOLVER. A longer
version of this example is provided by `example_basic.cpp` in the
[samples directory][5]. For a description of the `rocsolver_dgeqrf`
function, see the [rocSOLVER API documentation][6].

```cpp
/////////////////////////////
// example.cpp source code //
/////////////////////////////

#include <algorithm> // for std::min
#include <stddef.h>  // for size_t
#include <vector>
#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations

int main() {
  rocblas_int M;
  rocblas_int N;
  rocblas_int lda;

  // here is where you would initialize M, N and lda with desired values

  rocblas_handle handle;
  rocblas_create_handle(&handle);

  size_t size_A = size_t(lda) * N;          // the size of the array for the matrix
  size_t size_piv = size_t(std::min(M, N)); // the size of array for the Householder scalars

  std::vector<double> hA(size_A);      // creates array for matrix in CPU
  std::vector<double> hIpiv(size_piv); // creates array for householder scalars in CPU

  double *dA, *dIpiv;
  hipMalloc(&dA, sizeof(double)*size_A);      // allocates memory for matrix in GPU
  hipMalloc(&dIpiv, sizeof(double)*size_piv); // allocates memory for scalars in GPU

  // here is where you would initialize matrix A (array hA) with input data
  // note: matrices must be stored in column major format,
  //       i.e. entry (i,j) should be accessed by hA[i + j*lda]

  // copy data to GPU
  hipMemcpy(dA, hA.data(), sizeof(double)*size_A, hipMemcpyHostToDevice);
  // compute the QR factorization on the GPU
  rocsolver_dgeqrf(handle, M, N, dA, lda, dIpiv);
  // copy the results back to CPU
  hipMemcpy(hA.data(), dA, sizeof(double)*size_A, hipMemcpyDeviceToHost);
  hipMemcpy(hIpiv.data(), dIpiv, sizeof(double)*size_piv, hipMemcpyDeviceToHost);

  // the results are now in hA and hIpiv, so you can use them here

  hipFree(dA);                        // de-allocate GPU memory
  hipFree(dIpiv);
  rocblas_destroy_handle(handle);     // destroy handle
}
```

The exact command used to compile the example above may vary depending on the
system environment, but here is a typical example:

    /opt/rocm/bin/hipcc -I/opt/rocm/include -c example.cpp
    /opt/rocm/bin/hipcc -o example -L/opt/rocm/lib -lrocsolver -lrocblas example.o


[1]: https://www.netlib.org/lapack/
[2]: https://rocm.docs.amd.com/
[3]: https://rocm.docs.amd.com/projects/rocSOLVER/
[4]: https://rocm.docs.amd.com/projects/rocBLAS/
[5]: clients/samples/
[6]: https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/api/lapack.html#rocsolver-type-geqrf
[7]: CONTRIBUTING.md
