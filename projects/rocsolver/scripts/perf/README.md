# rocSOLVER Performance Scripts

`rocSOLVER/scripts/perf` includes scripts to benchmark rocSOLVER functions and collects the results for analysis and display.

## Building rocSOLVER for Benchmarking

To prepare rocSOLVER for benchmarking, follow the instructions from [rocSOLVER API documentation](https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/installation/installlinux.html#install-linux) to build and install the library and its clients.

## Benchmarking rocSOLVER with `perfoptim-suite`

The `perfoptim-suite` script executes the specified rocSOLVER functions, precision, and size cases. The results are written to csv files which are saved in the `rocsolver_customer01_benchmarks` directory.

Calling the script without any arguments
```
./perfoptim-suite
```
runs the default configuration which executes all available functions with real single and double precision and with small, medium and large size cases.

Options can be passed to the script as arguments to modify its behaviour. The available options are:
```
benchmark to run
valid options are: (default will run all of them)
potrf             -> Cholesky factorization (symmetric/Hermitian positive-definite)
potrfBatch        -> Cholesky factorization batch version
potrs             -> linear system solver with Cholesky
potrsBatch        -> linear system solver with Cholesky batch version
potri             -> matrix inversion with Cholesky
sytrf             -> Bunch-Kaufman factorization (symmetric indefinite)
sytrs             -> linear system solver with Bunch-Kaufman
getrf             -> LU factorization
getrfBatch        -> LU factorization batch version
getrfNpvt         -> LU factorization without pivoting
getrfNpvtBatch    -> LU factorization without pivoting batch version
getrs             -> linear system solver with LU
getrsBatch        -> linear system solver with LU batch version
getrsNpvt         -> linear system solver with no pivoting LU
getrsNpvtBatch    -> linear system solver with no pivoting LU batch version
getriBatch        -> matrix inversion with LU batch version
getriOOPBatch     -> Out-of-place matrix inversion with LU batch version
trtri             -> triangular matrix inversion
geqrf             -> QR factorization
geqrfBatch        -> QR factorization batch version
cholqr            -> Cholesky QR factorization
cholqrBatch       -> Cholesky QR factorization batch version
gels              -> Overdetermined linear system solver (least squares)
gelsBatch         -> Overdetermined linear system solver batch version
xxgqr             -> QR factorization Orthonormal/Unitary matrix construction
xxmqr             -> QR factorization Orthonormal/Unitary matrix multiply
larft             -> Triangular block reflector construction
xxtrd             -> Symmetric/Hermitian matrix tridiagonalization
xxgtr             -> to test QL factorization Orthonormal/Unitary matrix construction
xxmtr             -> to test QL factorization Orthonormal/Unitary matrix multiply
gebrd             -> General matrix bidiagonalization
xxgbr             -> to test LQ factorization Orthonormal/Unitary matrix construction
stedc             -> Tridiagonal eigenvalue problem (Divide and Conquer)
xxevd             -> Symmetric/Hermitian eigenvalue problem (Divide and Conquer)
xxgvd             -> Symmetric/Hermitian generalized eigenvalue problem (Divide and Conquer)
xxevdBatch        -> Symmetric/Hermitian eigenvalue problem (Divide and Conquer) batch version
xxevBatch         -> Symmetric/Hermitian eigenvalue problem (classic QR algorithm) batch version
xxevdx            -> Symmetric/Hermitian partial eigenvalue decomposition
xxgvdx            -> Symmetric/Hermitian partial generalized eigenvalue decomposition
xxevj             -> Symmetric/Hermitian eigenvalue problem (Jacobi iteration)
xxgvj             -> Symmetric/Hermitian generalized eigenvalue problem (Jacobi iteration)
xxevjBatch        -> Symmetric/Hermitian eigenvalue problem (Jacobi iteration) batch version
gesvd             -> Singular Value Decomposition (classic QR algorithm)
gesdd             -> Singular Value Decomposition (Divide & Conquer)
gesvdj            -> Singular Value Decomposition (Jacobi iteration)
gesvdjBatch       -> Singular Value Decomposition (Jacobi iteration) batch version
(note: several can be selected)

precisions to use
valid options are: (default is s,d)
s -> real single precision
d -> real double precision
c -> complex single precision
z -> complex double precision
(note: several can be selected)

size cases to run
valid options are: (default is small, medium, large)
small  -> see definitions in rocsolver_suites.py for included size values
medium -> see definitions in rocsolver_suites.py for included size values
large  -> see definitions in rocsolver_suites.py for included size values
huge   -> see definitions in rocsolver_suites.py for included size values
(note: several can be selected)
```

For example, benchmarking `geqrf` with real and complex single precisions on the small and large size cases would look like this:
```
./perfoptim-suite geqrf s c small large
```
After completion, the results of the benchmark will have been written to `rocsolver_customer01_benchmarks/sgeqrf_benchmarks.csv` and `rocsolver_customer01_benchmarks/cgeqrf_benchmarks.csv` for the real single precision case and the complex single precision case, respectively.
