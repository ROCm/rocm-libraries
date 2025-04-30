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
runs the default configuration which executes all available functions with real double precision and all the size cases.

Options can be passed to the script as arguments to modify its behaviour. The available options are:
```
benchmark to run
valid options are: (default will run all of them)
syevd         -> eigensolver D&C + QR algorithm (heevd in complex precision)
syevdx        -> eigensolver D&C + bisection (heevdx in complex precision)
syevj         -> eigensolver Jacobi (heevj in complex precision)
syevjBatch    -> eigensolver Jacobi batch version (heevjBatch in complex precision)
gesvd         -> SVD QR algorithm
gesvdj        -> SVD Jacobi
gesvdjBatch   -> SVD Jacobi batch version
potrf         -> Cholesky factorization
potrfBatch    -> Cholesky factorization batch version
geqrf         -> Orthogonal factorization
(note: several can be selected)

precisions to use
valid options are: (default is d)
s -> real single precision
d -> real double precision
c -> complex single precision
z -> complex double precision
(note: several can be selected)

size cases to run
valid options are: (default is large)
small  -> see definitions in rocsolver-perfoptim-suite.py for included size values
medium -> see definitions in rocsolver-perfoptim-suite.py for included size values
large  -> see definitions in rocsolver-perfoptim-suite.py for included size values
(note: select only one as small is a sub-set of medium which is a sub-set of large)
```

For example, benchmarking `geqrf` with real single and real double precisions on the medium and small size cases would look like this:
```
./perfoptim-suite geqrf s d medium
```
After completion, the results of the benchmark will have been written to `rocsolver_customer01_benchmarks/sgeqrf_benchmarks.csv` and `rocsolver_customer01_benchmarks/dgeqrf_benchmarks.csv` for the real single precision case and the real double precision case, respectively.
