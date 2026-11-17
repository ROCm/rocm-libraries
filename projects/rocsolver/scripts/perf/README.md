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

Options can be passed to the script as arguments to modify its behaviour. See `perfoptim-suite -h` help for available options. 

For example, benchmarking `geqrf` with real and complex single precisions on the small and large size cases would look like this:
```
./perfoptim-suite geqrf s c small large
```
After completion, the results of the benchmark will have been written to `rocsolver_customer01_benchmarks/sgeqrf_benchmarks.csv` and `rocsolver_customer01_benchmarks/cgeqrf_benchmarks.csv` for the real single precision case and the complex single precision case, respectively.
