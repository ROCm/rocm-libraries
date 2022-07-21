# Benchmark calculating the timing and bandwidth of rocBlas' saxpy.

Use the makefile to construct the executable saxpy.exe.
In the sourcefile saxpy_rocblas.cpp, hipEvents are used to time the rocblas saxpy function.
Ten calls to saxpy are used for every size ranging from 16384 to 1073741824.
The timing and read bandwidth for the first run is not counted as part of the average, as
it has start up costs associated with it.

The executable stores the results of the benchmark in a file called `rocblas_results.data`.

The steps to run this benchmark are:
1. Compile, run the command `make`
2. Execute, run the executable `saxpy.exe`
3. View the results in `rocblas_results.data`

The results should have the following form:

>
> Size = 16384
>
> Instance 0	Time 9.89702ms	Read bandwidth 0.00827724GB/s
>
> Instance 1	Time 0.01648ms	Read bandwidth 4.97087GB/s
>
> Instance 2	Time 0.01824ms	Read bandwidth 4.49123GB/s
>
> Instance 3	Time 0.01632ms	Read bandwidth 5.01961GB/s
>
> Instance 4	Time 0.01728ms	Read bandwidth 4.74074GB/s
>
> Instance 5	Time 0.01696ms	Read bandwidth 4.83019GB/s
>
> Instance 6	Time 0.01664ms	Read bandwidth 4.92308GB/s
>
> Instance 7	Time 0.01728ms	Read bandwidth 4.74074GB/s
>
> Instance 8	Time 0.01664ms	Read bandwidth 4.92308GB/s
>
> Instance 9	Time 0.01728ms	Read bandwidth 4.74074GB/s
>
> Averaged Time 0.0170133ms
>
> Averaged Effective Read Bandwidth = 4.82003GB/s

For each size. Notice how the time for Instance 0 is a large outlier in the data, illustrating why we
discard the first run.
