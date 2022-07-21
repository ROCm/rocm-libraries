#include <fstream>
#include <iostream>

#include <hip/hip_runtime.h>
#include <rocblas.h>

/* Function to fill the arrays to that will be accessed by rocBlas. */
__global__ void alloc(float dx[], float dy[])
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    dx[tid] = 1.f;
    dy[tid] = 1.f;
}

int main(int argc, char** argv)
{
    using lint          = long long int;
    float         alpha = 1.5f;
    std::ofstream resultsData;
    resultsData.open("rocblas_results.data");
    float *        dx, *dy;
    float          secs, secsAvg;
    double         rBand, rBandAvg;
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    for(lint n = 16384; n <= 1073741824; n <<= 1)
    {
        std::cout << "Computing rocblas_saxpy for size " << n << std::endl;
        resultsData << "Size = " << n << std::endl;
        secsAvg  = 0.f;
        rBandAvg = 0.;
        hipMalloc(&dx, n * sizeof(float));
        hipMalloc(&dy, n * sizeof(float));
        alloc<<<int(n / 1024), 1024>>>(dx, dy);
        hipDeviceSynchronize();
        for(int i = 0; i < 10; i++)
        {
            secs = 0.f;
            hipEventRecord(start, 0);
            rocblas_status status = rocblas_saxpy(handle, n, &alpha, dx, 1, dy, 1);
            hipEventRecord(stop, 0);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&secs, start, stop);
            lint   numer = n * 2 * 3;
            double denom = secs * 1.2e6;
            rBand        = numer / denom;
            resultsData << "Instance " << i << '\t' << "Time " << secs << "ms" << '\t'
                        << "Read bandwidth " << rBand << "GB/s" << std::endl;
            if(i > 0)
            {
                secsAvg += secs;
                rBandAvg += rBand;
            }
        }
        secsAvg /= 9; //Take the average.
        rBandAvg /= 9; //Average Read bandwdith

        resultsData << "Averaged Time " << secsAvg << "ms" << std::endl;
        resultsData << "Averaged Effective Read Bandwidth = " << rBandAvg << "GB/s" << std::endl;
        resultsData << std::endl;
        hipFree(dx);
        hipFree(dy);
    }
    hipEventDestroy(start);
    hipEventDestroy(stop);
    resultsData.close();
}
