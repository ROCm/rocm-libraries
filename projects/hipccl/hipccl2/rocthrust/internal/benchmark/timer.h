#pragma once

#include <cassert>
#include <thrust/detail/config.h>

#if (THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC)

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

class hip_timer
{
    hipEvent_t start_;
    hipEvent_t stop_;

 public:
    hip_timer()
    {
        HIP_CHECK(hipEventCreate(&start_));
        HIP_CHECK(hipEventCreate(&stop_));
    }

    ~hip_timer()
    {
        HIP_CHECK(hipEventDestroy(start_));
        HIP_CHECK(hipEventDestroy(stop_));
    }

    void start()
    {
        HIP_CHECK(hipEventRecord(start_, 0));
    }

    void stop()
    {
        HIP_CHECK(hipEventRecord(stop_, 0));
        HIP_CHECK(hipEventSynchronize(stop_));
    }

    double milliseconds_elapsed()
    {
        float elapsed_time;
        HIP_CHECK(hipEventElapsedTime(&elapsed_time, start_, stop_));
        return elapsed_time;
    }

    double seconds_elapsed()
    {
        return milliseconds_elapsed() / 1000.0;
    }
};

#endif

#if (THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC)

#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                                 \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#  define CUDA_SAFE_CALL( call) do {                                         \
    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
    cudaError err = cudaDeviceSynchronize();                                 \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

class cuda_timer
{
    cudaEvent_t start_;
    cudaEvent_t stop_;

 public:
    cuda_timer()
    {
        CUDA_SAFE_CALL(cudaEventCreate(&start_));
        CUDA_SAFE_CALL(cudaEventCreate(&stop_));
    }

    ~cuda_timer()
    {
        CUDA_SAFE_CALL(cudaEventDestroy(start_));
        CUDA_SAFE_CALL(cudaEventDestroy(stop_));
    }

    void start()
    {
        CUDA_SAFE_CALL(cudaEventRecord(start_, 0));
    }

    void stop()
    {
        CUDA_SAFE_CALL(cudaEventRecord(stop_, 0));
        CUDA_SAFE_CALL(cudaEventSynchronize(stop_));
    }

    double milliseconds_elapsed()
    {
        float elapsed_time;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, start_, stop_));
        return elapsed_time;
    }

    double seconds_elapsed()
    {
        return milliseconds_elapsed() / 1000.0;
    }
};

#endif

#if (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC)
#include <windows.h>

class steady_timer
{
    LARGE_INTEGER frequency_; // Cached to avoid system calls.
    LARGE_INTEGER start_;
    LARGE_INTEGER stop_;

 public:
    steady_timer() : start_(), stop_(), frequency_()
    {
        BOOL const r = QueryPerformanceFrequency(&frequency_);
        assert(0 != r);
    }

    void start()
    {
        BOOL const r = QueryPerformanceCounter(&start_);
        assert(0 != r);
    }

    void stop()
    {
        BOOL const r = QueryPerformanceCounter(&stop_);
        assert(0 != r);
    }

    double seconds_elapsed()
    {
        return double(stop_.QuadPart - start_.QuadPart)
             / double(frequency_.QuadPart);
    }
};
#else
#include <time.h>

class steady_timer
{
    timespec start_;
    timespec stop_;

 public:
    steady_timer() : start_(), stop_() {}

    void start()
    {
        int const r = clock_gettime(CLOCK_MONOTONIC, &start_);
        assert(0 == r);
    }

    void stop()
    {
        int const r = clock_gettime(CLOCK_MONOTONIC, &stop_);
        assert(0 == r);
    }

    double seconds_elapsed()
    {
        return double(stop_.tv_sec  - start_.tv_sec)
             + double(stop_.tv_nsec - start_.tv_nsec) * 1.0e-9;
    }
};
#endif
