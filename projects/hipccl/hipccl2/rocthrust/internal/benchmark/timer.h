#pragma once

#include <thrust/detail/config.h>

#include <cassert>

#if(THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC || defined(_WIN32))
#include <windows.h>

class steady_timer
{
    LARGE_INTEGER start_;
    LARGE_INTEGER stop_;
    LARGE_INTEGER frequency_; // Cached to avoid system calls.

 public:
    steady_timer() : start_(), stop_(), frequency_()
    {
        BOOL const r = QueryPerformanceFrequency(&frequency_);
        assert(0 != r);
        (void)r; // Silence unused variable 'r' in Release builds, when
                 // the assertion evaporates.
    }

    void start()
    {
        BOOL const r = QueryPerformanceCounter(&start_);
        assert(0 != r);
        (void)r; // Silence unused variable 'r' in Release builds, when
                 // the assertion evaporates.
    }


    void stop()
    {
        BOOL const r = QueryPerformanceCounter(&stop_);
        assert(0 != r);
        (void)r; // Silence unused variable 'r' in Release builds, when
                 // the assertion evaporates.
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
        (void)r; // Silence unused variable 'r' in Release builds, when
                 // the assertion evaporates.
    }

    void stop()
    {
        int const r = clock_gettime(CLOCK_MONOTONIC, &stop_);
        assert(0 == r);
        (void)r; // Silence unused variable 'r' in Release builds, when
                 // the assertion evaporates.
    }

    double seconds_elapsed()
    {
        return double(stop_.tv_sec  - start_.tv_sec)
             + double(stop_.tv_nsec - start_.tv_nsec) * 1.0e-9;
    }
};
#endif
