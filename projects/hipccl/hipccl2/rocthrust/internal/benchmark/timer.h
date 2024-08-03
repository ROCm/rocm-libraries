#pragma once

#include <thrust/detail/config.h>

#include <cassert>
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
