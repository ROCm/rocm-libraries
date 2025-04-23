/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

//!
//! @brief Implementation of a common benchmark code
//!
template <typename LAMBDA>
class Benchmark
{
public:
    //!
    //! @brief Constructor
    //! @param lambda_to_benchmark     The lambda to be benchmarked.
    //! @param stream                  The Hip stream.
    //! @param arg                     Arguments struct, arguments to run benchmark
    //! @param flush_batch_count       number of copies of arrays in rotating buffer, set to 1 for no rotating buffer
    //!
    Benchmark(LAMBDA           lambda_to_benchmark,
              hipStream_t      stream,
              const Arguments& arg,
              size_t           flush_batch_count)
        : m_lambda_to_benchmark(lambda_to_benchmark)
        , m_stream(stream)
        , m_arg(arg)
        , m_flush_batch_count(flush_batch_count)
        , m_hot_time(0)
        , m_cold_time(0){};

    void   run_timer();
    double get_hot_time() const;
    double get_cold_time() const;

private:
    LAMBDA      m_lambda_to_benchmark;
    Arguments   m_arg;
    hipStream_t m_stream;
    size_t      m_flush_batch_count;
    double      m_hot_time;
    double      m_cold_time;
};

// timer calls m_lambda_to_benchmark in a loop m_arg.iters + m_arg.cold_iters times
// timer returns the time to call the lambda m_arg.iters times
// timer rotates through m_flush_batch_count copies of arrays to flush MALL
template <typename LAMBDA>
void Benchmark<LAMBDA>::run_timer()
{
    double cold_start, cold_end, hot_end;
    cold_start = get_time_us_sync(m_stream);
    for(int iter = 0; iter < m_arg.iters + m_arg.cold_iters; iter++)
    {
        if(iter == m_arg.cold_iters)
            cold_end = get_time_us_sync(m_stream);

        int flush_index = iter % m_flush_batch_count;

        m_lambda_to_benchmark(flush_index);
    }

    hot_end     = get_time_us_sync(m_stream);
    m_cold_time = cold_end - cold_start;
    m_hot_time  = hot_end - cold_end;
}

template <typename LAMBDA>
double Benchmark<LAMBDA>::get_hot_time() const
{
    return this->m_hot_time;
}

template <typename LAMBDA>
double Benchmark<LAMBDA>::get_cold_time() const
{
    return this->m_cold_time;
}
