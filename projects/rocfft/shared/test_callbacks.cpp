// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "test_callbacks.h"
#include "rocfft_complex.h"

#ifdef ROCFFT_MPI_ENABLE
#include <mpi.h>
#endif

// load/store callbacks - cbdata in each is actually a scalar double
// with a number to apply to each element
template <typename Tdata>
__host__ __device__ Tdata load_callback(Tdata* input, size_t offset, void* cbdata, void* sharedMem)
{
    auto testdata = static_cast<const callback_test_data*>(cbdata);
    // multiply each element by scalar
    return input[offset] * static_cast<decltype(std::real(input[offset]))>(testdata->scalar);
}

__device__ auto load_callback_dev_half           = load_callback<rocfft_fp16>;
__device__ auto load_callback_dev_complex_half   = load_callback<rocfft_complex<rocfft_fp16>>;
__device__ auto load_callback_dev_float          = load_callback<float>;
__device__ auto load_callback_dev_complex_float  = load_callback<rocfft_complex<float>>;
__device__ auto load_callback_dev_double         = load_callback<double>;
__device__ auto load_callback_dev_complex_double = load_callback<rocfft_complex<double>>;

// load/store callbacks - cbdata in each is actually a scalar double
// with a number to apply to each element
template <typename Tdata>
__host__ __device__ Tdata
    load_callback_round_trip_inverse(Tdata* input, size_t offset, void* cbdata, void* sharedMem)
{
    auto testdata = static_cast<const callback_test_data*>(cbdata);
    // subtract each element by scalar
    return input[offset] - static_cast<decltype(std::real(input[offset]))>(testdata->scalar);
}

__device__ auto load_callback_round_trip_inverse_dev_half
    = load_callback_round_trip_inverse<rocfft_fp16>;
__device__ auto load_callback_round_trip_inverse_dev_complex_half
    = load_callback_round_trip_inverse<rocfft_complex<rocfft_fp16>>;
__device__ auto load_callback_round_trip_inverse_dev_float
    = load_callback_round_trip_inverse<float>;
__device__ auto load_callback_round_trip_inverse_dev_complex_float
    = load_callback_round_trip_inverse<rocfft_complex<float>>;
__device__ auto load_callback_round_trip_inverse_dev_double
    = load_callback_round_trip_inverse<double>;
__device__ auto load_callback_round_trip_inverse_dev_complex_double
    = load_callback_round_trip_inverse<rocfft_complex<double>>;

void* get_load_callback_host(fft_array_type             itype,
                             fft_precision              precision,
                             callback_hip_error_handler runtime_err_handler,
                             bool                       round_trip_inverse)
{
    void* load_callback_host = nullptr;
    switch(itype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(precision)
        {
        case fft_precision_half:
            if(round_trip_inverse)
            {
                if(hipMemcpyFromSymbol(
                       &load_callback_host,
                       HIP_SYMBOL(load_callback_round_trip_inverse_dev_complex_half),
                       sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            else
            {
                if(hipMemcpyFromSymbol(&load_callback_host,
                                       HIP_SYMBOL(load_callback_dev_complex_half),
                                       sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            return load_callback_host;
        case fft_precision_single:
            if(round_trip_inverse)
            {
                if(hipMemcpyFromSymbol(
                       &load_callback_host,
                       HIP_SYMBOL(load_callback_round_trip_inverse_dev_complex_float),
                       sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            else
            {
                if(hipMemcpyFromSymbol(&load_callback_host,
                                       HIP_SYMBOL(load_callback_dev_complex_float),
                                       sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            return load_callback_host;
        case fft_precision_double:
            if(round_trip_inverse)
            {
                if(hipMemcpyFromSymbol(
                       &load_callback_host,
                       HIP_SYMBOL(load_callback_round_trip_inverse_dev_complex_double),
                       sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            else
            {
                if(hipMemcpyFromSymbol(&load_callback_host,
                                       HIP_SYMBOL(load_callback_dev_complex_double),
                                       sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            return load_callback_host;
        }
    }
    case fft_array_type_real:
    {
        switch(precision)
        {
        case fft_precision_half:
            if(round_trip_inverse)
            {
                if(hipMemcpyFromSymbol(&load_callback_host,
                                       HIP_SYMBOL(load_callback_round_trip_inverse_dev_half),
                                       sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            else
            {
                if(hipMemcpyFromSymbol(
                       &load_callback_host, HIP_SYMBOL(load_callback_dev_half), sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            return load_callback_host;
        case fft_precision_single:
            if(round_trip_inverse)
            {
                if(hipMemcpyFromSymbol(&load_callback_host,
                                       HIP_SYMBOL(load_callback_round_trip_inverse_dev_float),
                                       sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            else
            {
                if(hipMemcpyFromSymbol(
                       &load_callback_host, HIP_SYMBOL(load_callback_dev_float), sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            return load_callback_host;
        case fft_precision_double:
            if(round_trip_inverse)
            {
                if(hipMemcpyFromSymbol(&load_callback_host,
                                       HIP_SYMBOL(load_callback_round_trip_inverse_dev_double),
                                       sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            else
            {
                if(hipMemcpyFromSymbol(
                       &load_callback_host, HIP_SYMBOL(load_callback_dev_double), sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }

            return load_callback_host;
        }
    }
    default:
        // planar is unsupported for now
        return load_callback_host;
    }
}

template <typename Tdata>
__host__ __device__ static void
    store_callback(Tdata* output, size_t offset, Tdata element, void* cbdata, void* sharedMem)
{
    auto testdata = static_cast<callback_test_data*>(cbdata);
    // add scalar to each element
    output[offset] = element + static_cast<decltype(std::real(output[offset]))>(testdata->scalar);
}
__device__ auto store_callback_dev_half           = store_callback<rocfft_fp16>;
__device__ auto store_callback_dev_complex_half   = store_callback<rocfft_complex<rocfft_fp16>>;
__device__ auto store_callback_dev_float          = store_callback<float>;
__device__ auto store_callback_dev_complex_float  = store_callback<rocfft_complex<float>>;
__device__ auto store_callback_dev_double         = store_callback<double>;
__device__ auto store_callback_dev_complex_double = store_callback<rocfft_complex<double>>;

template <typename Tdata>
__host__ __device__ static void store_callback_round_trip_inverse(
    Tdata* output, size_t offset, Tdata element, void* cbdata, void* sharedMem)
{
    auto testdata = static_cast<callback_test_data*>(cbdata);
    // divide each element by scalar
    output[offset] = element / static_cast<decltype(std::real(output[offset]))>(testdata->scalar);
}
__device__ auto store_callback_round_trip_inverse_dev_half
    = store_callback_round_trip_inverse<rocfft_fp16>;
__device__ auto store_callback_round_trip_inverse_dev_complex_half
    = store_callback_round_trip_inverse<rocfft_complex<rocfft_fp16>>;
__device__ auto store_callback_round_trip_inverse_dev_float
    = store_callback_round_trip_inverse<float>;
__device__ auto store_callback_round_trip_inverse_dev_complex_float
    = store_callback_round_trip_inverse<rocfft_complex<float>>;
__device__ auto store_callback_round_trip_inverse_dev_double
    = store_callback_round_trip_inverse<double>;
__device__ auto store_callback_round_trip_inverse_dev_complex_double
    = store_callback_round_trip_inverse<rocfft_complex<double>>;

void* get_store_callback_host(fft_array_type             otype,
                              fft_precision              precision,
                              callback_hip_error_handler runtime_err_handler,
                              bool                       round_trip_inverse)
{
    void* store_callback_host = nullptr;
    switch(otype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(precision)
        {
        case fft_precision_half:
            if(round_trip_inverse)
            {
                if(hipMemcpyFromSymbol(
                       &store_callback_host,
                       HIP_SYMBOL(store_callback_round_trip_inverse_dev_complex_half),
                       sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            else
            {
                if(hipMemcpyFromSymbol(&store_callback_host,
                                       HIP_SYMBOL(store_callback_dev_complex_half),
                                       sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            return store_callback_host;
        case fft_precision_single:
            if(round_trip_inverse)
            {
                if(hipMemcpyFromSymbol(
                       &store_callback_host,
                       HIP_SYMBOL(store_callback_round_trip_inverse_dev_complex_float),
                       sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            else
            {
                if(hipMemcpyFromSymbol(&store_callback_host,
                                       HIP_SYMBOL(store_callback_dev_complex_float),
                                       sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            return store_callback_host;
        case fft_precision_double:
            if(round_trip_inverse)
            {
                if(hipMemcpyFromSymbol(
                       &store_callback_host,
                       HIP_SYMBOL(store_callback_round_trip_inverse_dev_complex_double),
                       sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            else
            {
                if(hipMemcpyFromSymbol(&store_callback_host,
                                       HIP_SYMBOL(store_callback_dev_complex_double),
                                       sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            return store_callback_host;
        }
    }
    case fft_array_type_real:
    {
        switch(precision)
        {
        case fft_precision_half:
            if(round_trip_inverse)
            {
                if(hipMemcpyFromSymbol(&store_callback_host,
                                       HIP_SYMBOL(store_callback_round_trip_inverse_dev_half),
                                       sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            else
            {
                if(hipMemcpyFromSymbol(
                       &store_callback_host, HIP_SYMBOL(store_callback_dev_half), sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            return store_callback_host;
        case fft_precision_single:
            if(round_trip_inverse)
            {
                if(hipMemcpyFromSymbol(&store_callback_host,
                                       HIP_SYMBOL(store_callback_round_trip_inverse_dev_float),
                                       sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            else
            {
                if(hipMemcpyFromSymbol(
                       &store_callback_host, HIP_SYMBOL(store_callback_dev_float), sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            return store_callback_host;
        case fft_precision_double:
            if(round_trip_inverse)
            {
                if(hipMemcpyFromSymbol(&store_callback_host,
                                       HIP_SYMBOL(store_callback_round_trip_inverse_dev_double),
                                       sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            else
            {
                if(hipMemcpyFromSymbol(
                       &store_callback_host, HIP_SYMBOL(store_callback_dev_double), sizeof(void*))
                   != hipSuccess)
                    runtime_err_handler("hipMemcpyFromSymbol failed");
            }
            return store_callback_host;
        }
    }
    default:
        // planar is unsupported for now
        return store_callback_host;
    }
}

// Apply store callback if necessary
void apply_store_callback(const fft_params& params, std::vector<hostbuf>& output)
{
    if(!params.run_callbacks)
        return;

    callback_test_data cbdata;
    cbdata.scalar = params.store_cb_scalar;

    switch(params.otype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(params.precision)
        {
        case fft_precision_half:
        {
            const size_t elem_size = sizeof(rocfft_complex<rocfft_fp16>);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin
                = reinterpret_cast<rocfft_complex<rocfft_fp16>*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                store_callback(output_begin, i, element, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_single:
        {
            const size_t elem_size = sizeof(rocfft_complex<float>);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<rocfft_complex<float>*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                store_callback(output_begin, i, element, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_double:
        {
            const size_t elem_size = sizeof(rocfft_complex<double>);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<rocfft_complex<double>*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                store_callback(output_begin, i, element, &cbdata, nullptr);
            }
            break;
        }
        }
    }
    break;
    case fft_array_type_complex_planar:
    case fft_array_type_hermitian_planar:
    {
        throw std::runtime_error("planar callbacks are not supported");
    }
    break;
    case fft_array_type_real:
    {
        switch(params.precision)
        {
        case fft_precision_half:
        {
            const size_t elem_size = sizeof(rocfft_fp16);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<rocfft_fp16*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                store_callback(output_begin, i, element, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_single:
        {
            const size_t elem_size = sizeof(float);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<float*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                store_callback(output_begin, i, element, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_double:
        {
            const size_t elem_size = sizeof(double);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<double*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                store_callback(output_begin, i, element, &cbdata, nullptr);
            }
            break;
        }
        }
    }
    break;
    default:
        // this is FFTW data which should always be interleaved (if complex)
        abort();
    }
}

// apply load callback if necessary
void apply_load_callback(const fft_params& params, std::vector<hostbuf>& input)
{
    if(!params.run_callbacks)
        return;
    // we're applying callbacks to FFTW input/output which we can
    // assume is contiguous and non-planar

    callback_test_data cbdata;
    cbdata.scalar = params.load_cb_scalar;

    switch(params.itype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(params.precision)
        {
        case fft_precision_half:
        {
            const size_t elem_size = sizeof(rocfft_complex<rocfft_fp16>);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<rocfft_complex<rocfft_fp16>*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_single:
        {
            const size_t elem_size = sizeof(rocfft_complex<float>);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<rocfft_complex<float>*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_double:
        {
            const size_t elem_size = sizeof(rocfft_complex<double>);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<rocfft_complex<double>*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        }
    }
    break;
    case fft_array_type_real:
    {
        switch(params.precision)
        {
        case fft_precision_half:
        {
            const size_t elem_size = sizeof(rocfft_fp16);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<rocfft_fp16*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_single:
        {
            const size_t elem_size = sizeof(float);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<float*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_double:
        {
            const size_t elem_size = sizeof(double);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<double*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        }
    }
    break;
    default:
        // this is FFTW data which should always be interleaved (if complex)
        abort();
    }
}

// For the current rank, get a vector of load callback function +
// data pointers.  The pointers need to be in the order that
// fields+bricks were specified to the FFT plan.  Pointers need to be
// copied to the host from the device specified by the respective
// brick.
void get_rank_load_callbacks(const fft_params&                          params,
                             std::vector<void*>&                        load_cb_func,
                             std::vector<void*>&                        load_cb_data,
                             callback_hip_error_handler                 runtime_err_handler,
                             bool                                       round_trip_inverse,
                             std::vector<gpubuf_t<callback_test_data>>& all_cb_data)
{
    int mpi_rank = 0;
#ifdef ROCFFT_MPI_ENABLE
    if(params.mp_lib == fft_params::fft_mp_lib_mpi)
    {
        MPI_Comm_rank(*static_cast<MPI_Comm*>(params.mp_comm), &mpi_rank);
    }
#endif

    // Copy callback pointer from current device and add to output vec
    auto add_load_cb = [&]() {
        void* load_cb_host = get_load_callback_host(
            params.itype, params.precision, runtime_err_handler, round_trip_inverse);

        callback_test_data load_cb_data_host;

        if(round_trip_inverse)
        {
            load_cb_data_host.scalar = params.store_cb_scalar;
        }
        else
        {
            load_cb_data_host.scalar = params.load_cb_scalar;
        }

        auto& load_cb_data_dev = all_cb_data.emplace_back();
        auto  hip_status       = load_cb_data_dev.alloc(sizeof(callback_test_data));
        if(hip_status != hipSuccess)
        {
            runtime_err_handler(
                "Error occurred when allocating device memory for loading callback");
        }
        hip_status = hipMemcpy(load_cb_data_dev.data(),
                               &load_cb_data_host,
                               sizeof(callback_test_data),
                               hipMemcpyHostToDevice);
        if(hip_status != hipSuccess)
        {
            runtime_err_handler("Error occurred when copying device memory for loading callback");
        }
        load_cb_func.push_back(load_cb_host);
        load_cb_data.push_back(load_cb_data_dev.data());
    };

    if(params.ifields.empty())
    {
        // for library-decomposed multi-GPU, one cb for each device
        if(params.multiGPU > 1)
        {
            for(int i = 0; i < static_cast<int>(params.multiGPU); ++i)
            {
                rocfft_scoped_device dev(i);
                add_load_cb();
            }
        }
        else
        {
            // load cb for current HIP device
            add_load_cb();
        }
    }
    else
    {
        // user-specified decomposition - copy func+data for each brick
        // on this rank
        for(size_t i = 0; i < params.ifields.front().bricks.size(); ++i)
        {
            if(params.ifields.front().bricks[i].rank != mpi_rank)
                continue;

            // load cb for this brick's device
            rocfft_scoped_device dev(params.ifields.front().bricks[i].device);
            add_load_cb();
        }
    }
}

// For the current rank, get a vector of store callback function +
// data pointers.  The pointers need to be in the order that
// fields+bricks were specified to the FFT plan.  Pointers need to be
// copied to the host from the device specified by the respective
// brick.
void get_rank_store_callbacks(const fft_params&                          params,
                              std::vector<void*>&                        store_cb_func,
                              std::vector<void*>&                        store_cb_data,
                              callback_hip_error_handler                 runtime_err_handler,
                              bool                                       round_trip_inverse,
                              std::vector<gpubuf_t<callback_test_data>>& all_cb_data)
{
    int mpi_rank = 0;
#ifdef ROCFFT_MPI_ENABLE
    if(params.mp_lib == fft_params::fft_mp_lib_mpi)
    {
        MPI_Comm_rank(*static_cast<MPI_Comm*>(params.mp_comm), &mpi_rank);
    }
#endif

    // Copy callback pointer from current device and add to output vec
    auto add_store_cb = [&]() {
        void* store_cb_host = get_store_callback_host(
            params.otype, params.precision, runtime_err_handler, round_trip_inverse);

        callback_test_data store_cb_data_host;

        if(round_trip_inverse)
        {
            store_cb_data_host.scalar = params.load_cb_scalar;
        }
        else
        {
            store_cb_data_host.scalar = params.store_cb_scalar;
        }

        auto& store_cb_data_dev = all_cb_data.emplace_back();
        auto  hip_status        = store_cb_data_dev.alloc(sizeof(callback_test_data));
        if(hip_status != hipSuccess)
        {
            runtime_err_handler(
                "Error occurred when allocating device memory for storing callback");
        }

        hip_status = hipMemcpy(store_cb_data_dev.data(),
                               &store_cb_data_host,
                               sizeof(callback_test_data),
                               hipMemcpyHostToDevice);
        if(hip_status != hipSuccess)
        {
            runtime_err_handler("Error occurred when copying device memory for storing callback");
        }

        store_cb_func.push_back(store_cb_host);
        store_cb_data.push_back(store_cb_data_dev.data());
    };

    if(params.ofields.empty())
    {
        // for library-decomposed multi-GPU, one cb for each device
        if(params.multiGPU > 1)
        {
            for(int i = 0; i < static_cast<int>(params.multiGPU); ++i)
            {
                rocfft_scoped_device dev(i);
                add_store_cb();
            }
        }
        else
        {
            // store cb for current HIP device
            add_store_cb();
        }
    }
    else
    {
        // user-specified decomposition - copy func+data for each brick
        // on this rank
        for(size_t i = 0; i < params.ofields.front().bricks.size(); ++i)
        {
            if(params.ofields.front().bricks[i].rank != mpi_rank)
                continue;

            // store cb for this brick's device
            rocfft_scoped_device dev(params.ofields.front().bricks[i].device);
            add_store_cb();
        }
    }
}
