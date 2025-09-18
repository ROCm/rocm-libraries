// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TEST_LINKING_NEW_SCAN_HPP_
#define TEST_LINKING_NEW_SCAN_HPP_

#include <iterator>
#include <type_traits>

#include <rocprim/block/block_store_func.hpp>
#include <rocprim/common.hpp>
#include <rocprim/config.hpp>
#include <rocprim/device/detail/config/device_scan.hpp>
#include <rocprim/device/device_scan_config.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/type_traits.hpp>

// This file contains simplified versions of functions and kernels from
// rocprim/device/device_scan.hpp. The idea is to emulate a situation when two libraries use
// different versions of rocPRIM. Their implementations are intentionally different: the kernel
// fills the output with a constant value, and the function requests a different size of temporary
// storage.
// The functions/kernels here MUST have exactly the same signatures as in the original file.
// Then, if the inline namespace is used, the linker is able to distinguish between two
// implementations of one function. Otherwise, one of the tests in test_linking.cpp MAY fail
// because the linker uses an incorrect implementation of the function (as they share one name).

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class ArchConfig,
         bool Exclusive,
         class InputIterator,
         class OutputIterator,
         class BinaryFunction,
         class AccType>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void single_scan_kernel_impl(InputIterator  input,
                                                                 const size_t   input_size,
                                                                 AccType        initial_value,
                                                                 OutputIterator output,
                                                                 BinaryFunction scan_op)
{
    (void)input;
    (void)initial_value;
    (void)scan_op;

    static constexpr scan_config_params params = ArchConfig::params;

    constexpr unsigned int block_size       = params.kernel_config.block_size;
    constexpr unsigned int items_per_thread = params.kernel_config.items_per_thread;

    AccType values[items_per_thread];
    for(unsigned int i = 0; i < items_per_thread; ++i)
    {
        values[i] = 12345;
    }
    block_store_direct_striped<block_size>(flat_block_thread_id(), output, values, input_size);
}

template<class Config,
         bool Exclusive,
         class InputIterator,
         class OutputIterator,
         class BinaryFunction,
         class InitValueType,
         class AccType>
inline hipError_t launch_single_scan(detail::target_arch arch,
                                     InputIterator       input,
                                     const size_t        size,
                                     const InitValueType initial_value,
                                     OutputIterator      output,
                                     BinaryFunction      scan_op,
                                     dim3                grid,
                                     dim3                block,
                                     size_t              shmem,
                                     hipStream_t         stream)
{
    auto kernel = [=](auto arch_config)
    {
        single_scan_kernel_impl<decltype(arch_config), Exclusive>(
            input,
            size,
            static_cast<AccType>(get_input_value(initial_value)),
            output,
            scan_op);
    };

    return execute_launch_plan<Config>(arch, kernel, grid, block, shmem, stream);
}

template<lookback_scan_determinism Determinism,
         bool                      Exclusive,
         class Config,
         class InputIterator,
         class OutputIterator,
         class InitValueType,
         class BinaryFunction,
         class AccType>
inline auto scan_impl(void*               temporary_storage,
                      size_t&             storage_size,
                      InputIterator       input,
                      OutputIterator      output,
                      const InitValueType initial_value,
                      const size_t        size,
                      BinaryFunction      scan_op,
                      const hipStream_t   stream,
                      bool                debug_synchronous)
{
    (void)debug_synchronous;

    using config = wrapped_scan_config<Config, AccType>;

    detail::target_arch target_arch;
    hipError_t          result = host_target_arch(stream, target_arch);
    if(result != hipSuccess)
    {
        return result;
    }
    const scan_config_params params = dispatch_target_arch<config, false>(target_arch);

    const unsigned int block_size       = params.kernel_config.block_size;
    const unsigned int items_per_thread = params.kernel_config.items_per_thread;
    const auto         items_per_block  = block_size * items_per_thread;

    if(temporary_storage == nullptr)
    {
        storage_size = 12345;
        return hipSuccess;
    }

    const unsigned int number_of_blocks = (size + items_per_block - 1) / items_per_block;
    if(number_of_blocks > 1)
    {
        return hipErrorInvalidValue;
    }

    ROCPRIM_RETURN_ON_ERROR(launch_single_scan<config,
                                               Exclusive,
                                               InputIterator,
                                               OutputIterator,
                                               BinaryFunction,
                                               InitValueType,
                                               AccType>(target_arch,
                                                        input,
                                                        size,
                                                        initial_value,
                                                        output,
                                                        scan_op,
                                                        dim3(1),
                                                        dim3(block_size),
                                                        0,
                                                        stream));
    return hipGetLastError();
}

} // namespace detail

template<class Config = default_config,
         class InputIterator,
         class OutputIterator,
         class BinaryFunction
         = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>,
         class AccType
         = ::rocprim::accumulator_t<BinaryFunction,
                                    typename std::iterator_traits<InputIterator>::value_type>>
inline hipError_t inclusive_scan(void*             temporary_storage,
                                 size_t&           storage_size,
                                 InputIterator     input,
                                 OutputIterator    output,
                                 const size_t      size,
                                 BinaryFunction    scan_op           = BinaryFunction(),
                                 const hipStream_t stream            = 0,
                                 bool              debug_synchronous = false)
{
    return detail::scan_impl<detail::lookback_scan_determinism::default_determinism,
                             false,
                             Config,
                             InputIterator,
                             OutputIterator,
                             AccType,
                             BinaryFunction,
                             AccType>(temporary_storage,
                                      storage_size,
                                      input,
                                      output,
                                      AccType{},
                                      size,
                                      scan_op,
                                      stream,
                                      debug_synchronous);
}

END_ROCPRIM_NAMESPACE

#endif // TEST_LINKING_NEW_SCAN_HPP_
