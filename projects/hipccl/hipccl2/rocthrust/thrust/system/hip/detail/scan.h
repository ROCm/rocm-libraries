#pragma once


#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/functional.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>

#include <thrust/system/hip/detail/execution_policy.h>
#include <thrust/system/hip/detail/util.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/detail/memory_buffer.h>
#include <thrust/detail/mpl/math.h>
#include <thrust/detail/minmax.h>
#include <thrust/distance.h>

// rocprim include
#include <rocprim/rocprim.hpp>
#include <thrust/detail/cstdint.h>
#include <thrust/detail/alignment.h>

BEGIN_NS_THRUST
template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename AssociativeOperator>
__host__ __device__ OutputIterator
inclusive_scan(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
               InputIterator                                               first,
               InputIterator                                               last,
               OutputIterator                                              result,
               AssociativeOperator                                         binary_op);

template <typename DerivedPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename T,
          typename AssociativeOperator>
__host__ __device__ OutputIterator
exclusive_scan(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
               InputIterator                                               first,
               InputIterator                                               last,
               OutputIterator                                              result,
               T                                                           init,
               AssociativeOperator                                         binary_op);

namespace hip_rocprim {

namespace __scan {

template <class Policy,
          class InputIt,
          class OutputIt,
          class Size,
          class ScanOp>
OutputIt THRUST_HIP_RUNTIME_FUNCTION
inclusive_scan(Policy                 &policy,
               InputIt                input_it,
               OutputIt               output_it,
               Size                   num_items,
               ScanOp                 scan_op)
{
    if (num_items == 0)
        return output_it;

    void *       d_temp_storage     = nullptr;
    size_t       temp_storage_bytes = 0;
    hipStream_t  stream             = hip_rocprim::stream(policy);
    bool         debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

    // Determine temporary device storage requirements.
    hip_rocprim::throw_on_error(
            rocprim::inclusive_scan(d_temp_storage,
                                    temp_storage_bytes,
                                    input_it,
                                    output_it,
                                    num_items,
                                    scan_op,
                                    stream,
                                    debug_sync),
            "scan failed on 1st step");

    hipDeviceSynchronize();

    // Allocate temporary storage.
    temp_storage_bytes = rocprim::detail::align_size(temp_storage_bytes);
    d_temp_storage = hip_rocprim::get_memory_buffer(policy, temp_storage_bytes);
    hip_rocprim::throw_on_error(hipGetLastError(),
                                "scan failed to get memory buffer");

    // Run scan.
    hip_rocprim::throw_on_error(
            rocprim::inclusive_scan(d_temp_storage,
                                    temp_storage_bytes,
                                    input_it,
                                    output_it,
                                    num_items,
                                    scan_op,
                                    stream,
                                    debug_sync),
            "scan failed on 2nd step");

    hip_rocprim::return_memory_buffer(policy, d_temp_storage);
    hip_rocprim::throw_on_error(hipGetLastError(),
                                "scan failed to return memory buffer");

    return output_it + num_items;
}

template <class Policy,
          class InputIt,
          class OutputIt,
          class Size,
          class T,
          class ScanOp>
OutputIt THRUST_HIP_RUNTIME_FUNCTION
exclusive_scan(Policy                 &policy,
               InputIt                input_it,
               OutputIt               output_it,
               Size                   num_items,
               T                      init,
               ScanOp                 scan_op)
{
    if (num_items == 0)
        return output_it;

    void *       d_temp_storage     = nullptr;
    size_t       temp_storage_bytes = 0;
    hipStream_t  stream             = hip_rocprim::stream(policy);
    bool         debug_sync         = THRUST_HIP_DEBUG_SYNC_FLAG;

    // Determine temporary device storage requirements.
    hip_rocprim::throw_on_error(
            rocprim::exclusive_scan(d_temp_storage,
                                    temp_storage_bytes,
                                    input_it,
                                    output_it,
                                    init,
                                    num_items,
                                    scan_op,
                                    stream,
                                    debug_sync),
            "scan failed on 1st step");

    hipDeviceSynchronize();

    // Allocate temporary storage.
    temp_storage_bytes = rocprim::detail::align_size(temp_storage_bytes);
    d_temp_storage = hip_rocprim::get_memory_buffer(policy, temp_storage_bytes);
    hip_rocprim::throw_on_error(hipGetLastError(),
                                "scan failed to get memory buffer");

    // Run scan.
    hip_rocprim::throw_on_error(
            rocprim::exclusive_scan(d_temp_storage,
                                    temp_storage_bytes,
                                    input_it,
                                    output_it,
                                    init,
                                    num_items,
                                    scan_op,
                                    stream,
                                    debug_sync),
            "scan failed on 2nd step");

    hip_rocprim::return_memory_buffer(policy, d_temp_storage);
    hip_rocprim::throw_on_error(hipGetLastError(),
                                "scan failed to return memory buffer");

    return output_it + num_items;
}

}    // namespace __scan

//-------------------------
// Thrust API entry points
//-------------------------

template <class Derived,
          class InputIt,
          class Size,
          class OutputIt,
          class ScanOp>
OutputIt THRUST_HIP_FUNCTION
inclusive_scan_n(execution_policy<Derived> &policy,
                 InputIt                    input_it,
                 Size                       num_items,
                 OutputIt                   result,
                 ScanOp                     scan_op)
{
    OutputIt ret = result;
    THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((
        rocprim::inclusive_scan<rocprim::default_config, InputIt, OutputIt, ScanOp>
    ));
#if __THRUST_HAS_HIPRT__
    ret = __scan::inclusive_scan(policy,
                                 input_it,
                                 result,
                                 num_items,
                                 scan_op);
#else // __THRUST_HAS_HIPRT__
    ret = thrust::inclusive_scan(cvt_to_seq(derived_cast(policy)),
                                 input_it,
                                 input_it + num_items,
                                 result,
                                 scan_op);
#endif // __THRUST_HAS_HIPRT__
  return ret;
}

template <class Derived,
        class InputIt,
        class OutputIt,
        class ScanOp>
OutputIt __host__ __device__
inclusive_scan(execution_policy<Derived> &policy,
               InputIt                    first,
               InputIt                    last,
               OutputIt                   result,
               ScanOp                     scan_op)
{
  int num_items = static_cast<int>(thrust::distance(first, last));
  return hip_rocprim::inclusive_scan_n(policy, first, num_items, result, scan_op);
}

template <class Derived,
        class InputIt,
        class OutputIt>
OutputIt __host__ __device__
inclusive_scan(execution_policy<Derived> &policy,
               InputIt                    first,
               OutputIt                   last,
               OutputIt                   result)
{

  typedef typename thrust::detail::eval_if<
          thrust::detail::is_output_iterator<OutputIt>::value,
          thrust::iterator_value<InputIt>,
          thrust::iterator_value<OutputIt> >::type result_type;
  return inclusive_scan(policy, first, last, result, plus<result_type>());
}

__thrust_exec_check_disable__
template <class Derived,
          class InputIt,
          class Size,
          class OutputIt,
          class T,
          class ScanOp>
OutputIt __host__ __device__
exclusive_scan_n(execution_policy<Derived> &policy,
                 InputIt                    first,
                 Size                       num_items,
                 OutputIt                   result,
                 T                          init,
                 ScanOp                     scan_op)
{
  OutputIt ret = result;
  THRUST_HIP_PRESERVE_KERNELS_WORKAROUND((
    rocprim::exclusive_scan<rocprim::default_config, InputIt, OutputIt, T, ScanOp>
  ));
#if __THRUST_HAS_HIPRT__

  ret = __scan::exclusive_scan(policy,
                               first,
                               result,
                               num_items,
                               init,
                               scan_op);
#else // __THRUST_HAS_HIPRT__
  ret = thrust::exclusive_scan(cvt_to_seq(derived_cast(policy)),
                               first,
                               first + num_items,
                               result,
                               init,
                               scan_op);
#endif // __THRUST_HAS_HIPRT__
  return ret;
}

template <class Derived,
        class InputIt,
        class OutputIt,
        class T,
        class ScanOp>
OutputIt __host__ __device__
exclusive_scan(execution_policy<Derived> &policy,
               InputIt                    first,
               InputIt                    last,
               OutputIt                   result,
               T                          init,
               ScanOp                   scan_op)
{
  int num_items = static_cast<int>(thrust::distance(first, last));
  return hip_rocprim::exclusive_scan_n(policy, first, num_items, result, init, scan_op);
}

template <class Derived,
        class InputIt,
        class OutputIt,
        class T>
OutputIt __host__ __device__
exclusive_scan(execution_policy<Derived> &policy,
               InputIt                    first,
               OutputIt                   last,
               OutputIt                   result,
               T                          init)
{
  return exclusive_scan(policy, first, last, result, init, plus<T>());
}

template <class Derived,
        class InputIt,
        class OutputIt>
OutputIt __host__ __device__
exclusive_scan(execution_policy<Derived> &policy,
               InputIt                    first,
               OutputIt                   last,
               OutputIt                   result)
{
  typedef typename thrust::detail::eval_if<
          thrust::detail::is_output_iterator<OutputIt>::value,
          thrust::iterator_value<InputIt>,
          thrust::iterator_value<OutputIt>
  >::type result_type;
  return exclusive_scan(policy, first, last, result, result_type(0));
}

} // namespace  hip_rocprim

END_NS_THRUST

#include <thrust/scan.h>

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
