/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <hip/hip_runtime.h>

#include <thrust/detail/config.h>
#include <thrust/system/hip/detail/execution_policy.h>
#include <thrust/system/hip/detail/par.h>

// pass
// ----------------
#include <thrust/system/hip/detail/adjacent_difference.h>
#include <thrust/system/hip/detail/copy.h>
#include <thrust/system/hip/detail/copy_if.h>
#include <thrust/system/hip/detail/count.h>
#include <thrust/system/hip/detail/equal.h>
#include <thrust/system/hip/detail/extrema.h>
#include <thrust/system/hip/detail/fill.h>
#include <thrust/system/hip/detail/find.h>
#include <thrust/system/hip/detail/for_each.h>
#include <thrust/system/hip/detail/gather.h>
#include <thrust/system/hip/detail/generate.h>
#include <thrust/system/hip/detail/inner_product.h>
#include <thrust/system/hip/detail/mismatch.h>
#include <thrust/system/hip/detail/partition.h>
#include <thrust/system/hip/detail/reduce_by_key.h>
#include <thrust/system/hip/detail/remove.h>
#include <thrust/system/hip/detail/replace.h>
#include <thrust/system/hip/detail/reverse.h>
#include <thrust/system/hip/detail/scatter.h>
#include <thrust/system/hip/detail/swap_ranges.h>
#include <thrust/system/hip/detail/tabulate.h>
#include <thrust/system/hip/detail/transform.h>
#include <thrust/system/hip/detail/transform_reduce.h>
#include <thrust/system/hip/detail/transform_scan.h>
#include <thrust/system/hip/detail/uninitialized_copy.h>
#include <thrust/system/hip/detail/uninitialized_fill.h>
#include <thrust/system/hip/detail/unique.h>
#include <thrust/system/hip/detail/unique_by_key.h>

// fail
// ----------------
// fails with mixed types
#include <thrust/system/hip/detail/reduce.h>

// mixed types are not compiling, commented in testing/scan.cu
#include <thrust/system/hip/detail/scan.h>

// // stubs passed
// // ----------------
#include <thrust/system/hip/detail/binary_search.h>
#include <thrust/system/hip/detail/merge.h>
#include <thrust/system/hip/detail/scan_by_key.h>
#include <thrust/system/hip/detail/set_operations.h>
#include <thrust/system/hip/detail/sort.h>

/*! \file thrust/system/hip/execution_policy.h
 *  \brief Execution policies for Thrust's hip system.
 */

#if 0
namespace thrust
{
namespace system
{
/*! \addtogroup system_backends Systems
 *  \ingroup system
 *  \{
 */

/*! \namespace thrust::system::hip
 *  \brief \p thrust::system::hip is the namespace containing functionality for allocating, manipulating,
 *         and deallocating memory available to Thrust's hip backend system.
 *         The identifiers are provided in a separate namespace underneath <tt>thrust::system</tt>
 *         for import convenience but are also aliased in the top-level <tt>thrust::hip</tt>
 *         namespace for easy access.
 *
 */
namespace hip
{

/*! \addtogroup execution_policies
 *  \{
 */


/*! \p thrust::hip::execution_policy is the base class for all Thrust parallel execution
 *  policies which are derived from Thrust's hip backend system.
 */
template<typename DerivedPolicy>
struct execution_policy : thrust::execution_policy<DerivedPolicy>
{};


/*! \p hip::tag is a type representing Thrust's hip backend system in C++'s type system.
 *  Iterators "tagged" with a type which is convertible to \p hip::tag assert that they may be
 *  "dispatched" to algorithm implementations in the \p hip system.
 */
struct tag : thrust::system::hip::execution_policy<tag> { unspecified };


/*! \p thrust::hip::par is the parallel execution policy associated with Thrust's hip
 *  backend system.
 *
 *  Instead of relying on implicit algorithm dispatch through iterator system tags, users may
 *  directly target Thrust's hip backend system by providing \p thrust::hip::par as an algorithm
 *  parameter.
 *
 *  Explicit dispatch can be useful in avoiding the introduction of data copies into containers such
 *  as \p thrust::hip::vector.
 *
 *  The type of \p thrust::hip::par is implementation-defined.
 *
 *  The following code snippet demonstrates how to use \p thrust::hip::par to explicitly dispatch an
 *  invocation of \p thrust::for_each to the hip backend system:
 *
 *  \code
 *  #include <thrust/for_each.h>
 *  #include <thrust/system/hip/execution_policy.h>
 *  #include <cstdio>
 *
 *  struct printf_functor
 *  {
 *    __host__ __device__
 *    void operator()(int x)
 *    {
 *      printf("%d\n", x);
 *    }
 *  };
 *  ...
 *  int vec[3];
 *  vec[0] = 0; vec[1] = 1; vec[2] = 2;
 *
 *  thrust::for_each(thrust::hip::par, vec.begin(), vec.end(), printf_functor());
 *
 *  // 0 1 2 is printed to standard output in some unspecified order
 *  \endcode
 *
 *  Explicit dispatch may also be used to direct Thrust's hip backend to launch hip kernels implementing
 *  an algorithm invocation on a particular hip stream. In some cases, this may achieve concurrency with the
 *  caller and other algorithms and hip kernels executing on a separate hip stream. The following code
 *  snippet demonstrates how to use the \p thrust::hip::par execution policy to explicitly dispatch invocations
 *  of \p thrust::for_each on separate hip streams:
 *
 *  \code
 *  #include <thrust/for_each.h>
 *  #include <thrust/system/hip/execution_policy.h>
 *
 *  struct printf_functor
 *  {
 *    hipStream_t s;
 *
 *    printf_functor(hipStream_t s) : s(s) {}
 *
 *    __host__ __device__
 *    void operator()(int)
 *    {
 *      printf("Hello, world from stream %p\n", static_cast<void*>(s));
 *    }
 *  };
 *
 *  int main()
 *  {
 *    // create two hip streams
 *    hipStream_t s1, s2;
 *    hipStreamCreate(&s1);
 *    hipStreamCreate(&s2);
 *  
 *    thrust::counting_iterator<int> iter(0);
 *  
 *    // execute for_each on two different streams
 *    thrust::for_each(thrust::hip::par.on(s1), iter, iter + 1, printf_functor(s1));
 *    thrust::for_each(thrust::hip::par.on(s2), iter, iter + 1, printf_functor(s2));
 *  
 *    // synchronize with both streams
 *    hipStreamSynchronize(s1);
 *    hipStreamSynchronize(s2);
 *  
 *    // destroy streams
 *    hipStreamDestroy(s1);
 *    hipStreamDestroy(s2);
 *  
 *    return 0;
 *  }
 *  \endcode
 *
 *  Even when using hip streams with \p thrust::hip::par.on(), there is no guarantee of concurrency. Algorithms
 *  which return a data-dependent result or whose implementations require temporary memory allocation may
 *  cause blocking synchronization events. Moreover, it may be necessary to explicitly synchronize through
 *  \p hipStreamSynchronize or similar before any effects induced through algorithm execution are visible to
 *  the rest of the system. Finally, it is the responsibility of the caller to own the lifetime of any hip
 *  streams involved.
 */
static const unspecified par;


/*! \}
 */


} // end hip
} // end system
} // end thrust
#endif
