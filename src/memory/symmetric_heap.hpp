/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_MEMORY_SYMMETRIC_HEAP_HPP_
#define LIBRARY_SRC_MEMORY_SYMMETRIC_HEAP_HPP_

/**
 * @file symmetric_heap.hpp
 *
 * @brief Contains a symmetric heap
 *
 * Every processing element allocates a symmetric heap. (For details on
 * the symmetric heap, refer to the OpenSHMEM specification.)
 *
 * The symmetric heap is allocated using fine-grained memory to allow
 * both host access and device access to the memory space.
 *
 * The symmetric heaps are visible to network by registering them as
 * memory regions. Every memory region has a remote key
 * which needs to be shared across the network (to access the memory
 * region).
 */

#include <hip/hip_runtime_api.h>

#include "remote_heap_info.hpp"
#include "single_heap.hpp"

namespace rocshmem {

class SymmetricHeap {
  /**
   * @brief Helper type for RemoteHeapInfo with MPI
   */
  using RemoteHeapInfoType = RemoteHeapInfo<CommunicatorMPI>;

 public:
  /**
   * @brief Allocates heap memory and returns ptr to caller
   *
   * @param[in,out] A pointer to memory handle
   * @param[in] Number of bytes of requested
   */
  void malloc(void** ptr, size_t size) { single_heap_.malloc(ptr, size); }

  /**
   * @brief Frees previously allocated network visible memory
   *
   * @param[in] Handle of previously allocated memory
   */
  void free(void* ptr) { single_heap_.free(ptr); }

  /**
   * @brief Accessor for local heap base
   *
   * @return Base address of the local symmetric heap
   */
  __host__ char* get_local_heap_base() { return single_heap_.get_base_ptr(); }

  /**
   * @brief Accessor method for heap size
   */
  auto get_size() { return single_heap_.get_size(); }

  /**
   * @brief Accessor method for heap_window_info_
   */
  auto get_window_info() { return remote_heap_info_.get_window_info(); }

  /**
   * @brief Accessor for heap bases
   *
   * @return Vector containing the addresses of the symmetric heap bases
   */
  __host__ const auto& get_heap_bases() {
    return remote_heap_info_.get_heap_bases();
  }

  /**
   * @brief Accessor for heap bases
   *
   * @return Vector containing the addresses of the symmetric heap bases
   */
  __device__ auto get_heap_bases() {
    return remote_heap_info_.get_heap_bases();
  }

  /**
   * @brief Returns is the heap is allocated with managed memory
   *
   * @return bool
   */
  bool is_managed() { return single_heap_.is_managed(); }

 private:
  /**
   * @brief Processing element's implementation of heap
   */
  SingleHeap single_heap_{};

  /**
   * @brief Implementation of remote heaps
   */
  RemoteHeapInfoType remote_heap_info_{single_heap_.get_base_ptr(),
                                       single_heap_.get_size()};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_SYMMETRIC_HEAP_HPP_
