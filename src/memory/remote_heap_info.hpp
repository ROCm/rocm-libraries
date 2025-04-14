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

#ifndef LIBRARY_SRC_MEMORY_REMOTE_HEAP_INFO_HPP_
#define LIBRARY_SRC_MEMORY_REMOTE_HEAP_INFO_HPP_

#include <hip/hip_runtime_api.h>
#include <mpi.h>

#include <cassert>
#include <vector>

#include "hip_allocator.hpp"
#include "window_info.hpp"

/**
 * @file remote_heap_info.hpp
 *
 * @brief Contains information about remote processing elements' heaps
 */

namespace rocshmem {

class CommunicatorMPI {
 public:
  /**
   * @brief Default constructor
   */
  CommunicatorMPI() = default;

  /**
   * @brief Primary constructor
   */
  CommunicatorMPI(char* heap_base, size_t heap_size,
                  MPI_Comm comm = MPI_COMM_WORLD)
    : comm_{comm} {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
      int provided;
      MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    }
    MPI_Comm_rank(comm_, &my_pe_);
    MPI_Comm_size(comm_, &num_pes_);

    heap_window_info_ = WindowInfo(comm_, heap_base, heap_size);
  }

  /**
   * @brief Destructor
   */
  ~CommunicatorMPI() {}

  /**
   * @brief Returns my processing element ID
   */
  int my_pe() { return my_pe_; }

  /**
   * @brief Returns number of processing elements
   */
  int num_pes() { return num_pes_; }

  /**
   * @brief Performs MPI_Barrier
   */
  void barrier() { MPI_Barrier(comm_); }

  /**
   * @brief Performs MPI_Allgather on recvbuf
   */
  void allgather(void* recvbuf) {
    MPI_Allgather(MPI_IN_PLACE, sizeof(void*), MPI_CHAR, recvbuf,
                  sizeof(void*), MPI_CHAR, comm_);
  }

  /**
   * @brief Accessor method for heap_window_info_
   */
  WindowInfo* get_window_info() { return &heap_window_info_; }

 private:
  /**
   * @brief Identifier for this processing element
   */
  MPI_Comm comm_{};

  /**
   * @brief Identifier for this processing element
   */
  int my_pe_{-1};

  /**
   * @brief The total number of processing elements
   */
  int num_pes_{-1};

  /**
   * @brief MPI window on the symmetric GPU heap
   */
  WindowInfo heap_window_info_{};
};

template <typename COMMUNICATOR_T>
class RemoteHeapInfo {
  /**
   * @brief Helper type for heap_bases_ member
   */
  using HEAP_BASES_T = std::vector<char*, StdAllocatorHIP<char*>>;

 public:
  /**
   * @brief Required for default construction of other objects
   *
   * @note Not intended for direct usage
   */
  RemoteHeapInfo() = default;

  /**
   * @brief Primary constructor
   *
   * @param[in] The identifier for this processing element
   * @param[in] The total number of processing elements
   */
  RemoteHeapInfo(char* heap_ptr, size_t heap_size,
                 MPI_Comm comm = MPI_COMM_WORLD)
    : communicator_{heap_ptr, heap_size, comm} {
    heap_bases_.resize(communicator_.num_pes());
    for (auto& base : heap_bases_) {
      base = nullptr;
    }
    heap_bases_[communicator_.my_pe()] = heap_ptr;
    communicator_.allgather(heap_bases_.data());

    device_heap_bases_ = heap_bases_.data();
  }

  /**
   * @brief Invoke barrier on communicator
   */
  void barrier() { communicator_.barrier(); }

  /**
   * @brief Accessor for my processing element identifier
   *
   * @return My processing element identifier
   */
  int my_pe() { return communicator_.my_pe(); }

  /**
   * @brief Accessor for number of processing elements
   *
   * @return Number of processing elements
   */
  int num_pes() { return communicator_.num_pes(); }

  /**
   * @brief Accessor method for heap_window_info_
   *
   * @return Window info for Reverse Offload interface
   */
  auto get_window_info() { return communicator_.get_window_info(); }

  /**
   * @brief Accessor for heap bases
   *
   * @return Vector containing the addresses of the symmetric heap bases
   */
  __host__ const HEAP_BASES_T& get_heap_bases() { return heap_bases_; }

  /**
   * @brief Accessor for heap bases
   *
   * @return Vector containing the addresses of the symmetric heap bases
   */
  __device__ auto get_heap_bases() { return device_heap_bases_; }

 private:
  /**
   * @brief Communicator implementation
   */
  COMMUNICATOR_T communicator_{};

  /**
   * @brief C-array of symmetric heap base pointers
   *
   * A vector of char* pointers corresponding to the heap_bases virtual
   * address for all processing element that we can communicate with.
   *
   * @note Used by __host__ code
   */
  HEAP_BASES_T heap_bases_{};

  /**
   * @brief C-array of symmetric heap base pointers
   *
   * A vector of char* pointers corresponding to the heap_bases virtual
   * address for all processing element that we can communicate with.
   *
   * @note Used by __device__ code
   */
  char** device_heap_bases_{nullptr};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_REMOTE_HEAP_INFO_HPP_
