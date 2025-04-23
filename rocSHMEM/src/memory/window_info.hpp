/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_MEMORY_WINDOW_INFO_HPP_
#define LIBRARY_SRC_MEMORY_WINDOW_INFO_HPP_

#include <mpi.h>

#include <cassert>
#include <memory>

/**
 * @file window_info.hpp
 *
 * @brief Contains information about symmetric heaps' windows
 */

namespace rocshmem {

class WindowInfo {
 public:
  /**
   * @brief Default constructor
   */
  WindowInfo() = default;

  /**
   * @brief Primary constructor
   */
  WindowInfo(MPI_Comm comm, void* start, size_t size)
      : comm_{comm},
        win_start_{start},
        win_end_{reinterpret_cast<char*>(start) + size} {
    up_win_ = std::unique_ptr<MPI_Win>(new MPI_Win);
    MPI_Win_create(win_start_, size, 1, MPI_INFO_NULL, comm_, up_win_.get());
    MPI_Win_lock_all(MPI_MODE_NOCHECK, *up_win_.get());
  }

  /**
   * @brief Destructor
   */
  ~WindowInfo() {
    if (up_win_) {
      MPI_Win_unlock_all(*up_win_.get());
      MPI_Win_free(up_win_.get());
    }
  }

  /**
   * @brief Copy constructor
   *
   * @note Disabled due to up_win_
   */
  WindowInfo(WindowInfo& other) = delete;  // NOLINT

  /**
   * @brief Const copy constructor
   *
   * @note Disabled due to up_win_
   */
  WindowInfo(const WindowInfo& other) = delete;

  /**
   * @brief Copy assignment
   *
   * @note Disabled due to up_win_
   */
  WindowInfo& operator=(WindowInfo other) = delete;

  /**
   * @brief Move constructor
   */
  WindowInfo(WindowInfo&& other) = default;

  /**
   * @brief Move assignment
   */
  WindowInfo& operator=(WindowInfo&& other) = default;

  /**
   * @brief Accessor for object in up_win_
   *
   * @return MPI_Win object
   */
  MPI_Win get_win() const { return *up_win_.get(); }

  /**
   * @brief Accessor for win_start_
   *
   * @return Raw start pointer
   */
  void* get_start() const { return win_start_; }

  /**
   * @brief Accessor for win_end_
   *
   * @return Raw end pointer
   */
  void* get_end() const { return win_end_; }

  /**
   * @brief Setter for object in up_win_
   *
   * @param[in] An MPI Window object
   */
  void set_win(MPI_Win win) { *up_win_.get() = win; }

  /**
   * @brief Setter for win_start_
   *
   * @param[in] Start raw pointer
   */
  void set_start(void* start) { win_start_ = start; }

  /**
   * @brief Setter for win_end_
   *
   * @param[in] End raw pointer
   */
  void set_end(void* end) { win_end_ = end; }

  /**
   * @brief Get offset between address and start of window
   *
   * @param[in] Address in raw pointer format
   *
   * @return Difference between dest and window start
   */
  MPI_Aint get_offset(const void* dest) {
    assert(reinterpret_cast<char*>(const_cast<void*>(dest)) >=
           reinterpret_cast<char*>(win_start_));
    assert(reinterpret_cast<char*>(const_cast<void*>(dest)) >=
           reinterpret_cast<char*>(win_start_));
    assert(reinterpret_cast<char*>(const_cast<void*>(dest)) <
           reinterpret_cast<char*>(win_end_));

    MPI_Aint dest_disp;
    MPI_Get_address(dest, &dest_disp);
    MPI_Aint start_disp;
    MPI_Get_address(win_start_, &start_disp);

    return MPI_Aint_diff(dest_disp, start_disp);
  }

 private:
  /**
   * @brief MPI Communicator
   */
  MPI_Comm comm_{};

  /**
   * @brief Owning pointer to MPI_Win
   *
   * The pointer is used to track which object is responsible for
   * releasing window resources during class destruction.
   *
   * This becomes an issue if objects of this class are move constructed
   * or move assigned.
   */
  std::unique_ptr<MPI_Win> up_win_{nullptr};

  /**
   * @brief Raw pointer marking the start of window
   */
  void* win_start_{nullptr};

  /**
   * @brief Raw pointer marking the end of window
   */
  void* win_end_{nullptr};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_WINDOW_INFO_HPP_
