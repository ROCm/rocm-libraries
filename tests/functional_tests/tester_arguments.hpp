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

#ifndef _TESTER_ARGUMENTS_HPP_
#define _TESTER_ARGUMENTS_HPP_

#include <climits>
#include <cstdint>
#include <rocshmem/rocshmem.hpp>
#include <string>
#include <iostream>

class TesterArguments {
 public:
  TesterArguments(int argc, char *argv[]);

  /**
   * Initialize rocshmem members
   * Valid after rocshmem_init function called.
   */
  void get_rocshmem_arguments();

 private:
  /**
   * Output method which displays available command line options
   */
  static void show_usage(std::string executable_name);

 public:
  /**
   * Arguments obtained from command line
   */
  unsigned num_wgs = 1;
  unsigned num_threads = 1;
  unsigned algorithm = 0;
  uint64_t min_msg_size = 1;
  uint64_t max_msg_size = 1 << 20;
  unsigned wg_size = 64;
  unsigned thread_access = 64;
  unsigned coal_coef = 64;
  unsigned op_type = 0;
  unsigned shmem_context = rocshmem::ROCSHMEM_CTX_WG_PRIVATE;

  /**
   * Arguments obtained from rocshmem
   */
  int numprocs = INT_MAX;
  int myid = INT_MAX;

  /**
   * Defaults tester values
   */
  int loop = 10;
  int skip = 10;
  int loop_large = 10;
  uint64_t large_message_size = 32768;
};

#endif
