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

#include <rocshmem/rocshmem.hpp>
#include <vector>

#include "tester.hpp"
#include "tester_arguments.hpp"

using namespace rocshmem;

int main(int argc, char *argv[]) {
  /**
   * Setup the tester arguments.
   */
  TesterArguments args(argc, argv);

  /***
   * Select a GPU
   */
  int rank = rocshmem_my_pe();
  int ndevices, my_device = 0;
  CHECK_HIP(hipGetDeviceCount(&ndevices));
  my_device = rank % ndevices;
  CHECK_HIP(hipSetDevice(my_device));

  /**
   * Must initialize rocshmem to access arguments needed by the tester.
   */
  rocshmem_init();

  /**
   * Now grab the arguments from rocshmem.
   */
  args.get_rocshmem_arguments();

  /**
   * Using the arguments we just constructed, call the tester factory
   * method to get the tester (specified by the arguments).
   */
  std::vector<Tester *> tests = Tester::create(args);

  /**
   * Run the tests
   */
  for (auto test : tests) {
    test->execute();

    /**
     * The tester factory method news the tester to create it so we clean
     * up the memory here.
     */
    delete test;
  }

  /**
   * The rocshmem library needs to be cleaned up with this call. It pairs
   * with the init function above.
   */
  rocshmem_finalize();

  return 0;
}
