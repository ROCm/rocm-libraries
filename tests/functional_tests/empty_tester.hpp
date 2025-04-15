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

#ifndef ROCSHMEM_CLIENTS_FUNCTIONAL_TESTS_EMPTY_TESTER_HPP
#define ROCSHMEM_CLIENTS_FUNCTIONAL_TESTS_EMPTY_TESTER_HPP

#include "tester.hpp"

/******************************************************************************
 * HOST TESTER CLASS
 *****************************************************************************/
class EmptyTester : public Tester {
 public:
  explicit EmptyTester(TesterArguments args);
  virtual ~EmptyTester();

 protected:
  virtual void resetBuffers(uint64_t size) override;

  virtual void launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                            uint64_t size) override;

  virtual void verifyResults(uint64_t size) override;
};

#endif  // ROCSHMEM_CLIENTS_FUNCTIONAL_TESTS_EMPTY_TESTER_HPP
