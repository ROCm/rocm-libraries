/******************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Modifications Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCSHMEM_ENV_HPP_
#define ROCSHMEM_ENV_HPP_

#include <memory>
#include <string>

namespace rocshmem {

class Env;

/// Get the environment.
/// @return A reference to the global environment object.
std::shared_ptr<Env> env();

/// The constructor reads environment variables and sets the corresponding fields.
/// Use the @ref env() function to get the environment object.
class Env {
 public:
  const std::string debug;
  const std::string debugSubsys;
  const std::string debugFile;
  const std::string hostid;
  const std::string socketFamily;
  const std::string socketIfname;

 private:
  Env();

  friend std::shared_ptr<Env> env();
};

}  // namespace rocshmem

#endif  // ROCSHMEM_ENV_HPP_
