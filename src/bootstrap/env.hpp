// Copyright (c) Microsoft Corporation.
// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc,
// Licensed under the MIT license.

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
