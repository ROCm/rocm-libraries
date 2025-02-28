/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

// for demangling the result of type_info.name()
// with msvc, type_info.name() is already demangled
#ifdef __GNUC__
#include <cxxabi.h>
#endif // __GNUC__

#include <string>
#include <cstdlib>

namespace unittest
{

#if __GNUC__ && !_NVHPC_CUDA
inline std::string demangle(const char* name)
{
  int status = 0;
  char* realname = abi::__cxa_demangle(name, 0, 0, &status);
  std::string result(realname);
  std::free(realname);

  return result;
}
#else
inline std::string demangle(const char* name)
{
  return name;
}
#endif

} // end unittest

