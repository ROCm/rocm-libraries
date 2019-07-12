# Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

function(print_configuration_summary)
  message(STATUS "******** Summary ********")
  message(STATUS "General:")
  message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
  message(STATUS "  HIP ROOT              : ${HIP_ROOT_DIR}")
  message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
  message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")
  string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS_STRIP)
  message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS_STRIP}")
  message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
  message(STATUS "  Install prefix        : ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "")
  message(STATUS "  BUILD_TEST            : ${BUILD_TEST}")
endfunction()
