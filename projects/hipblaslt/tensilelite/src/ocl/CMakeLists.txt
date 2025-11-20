# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_library(TensileOcl STATIC
            OclHardware.cpp
            OclSolutionAdapter.cpp
            OclUtils.cpp)

message( TensileOcl SYSTEM ${HIP_INCLUDE_DIRS} ${HCC_INCLUDE_DIRS} )

include_directories(${HIP_INCLUDE_DIRS} ${HCC_INCLUDE_DIRS})
