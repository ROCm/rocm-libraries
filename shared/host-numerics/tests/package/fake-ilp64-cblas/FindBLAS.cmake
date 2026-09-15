# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(BLAS_FOUND TRUE)
set(BLAS_LIBRARIES BLAS::BLAS)
if(NOT TARGET BLAS::BLAS)
    add_library(BLAS::BLAS INTERFACE IMPORTED)
endif()
