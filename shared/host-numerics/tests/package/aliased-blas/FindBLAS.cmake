# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Reproduce superprojects such as TheRock that expose the semantic BLAS target
# as an alias to their provider-specific imported target.
find_library(ALIASED_BLAS_LIBRARY NAMES openblas blas REQUIRED)

if(NOT TARGET AliasedBLAS::BLAS)
    add_library(AliasedBLAS::BLAS UNKNOWN IMPORTED)
    set_target_properties(
        AliasedBLAS::BLAS
        PROPERTIES
            IMPORTED_LOCATION "${ALIASED_BLAS_LIBRARY}"
    )
endif()
if(NOT TARGET BLAS::BLAS)
    add_library(BLAS::BLAS ALIAS AliasedBLAS::BLAS)
endif()

set(BLAS_LINKER_FLAGS)
set(BLAS_LIBRARIES AliasedBLAS::BLAS)
set(BLAS_FOUND TRUE)
