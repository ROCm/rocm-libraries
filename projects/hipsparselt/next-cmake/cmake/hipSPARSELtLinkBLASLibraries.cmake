# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(hipsparselt_link_blas_libraries target_name)
    if(HIPSPARSELT_ENABLE_BLIS AND BLIS_FOUND)
        target_link_libraries(${target_name} PRIVATE BLIS::blis)
        message(STATUS "Linking ${target_name} with BLIS")
    else()
        target_link_libraries(${target_name} PRIVATE ${BLAS_LIBRARIES} ${CBLAS_LIBRARIES})
        message(STATUS "Linking ${target_name} with standard BLAS/CBLAS")
    endif()

    target_link_libraries(${target_name} PRIVATE ${LAPACK_LIBRARIES})
endfunction()
