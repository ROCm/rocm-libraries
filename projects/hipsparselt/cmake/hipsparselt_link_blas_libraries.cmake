# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# .rst: Links the appropriate BLAS and LAPACK libraries to a target based on the build
# configuration.
#
# ``hipsparselt_link_blas_libraries(<target>)``
#
# Links BLAS/LAPACK libraries to <target> based on the following options:
#
# * If HIPSPARSELT_ENABLE_THEROCK is ON: Links with TheRock's cblas package
# * If HIPSPARSELT_ENABLE_BLIS is ON and BLIS is found: Links with BLIS and LAPACK
# * Otherwise: Links with standard LAPACK libraries (includes BLAS)
#
# Assumes find_package() calls have been made prior to invocation.
function(hipsparselt_link_blas_libraries target_name)
    if(HIPSPARSELT_ENABLE_THEROCK)
        # TheRock's cblas package provides OpenBLAS which includes BLAS, CBLAS, and LAPACK
        target_link_libraries(${target_name} PRIVATE cblas)
        message(STATUS "Linking ${target_name} with TheRock OpenBLAS (${OpenBLAS_DIR})")
        return()
    endif()

    # Prefer an imported LAPACK target if available; otherwise fall back to LAPACK_LIBRARIES.
    set(_hipsparselt_lapack_link "")
    if(TARGET LAPACK::LAPACK)
        set(_hipsparselt_lapack_link LAPACK::LAPACK)
    elseif(LAPACK_LIBRARIES)
        set(_hipsparselt_lapack_link ${LAPACK_LIBRARIES})
    else()
        message(FATAL_ERROR "LAPACK is required for hipsparselt clients but neither LAPACK::LAPACK nor LAPACK_LIBRARIES is available")
    endif()

    if(HIPSPARSELT_ENABLE_BLIS AND BLIS_FOUND)
        target_link_libraries(${target_name} PRIVATE BLIS::blis ${_hipsparselt_lapack_link})
        message(STATUS "Linking ${target_name} with BLIS and LAPACK")
    else()
        target_link_libraries(${target_name} PRIVATE ${_hipsparselt_lapack_link})
        message(STATUS "Linking ${target_name} with LAPACK (includes BLAS)")
    endif()
endfunction()
