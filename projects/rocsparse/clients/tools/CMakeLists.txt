# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

foreach(app rocsparseio-info rocsparseio-convert)
    add_executable(${app}
        ${app}.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/rocsparseio.cpp
    )

    target_compile_features(${app} PRIVATE cxx_std_17)

    target_include_directories(${app} PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../common>
    )

    target_link_libraries(${app} PRIVATE hip::host)

    if(ROCSPARSE_ENABLE_ROCBLAS AND rocblas_FOUND)
        target_compile_definitions(${app} PRIVATE ROCSPARSE_WITH_ROCBLAS)
    endif()

    if(ROCSPARSE_ENABLE_ASAN)
        rocsparse_target_configure_sanitizers(${app} PRIVATE)
    endif()

    # Install tools
    if(WIN32)
        rocm_install(PROGRAMS $<TARGET_FILE:${app}>
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            COMPONENT clients
        )
    else()
        rocm_install(PROGRAMS $<TARGET_FILE:${app}>
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/rocsparse
            COMPONENT clients
        )
    endif()
endforeach()
