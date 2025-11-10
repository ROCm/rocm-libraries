# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

foreach(app rocsparseio-info rocsparseio-convert)
    add_executable(${app}
        ${app}.cpp
        ${PROJECT_SOURCE_DIR}/clients/common/rocsparseio.cpp
    )

    target_compile_features(${app} PRIVATE cxx_std_17)

    target_include_directories(${app} PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/clients/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/clients/common>
    )

    target_link_libraries(${app} PRIVATE hip::host)

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
