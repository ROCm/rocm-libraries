# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_library(rocroller-hostnumerics STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/HostDataGeneration.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/HostReference.cpp"
)
add_library(rocroller::hostnumerics ALIAS rocroller-hostnumerics)

target_sources(rocroller-hostnumerics
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include/rocRoller/hostnumerics/HostDataGeneration.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/rocRoller/hostnumerics/HostReference.hpp"
)
target_include_directories(rocroller-hostnumerics
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_link_libraries(rocroller-hostnumerics
    PUBLIC
        roc::rocroller-interface
        roc::hostnumerics-core
    PRIVATE
        roc::hostnumerics
        roc::hostnumerics-blas
)

if(ROCROLLER_ENABLE_ASAN)
    rocroller_target_configure_sanitizers(rocroller-hostnumerics PRIVATE)
endif()

if(ROCROLLER_BUILD_TESTING OR BUILD_TESTING)
    add_executable(rocroller-client-host-data-generation-tests
        "${CMAKE_CURRENT_SOURCE_DIR}/test/HostDataGenerationTest.cpp"
    )
    target_link_libraries(rocroller-client-host-data-generation-tests
        PRIVATE
            rocroller::hostnumerics
    )
    add_test(
        NAME rocroller-client-host-data-generation-tests
        COMMAND rocroller-client-host-data-generation-tests
    )

    add_executable(rocroller-client-host-reference-tests
        "${CMAKE_CURRENT_SOURCE_DIR}/test/HostReferenceTest.cpp"
    )
    target_link_libraries(rocroller-client-host-reference-tests
        PRIVATE
            rocroller::hostnumerics
    )
    add_test(
        NAME rocroller-client-host-reference-tests
        COMMAND rocroller-client-host-reference-tests
    )

    if(ROCROLLER_ENABLE_ASAN)
        rocroller_target_configure_sanitizers(
            rocroller-client-host-data-generation-tests
            PRIVATE
        )
        rocroller_target_configure_sanitizers(
            rocroller-client-host-reference-tests
            PRIVATE
        )
    endif()
endif()
