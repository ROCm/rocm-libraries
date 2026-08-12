# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_library(rocroller-host-numerics STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/HostDataGeneration.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/HostReference.cpp"
)
add_library(rocroller::host-numerics ALIAS rocroller-host-numerics)

target_sources(rocroller-host-numerics
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include/rocRoller/HostNumerics/HostDataGeneration.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/rocRoller/HostNumerics/HostReference.hpp"
)
target_include_directories(rocroller-host-numerics
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_link_libraries(rocroller-host-numerics
    PUBLIC
        roc::rocroller-interface
        roc::host-validation-core
    PRIVATE
        roc::host-validation
        roc::host-validation-blas
        roc::host-validation-mx
)

if(ROCROLLER_ENABLE_ASAN)
    rocroller_target_configure_sanitizers(rocroller-host-numerics PRIVATE)
endif()

if(ROCROLLER_BUILD_TESTING OR BUILD_TESTING)
    add_executable(rocroller-client-host-data-generation-tests
        "${CMAKE_CURRENT_SOURCE_DIR}/test/HostDataGenerationTest.cpp"
    )
    target_link_libraries(rocroller-client-host-data-generation-tests
        PRIVATE
            rocroller::host-numerics
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
            rocroller::host-numerics
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
