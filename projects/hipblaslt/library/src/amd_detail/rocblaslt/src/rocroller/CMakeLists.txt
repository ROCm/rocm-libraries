# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_library(hipblaslt-rocroller STATIC)

string(REGEX MATCHALL "gfx[a-z0-9]+" _rocroller_archs "${GPU_TARGETS}")
if("gfx950" IN_LIST _rocroller_archs)
    add_subdirectory(custom_kernels)
endif()

target_sources(hipblaslt-rocroller
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/custom_kernels.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/gemm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/parameter_selection.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/runtime_args_selection.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocroller_host.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/solution_cache.cpp"
)

target_compile_features(hipblaslt-rocroller PRIVATE cxx_std_20)
set_target_properties(hipblaslt-rocroller PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hipblaslt-rocroller PRIVATE roc::origami-headers)
if(TARGET roc::rocroller-interface)
    target_link_libraries(hipblaslt-rocroller PRIVATE roc::rocroller-interface)
else()
    target_link_libraries(hipblaslt-rocroller PRIVATE roc::rocroller_interface)
endif()
target_link_libraries(hipblaslt-rocroller PRIVATE hip::host roc::hipblas-common)
# This is the only target whose sources are built against rocRoller's headers.
# rocroller_interop.hpp is included only from those sources. The ranker does
# not define this macro and does not include rocRoller.
target_compile_definitions(hipblaslt-rocroller PRIVATE HIPBLASLT_USE_ROCROLLER)
target_include_directories(hipblaslt-rocroller
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
        "${PROJECT_BINARY_DIR}/library/include"
        "${PROJECT_SOURCE_DIR}/library/include"
        "${PROJECT_SOURCE_DIR}/tensilelite/include"
)
if(HIPBLASLT_ENABLE_COVERAGE)
    # These sources land inside hipblaslt via the WHOLE_ARCHIVE link below, so
    # without their own instrumentation they are missing from the coverage
    # report even when the tests execute them. Host-only translation units, so
    # no -Xarch_host selector is needed.
    target_compile_options(hipblaslt-rocroller PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(hipblaslt-rocroller PUBLIC -fprofile-instr-generate)
endif()

target_link_libraries(hipblaslt PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,hipblaslt-rocroller>)

if(NOT BUILD_SHARED_LIBS)
    install(TARGETS hipblaslt-rocroller
        EXPORT hipblaslt-targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT devel)
endif()
