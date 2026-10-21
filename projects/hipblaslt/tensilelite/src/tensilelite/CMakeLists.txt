# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_library(tensilelite-jit)
add_library(roc::tensilelite-jit ALIAS tensilelite-jit)

set_target_properties(tensilelite-jit
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)
rocm_set_soversion(tensilelite-jit "1.0")

target_compile_features(tensilelite-jit PUBLIC cxx_std_17)

target_compile_definitions(tensilelite-jit PRIVATE TENSILELITE_BUILDING_LIBRARY)
if(BUILD_SHARED_LIBS AND WIN32)
    # Without declspec annotations a Windows DLL exports nothing.
    target_compile_definitions(tensilelite-jit PUBLIC TENSILELITE_ENABLE_WINDOWS_DLL_DECLSPEC)
endif()

target_sources(tensilelite-jit
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/Compiler.cpp"
)

target_sources(tensilelite-jit
    PUBLIC
        FILE_SET tensilelite_jit_public_headers TYPE HEADERS
            BASE_DIRS "${PROJECT_SOURCE_DIR}/include"
            FILES
                "${PROJECT_SOURCE_DIR}/include/tensilelite/Compiler.hpp"
                "${PROJECT_SOURCE_DIR}/include/tensilelite/Export.hpp"
)

target_include_directories(tensilelite-jit
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(tensilelite-jit PRIVATE amd_comgr)

install(TARGETS tensilelite-jit
    EXPORT tensilelite-targets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT ${TENSILELITE_RUNTIME_COMPONENT}
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT ${TENSILELITE_RUNTIME_COMPONENT}
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT ${TENSILELITE_DEVEL_COMPONENT}
    FILE_SET tensilelite_jit_public_headers
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT ${TENSILELITE_DEVEL_COMPONENT}
)

if(BUILD_SHARED_LIBS AND NOT WIN32)
    # amd_comgr is installed next to this library.
    set_property(TARGET tensilelite-jit PROPERTY INSTALL_RPATH "$ORIGIN")
endif()

if(TENSILELITE_ENABLE_JIT_PYTHON)
    add_subdirectory(bindings)
endif()
