# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Bundles (integration-test-bundles/) are the CI driver. The C++ graph tests in
# this directory are the pre-bundle authoring style: they are kept buildable for
# local debugging and for bisecting against the bundle harness, but they do not
# build — and therefore do not run — unless a developer opts in on their branch.
#
# Turn them on with: cmake -DBUILD_CPP_GRAPH_TESTS=ON ...
option(BUILD_CPP_GRAPH_TESTS
    "Build the legacy C++ graph integration tests (OFF: bundles are the CI driver)"
    OFF)

set(HIPDNN_IT_CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

# The complete set of sources under this directory that build regardless of
# BUILD_CPP_GRAPH_TESTS. Paths are relative to this directory.
#
# Adding an entry here is the only way to make a new C++ test run in CI, and it
# is deliberately a central, reviewable edit — see add_always_built_test_sources()
# below. New *graph* tests do not belong here: author them as bundles under
# integration-test-bundles/ instead.
set(HIPDNN_IT_ALWAYS_BUILT_SOURCES
    IntegrationIsSupportedExtPerformance.cpp
    conv/IntegrationConvForwardSerializeRoundTrip.cpp
    layernorm/IntegrationGpuLayernormBackward.cpp
)

foreach(_rel IN LISTS HIPDNN_IT_ALWAYS_BUILT_SOURCES)
    if(NOT EXISTS "${HIPDNN_IT_CPP_ROOT}/${_rel}")
        message(FATAL_ERROR
            "${_rel} is in HIPDNN_IT_ALWAYS_BUILT_SOURCES but does not exist. "
            "Remove the stale entry.")
    endif()
endforeach()

# Registers C++ graph-verification tests. They are recorded unconditionally (so
# the orphan check below sees them) but compiled only when the developer opts in
# with BUILD_CPP_GRAPH_TESTS=ON.
function(add_cpp_graph_test_sources)
    foreach(_src IN LISTS ARGN)
        get_filename_component(_abs "${_src}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        set_property(GLOBAL APPEND PROPERTY HIPDNN_IT_REGISTERED_SOURCES "${_abs}")
    endforeach()

    if(BUILD_CPP_GRAPH_TESTS)
        target_sources(${INTEGRATION_TESTS_EXE} PRIVATE ${ARGN})
    endif()
endfunction()

# Registers C++ tests that build even with BUILD_CPP_GRAPH_TESTS=OFF. Each file
# must be listed in HIPDNN_IT_ALWAYS_BUILT_SOURCES above, so that exempting a
# test from the bundle-first rule cannot happen quietly inside a leaf directory.
function(add_always_built_test_sources)
    foreach(_src IN LISTS ARGN)
        get_filename_component(_abs "${_src}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        file(RELATIVE_PATH _rel "${HIPDNN_IT_CPP_ROOT}" "${_abs}")

        if(NOT "${_rel}" IN_LIST HIPDNN_IT_ALWAYS_BUILT_SOURCES)
            message(FATAL_ERROR
                "${_rel} is registered with add_always_built_test_sources() but is not "
                "listed in HIPDNN_IT_ALWAYS_BUILT_SOURCES in "
                "${HIPDNN_IT_CPP_ROOT}/CMakeLists.txt.\n"
                "Bundles are the CI driver: new graph coverage belongs under "
                "integration-test-bundles/, not in a new C++ test. If this file is a "
                "C++ graph test, register it with add_cpp_graph_test_sources() instead "
                "(it then builds only under -DBUILD_CPP_GRAPH_TESTS=ON). If it genuinely "
                "cannot be expressed as a bundle, add it to "
                "HIPDNN_IT_ALWAYS_BUILT_SOURCES with a comment saying why.")
        endif()

        set_property(GLOBAL APPEND PROPERTY HIPDNN_IT_REGISTERED_SOURCES "${_abs}")
    endforeach()

    target_sources(${INTEGRATION_TESTS_EXE} PRIVATE ${ARGN})
endfunction()

add_always_built_test_sources(
    IntegrationIsSupportedExtPerformance.cpp
)

add_subdirectory(batchnorm)
add_subdirectory(conv)
add_subdirectory(layernorm)
add_subdirectory(matmul)
add_subdirectory(moe)
add_subdirectory(sdpa)
add_subdirectory(reduction)
add_subdirectory(resample)
add_subdirectory(rmsnorm)
add_subdirectory(pointwise)

# Every .cpp under this directory must go through one of the two helpers above.
# Calling target_sources() directly would bypass the BUILD_CPP_GRAPH_TESTS gate
# and put a new C++ test back into CI unnoticed; a file that is added but never
# registered would silently never build at all. Both show up here as orphans.
file(GLOB_RECURSE HIPDNN_IT_ALL_SOURCES
    CONFIGURE_DEPENDS
    RELATIVE "${HIPDNN_IT_CPP_ROOT}"
    "${HIPDNN_IT_CPP_ROOT}/*.cpp"
)

get_property(HIPDNN_IT_REGISTERED GLOBAL PROPERTY HIPDNN_IT_REGISTERED_SOURCES)
set(HIPDNN_IT_REGISTERED_REL "")
foreach(_abs IN LISTS HIPDNN_IT_REGISTERED)
    file(RELATIVE_PATH _rel "${HIPDNN_IT_CPP_ROOT}" "${_abs}")
    list(APPEND HIPDNN_IT_REGISTERED_REL "${_rel}")
endforeach()

set(HIPDNN_IT_ORPHAN_SOURCES "")
foreach(_rel IN LISTS HIPDNN_IT_ALL_SOURCES)
    if(NOT "${_rel}" IN_LIST HIPDNN_IT_REGISTERED_REL)
        list(APPEND HIPDNN_IT_ORPHAN_SOURCES "${_rel}")
    endif()
endforeach()

if(HIPDNN_IT_ORPHAN_SOURCES)
    string(REPLACE ";" "\n  " HIPDNN_IT_ORPHAN_LIST "${HIPDNN_IT_ORPHAN_SOURCES}")
    message(FATAL_ERROR
        "The following sources under ${HIPDNN_IT_CPP_ROOT} are not registered:\n"
        "  ${HIPDNN_IT_ORPHAN_LIST}\n"
        "Register C++ graph tests with add_cpp_graph_test_sources() so they stay gated "
        "behind -DBUILD_CPP_GRAPH_TESTS=ON. Bundles under integration-test-bundles/ are "
        "the CI driver — prefer adding a bundle over a new C++ graph test.")
endif()
