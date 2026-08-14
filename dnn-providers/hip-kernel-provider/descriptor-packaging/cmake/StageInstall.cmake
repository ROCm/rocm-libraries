# Configure, build, and install a throwaway harness project so the shipped
# descriptor tree is produced by the REAL install() rule (not a file-copy).
# The harness enables the production pack+install against a fixture source root,
# so this exercises the exact install(DIRECTORY ...) wiring a release uses while
# staying scoped to the test build. Invoked as:
#   cmake -DHARNESS_DIR=<dir> -DBUILD_DIR=<dir> -DSTAGING=<dir>
#         -DSOURCE_ROOT=<fixture> -DGPU_TARGETS=gfx942;gfx950
#         -DKPACK_PYTHON=<dir> -DHIPCC=<path> -DENGINE_DIR=<rel>
#         -DGENERATOR=<gen> -P StageInstall.cmake

foreach(_req HARNESS_DIR BUILD_DIR STAGING SOURCE_ROOT GPU_TARGETS HIPCC ENGINE_DIR)
    if(NOT DEFINED ${_req})
        message(FATAL_ERROR "${_req} must be defined")
    endif()
endforeach()

file(REMOVE_RECURSE "${BUILD_DIR}" "${STAGING}")

set(_gen_arg "")
if(DEFINED GENERATOR AND GENERATOR)
    set(_gen_arg -G "${GENERATOR}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" ${_gen_arg}
            -S "${HARNESS_DIR}"
            -B "${BUILD_DIR}"
            "-DHIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT=${SOURCE_ROOT}"
            "-DGPU_TARGETS=${GPU_TARGETS}"
            "-DHIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR=${ENGINE_DIR}"
            "-DHIPKERNELPROVIDER_KPACK_PYTHON_DIR=${KPACK_PYTHON}"
            "-DHKP_HIPCC=${HIPCC}"
            "-DHIPKERNELPROVIDER_ENABLE_TESTS=OFF"
            "-DCMAKE_INSTALL_PREFIX=${STAGING}"
    RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "hkp_stage: harness configure failed (${_rc})")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${BUILD_DIR}"
    RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "hkp_stage: harness build failed (${_rc})")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${STAGING}"
    RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "hkp_stage: harness install failed (${_rc})")
endif()

message(STATUS "hkp_stage: staged real install() into ${STAGING}")
