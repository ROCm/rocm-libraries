# Prints a project configuration summary (system, compiler, build, and Git info).
function(print_configuration_summary)
    _print_git_info()
    _get_compiler_details(COMPILER_DETAILS)
    _get_system_info(SYSTEM_DETAILS)

    _print_general_info()

    message(STATUS "Detailed:")
    message(STATUS "  C++ compiler details       : \n${COMPILER_DETAILS}")

if(GIT_FOUND)
    message(STATUS "  Commit                     : ${COMMIT_HASH}")
    message(STATUS "                               ${COMMIT_SUBJECT}")
endif()

if(UNAME_EXECUTABLE)
    message(STATUS "  Unix name                  : ${SYSTEM_DETAILS}")
endif()
endfunction()

# Retrieves Git commit information if available.
function(_print_git_info)
    find_package(Git)
    if(GIT_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} show --format=%H --no-patch
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            OUTPUT_VARIABLE COMMIT_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        execute_process(
            COMMAND ${GIT_EXECUTABLE} show --format=%s --no-patch
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            OUTPUT_VARIABLE COMMIT_SUBJECT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(COMMIT_HASH "${COMMIT_HASH}" PARENT_SCOPE)
        set(COMMIT_SUBJECT "${COMMIT_SUBJECT}" PARENT_SCOPE)
    endif()
    set(GIT_FOUND "${GIT_FOUND}" PARENT_SCOPE)
endfunction()

# Fetches and formats the C++ compiler's verbose version information.
function(_get_compiler_details OUTPUT_VAR)
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} --version
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE DETAILS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REPLACE "\n" ";" DETAILS "${DETAILS}")
    list(TRANSFORM DETAILS PREPEND "--     ")
    string(REPLACE ";" "\n" DETAILS "${DETAILS}")
    set(${OUTPUT_VAR} "${DETAILS}" PARENT_SCOPE)
endfunction()

# Retrieves system information using the uname command.
function(_get_system_info OUTPUT_VAR)
    find_program(UNAME_EXECUTABLE uname)
    if(UNAME_EXECUTABLE)
        execute_process(
            COMMAND ${UNAME_EXECUTABLE} -a
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            OUTPUT_VARIABLE DETAILS
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(${OUTPUT_VAR} "${DETAILS}" PARENT_SCOPE)
    endif()
    set(UNAME_EXECUTABLE "${UNAME_EXECUTABLE}" PARENT_SCOPE)
endfunction()

# Prints the 'General' section of the configuration summary.
function(_print_general_info)
    message(STATUS "")
    message(STATUS "******** Summary ********")
    message(STATUS "General:")
    message(STATUS "  System                     : ${CMAKE_SYSTEM_NAME}")
    message(STATUS "  HIP ROOT                   : ${HIP_ROOT_DIR}")
    message(STATUS "  C++ compiler               : ${CMAKE_CXX_COMPILER}")
    message(STATUS "  C++ compiler version       : ${CMAKE_CXX_COMPILER_VERSION}")
    string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS_STRIP)
    message(STATUS "  CXX flags                  : ${CMAKE_CXX_FLAGS_STRIP}")
    if(HIP_COMPILER STREQUAL "nvcc")
        string(REPLACE ";" " " HIP_NVCC_FLAGS_STRIP "${HIP_NVCC_FLAGS}")
        string(STRIP "${HIP_NVCC_FLAGS_STRIP}" HIP_NVCC_FLAGS_STRIP)
        string(REPLACE ";" " " HIP_CPP_CONFIG_FLAGS_STRIP "${HIP_CPP_CONFIG_FLAGS}")
        string(STRIP "${HIP_CPP_CONFIG_FLAGS_STRIP}" HIP_CPP_CONFIG_FLAGS_STRIP)
        message(STATUS "  HIP flags                  : ${HIP_CPP_CONFIG_FLAGS_STRIP}")
        message(STATUS "  NVCC flags                 : ${HIP_NVCC_FLAGS_STRIP}")
    endif()
    message(STATUS "  ROCRAND_HAVE_ASM_INCBIN    : ${ROCRAND_HAVE_ASM_INCBIN}")
    message(STATUS "  Build type                 : ${CMAKE_BUILD_TYPE}")
    message(STATUS "  Install prefix             : ${CMAKE_INSTALL_PREFIX}")
    if(HIP_COMPILER STREQUAL "clang")
        message(STATUS "  Device targets             : ${GPU_TARGETS}")
    else()
        message(STATUS "  Device targets             : ${NVGPU_TARGETS}")
    endif()
    message(STATUS "")
    message(STATUS "  BUILD_SHARED_LIBS          : ${BUILD_SHARED_LIBS}")
    message(STATUS "  BUILD_FORTRAN_WRAPPER      : ${BUILD_FORTRAN_WRAPPER}")
    message(STATUS "  BUILD_TEST                 : ${BUILD_TEST}")
    message(STATUS "  BUILD_BENCHMARK            : ${BUILD_BENCHMARK}")
    if(BUILD_BENCHMARK)
        message(STATUS "  BUILD_BENCHMARK_TUNING     : ${BUILD_BENCHMARK_TUNING}")
    endif()
    message(STATUS "  BUILD_ADDRESS_SANITIZER    : ${BUILD_ADDRESS_SANITIZER}")
    message(STATUS "  DEPENDENCIES_FORCE_DOWNLOAD: ${DEPENDENCIES_FORCE_DOWNLOAD}")
    message(STATUS "  USE_SYSTEM_LIB             : ${USE_SYSTEM_LIB}")
    message(STATUS "")
endfunction()
