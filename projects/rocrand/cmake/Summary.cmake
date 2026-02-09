function(print_configuration_summary)
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
    endif()

    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} --version
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        OUTPUT_VARIABLE CMAKE_CXX_COMPILER_VERBOSE_DETAILS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    find_program(UNAME_EXECUTABLE uname)
    if(UNAME_EXECUTABLE)
        execute_process(
            COMMAND ${UNAME_EXECUTABLE} -a
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            OUTPUT_VARIABLE LINUX_KERNEL_DETAILS
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()  

    string(REPLACE "\n" ";" CMAKE_CXX_COMPILER_VERBOSE_DETAILS "${CMAKE_CXX_COMPILER_VERBOSE_DETAILS}")
    list(TRANSFORM CMAKE_CXX_COMPILER_VERBOSE_DETAILS PREPEND "--     ")
    string(REPLACE ";" "\n" CMAKE_CXX_COMPILER_VERBOSE_DETAILS "${CMAKE_CXX_COMPILER_VERBOSE_DETAILS}")

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
        message(STATUS "  BUILD_LEGACY_BENCHMARK     : ${BUILD_LEGACY_BENCHMARK}")
        message(STATUS "  BUILD_BENCHMARK_TUNING     : ${BUILD_BENCHMARK_TUNING}")
    endif()
    message(STATUS "  BUILD_ADDRESS_SANITIZER    : ${BUILD_ADDRESS_SANITIZER}")
    message(STATUS "  DEPENDENCIES_FORCE_DOWNLOAD: ${DEPENDENCIES_FORCE_DOWNLOAD}")
    message(STATUS "  USE_SYSTEM_LIB             : ${USE_SYSTEM_LIB}")
    message(STATUS "")
    message(STATUS "Detailed:")
    message(STATUS "  C++ compiler details       : \n${CMAKE_CXX_COMPILER_VERBOSE_DETAILS}")
if(GIT_FOUND)
    message(STATUS "  Commit                     : ${COMMIT_HASH}")
    message(STATUS "                               ${COMMIT_SUBJECT}")
endif()
if(UNAME_EXECUTABLE)
    message(STATUS "  Unix name                  : ${LINUX_KERNEL_DETAILS}")
endif()
  
endfunction()
