cmake_minimum_required(VERSION 3.21)

set(VERSION_STRING "1.0.0")

project(hipthreads VERSION ${VERSION_STRING} LANGUAGES CXX HIP)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT WIN32)
  add_compile_options("-Wno-error=#warnings")
  set(CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=lld")
  set(CMAKE_MODULE_LINKER_FLAGS_INIT "-fuse-ld=lld")
  set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=lld")
else()
  set(CMAKE_CXX_USING_LINKER_DEFAULT "-fuse-ld=lld")
  set(CMAKE_CXX_USING_LINKER_LLD "-fuse-ld=lld")
  set(CMAKE_HIP_USING_LINKER_DEFAULT "-fuse-ld=lld")
  set(CMAKE_HIP_USING_LINKER_LLD "-fuse-ld=lld")
endif()

list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} /opt/rocm)

find_package(hip REQUIRED)
find_package(libhipcxx REQUIRED)

# # Used for automatic generation of LLVM IR (by dissasembling bitcode generated during compilation from the "-save-temps" option)
# find_program(LLVM_DIS llvm-dis REQUIRED HINTS /opt/rocm/llvm/bin)

include(GNUInstallDirs)

# ON: copy the source tree into the build folder. The TheRock tester needs this
# so it can run the lit suite against the packaged build artifact (lit compiles
# the test sources on the tester, so they must be present in the artifact). Set
# by TheRock's build recipe; OFF for normal local/CI builds.
option(HIPTHREADS_COPY_TO_BUILD
  "Copy source tree into build folder (set by TheRock; required for the tester pipeline)." OFF)

if(HIPTHREADS_COPY_TO_BUILD)
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}"
    REGEX ".*(matrices\\.tgz|shallow_water1\\.mtx)$" EXCLUDE
  )
endif()

add_subdirectory(src)
if (CMAKE_BUILD_TYPE MATCHES Debug)
  add_subdirectory(test)
endif()

#-----------------------------------------------------------------------------
# Installation and Packaging Rules
#-----------------------------------------------------------------------------
include(CMakePackageConfigHelpers)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  if(DEFINED ENV{ROCM_PATH})
    set(CMAKE_INSTALL_PREFIX "$ENV{ROCM_PATH}" CACHE PATH "Install path prefix" FORCE)
  else()
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix" FORCE)
  endif()
endif()

install(DIRECTORY ${CMAKE_SOURCE_DIR}/inc/hip/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hipthreads/hip
)

install(TARGETS hipthreads
    EXPORT hipthreadsTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/hipthreads
)

set(CMAKECONFIG_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/hipthreads)

install(EXPORT hipthreadsTargets
    FILE hipthreadsTargets.cmake
    NAMESPACE hipthreads::
    DESTINATION ${CMAKECONFIG_INSTALL_DIR}
)

configure_package_config_file(
    "cmake/hipthreadsConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/hipthreadsConfig.cmake"
    INSTALL_DESTINATION ${CMAKECONFIG_INSTALL_DIR}
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/hipthreadsConfig.cmake"
    DESTINATION ${CMAKECONFIG_INSTALL_DIR}
)

# TEMP: until libhipcxx resolves this
target_compile_definitions(hipthreads PUBLIC _LIBCUDACXX_ALLOW_UNSUPPORTED_ARCHITECTURE)
