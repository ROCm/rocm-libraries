
cmake_minimum_required(VERSION 3.16...3.19)

project(rocRoller VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)

option(ROCROLLER_ENABLE_TIMERS         "Enable rocRoller timer code." OFF)
option(ROCROLLER_USE_HIP               "Use HIP within the rocRoller." ON)
option(SKIP_CPPCHECK                   "Skip CPPCHECK." ON)

option(CODE_ANALYSIS                    "When building, don't query for tests etc." OFF)
option(CODE_COVERAGE                    "Build with code coverage flags (clang only)" OFF)

option(BUILD_SHARED_LIBS                "Build using shared libraries" ON)
option(BUILD_CLIENTS                    "Build performance testing clients" ON)
option(BUILD_TESTING                    "Build rocRoller test clients" ON)
option(BUILD_DOCS                       "Build rocRoller documentation" ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(YAML_BACKEND "YAML_CPP" CACHE STRING "YAML backend used for serialization")
set_property(CACHE YAML_BACKEND PROPERTY STRINGS YAML_CPP LLVM)

# For paths given when reporting errors
add_compile_options(-fmacro-prefix-map=${CMAKE_SOURCE_DIR}=.)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(COROUTINES_COMPILE_OPTION )
  set(EXTRA_COMPILE_OPTIONS )
  set(EXTRA_LINK_OPTIONS -fuse-ld=lld)
else()
  set(COROUTINES_COMPILE_OPTION -fcoroutines)
  set(EXTRA_COMPILE_OPTIONS )
  set(EXTRA_LINK_OPTIONS )
endif()

if(ROCROLLER_ENABLE_TIMERS)
  add_definitions(-DROCROLLER_ENABLE_TIMERS)
  add_compile_options(-pg -fno-omit-frame-pointer)
endif()

if(CODE_COVERAGE)
  set(EXTRA_COMPILE_OPTIONS -fprofile-instr-generate -fcoverage-mapping ${EXTRA_COMPILE_OPTIONS})
  set(EXTRA_LINK_OPTIONS -fprofile-instr-generate -fcoverage-mapping ${EXTRA_LINK_OPTIONS})
endif()

if(ROCROLLER_USE_HIP)
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip)
  find_package(hip)

  find_package(LLVM REQUIRED HINTS /opt/rocm/llvm/)

  if(YAML_BACKEND STREQUAL "LLVM")
    set(LLVM_YAML_COMPONENT "objectyaml")
  endif()

  llvm_map_components_to_libnames(llvm_libs mc support ${LLVM_YAML_COMPONENT})
  find_package(LLD REQUIRED CONFIG HINTS ${LLVM_DIR}/../lld/)
endif()

include(FetchContent)

FetchContent_Declare(
  libdivide
  GIT_REPOSITORY https://github.com/ridiculousfish/libdivide.git
  GIT_TAG v5.1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)

if(NOT libdivide_POPULATED)
  FetchContent_Populate(libdivide)
endif()

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG 27cb4c76708608465c413f6d0e6b8d99a4d84302 # release 1.14.1
)
if(NOT spdlog_POPULATED)
  FetchContent_Populate(spdlog)
  add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR})
  set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(spdlog PRIVATE SPDLOG_USE_STD_FORMAT)
endif()
FetchContent_MakeAvailable(spdlog)
set(spdlog_INCLUDES ${spdlog_SOURCE_DIR}/include)
add_compile_definitions(SPDLOG_USE_STD_FORMAT)

add_definitions(-DMSGPACK_NO_BOOST)
set(MSGPACK_USE_BOOST OFF CACHE INTERNAL "")
FetchContent_Declare(
  msgpack
  GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
  GIT_TAG cpp-4.1.1
)
FetchContent_MakeAvailable(msgpack)

if(YAML_BACKEND STREQUAL "YAML_CPP")
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set(YAML_CPP_BUILD_TOOLS OFF)
  FetchContent_Declare(
    yaml-cpp
    URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.7.0.zip
  )
  FetchContent_MakeAvailable(yaml-cpp)

  target_compile_options(yaml-cpp PUBLIC ${EXTRA_COMPILE_OPTIONS})
  target_link_options(yaml-cpp PUBLIC ${EXTRA_LINK_OPTIONS})
  set_property(TARGET yaml-cpp PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET yaml-cpp APPEND PROPERTY COMPILE_OPTIONS "-Wno-shadow")

  add_definitions(-DROCROLLER_USE_YAML_CPP)
elseif(YAML_BACKEND STREQUAL "LLVM")
  add_definitions(-DROCROLLER_USE_LLVM)
endif()

if(SKIP_CPPCHECK)
  unset(CMAKE_CXX_CPPCHECK CACHE)
else()
  find_program(CMAKE_CXX_CPPCHECK NAMES cppcheck)
  if (CMAKE_CXX_CPPCHECK)
      list(
          APPEND CMAKE_CXX_CPPCHECK
              "--enable=warning"
              "--enable=performance"
              "--enable=portability"
              "--enable=style"
              "--error-exitcode=10"
              "--force"
              "--inline-suppr"
              "-q"
              "--library=googletest"
              "--suppressions-list=${CMAKE_SOURCE_DIR}/CppCheckSuppressions.txt"
      )
  endif()
endif()


# Default ARCHITECTURE_FILE_PATH
set(ROCROLLER_DEFAULT_ARCHITECTURE_FILE_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/source/rocRoller/GPUArchitecture_def.msgpack")
set(ROCROLLER_DEFAULT_ARCHITECTURE_CPP_FILE ${CMAKE_BINARY_DIR}/source/rocRoller/DefaultArchitectureFilePath.cpp)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lib/source/DefaultArchitectureFilePath.cpp.in ${ROCROLLER_DEFAULT_ARCHITECTURE_CPP_FILE})

# Git hash
execute_process(
    COMMAND git describe --always --dirty
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    OUTPUT_VARIABLE ROCROLLER_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

set(ROCROLLER_GIT_VERSION_CPP_FILE ${CMAKE_BINARY_DIR}/source/rocRoller/Utilities/Version.cpp)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lib/source/Utilities/Version.cpp.in ${ROCROLLER_GIT_VERSION_CPP_FILE})

add_subdirectory(GPUArchitectureGenerator)
add_subdirectory(lib)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

if(BUILD_DOCS)
  # Add docs directory
  add_subdirectory(docs)
endif()

if(BUILD_CLIENTS)
  # Add client directory
  add_subdirectory(client)
endif()
