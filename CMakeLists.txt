
cmake_minimum_required(VERSION 3.16...3.22)

set(ROCROLLER_VERSION "1.0.0")
set(ROCROLLER_SOVERSION 1)

project(rocRoller VERSION ${ROCROLLER_VERSION} LANGUAGES CXX)
if(CMAKE_C_COMPILER)
  # If the compiler is specified but the C language isn't enabled,
  # LLVM pitches a fit, so for easy compatibility just enable the C language
  enable_language(C)
endif()

set(CMAKE_CXX_STANDARD 20)

option(ROCROLLER_ENABLE_TIMERS         "Enable rocRoller timer code." OFF)
option(ROCROLLER_USE_HIP               "Use HIP within the rocRoller." ON)
option(ROCROLLER_EMBED_ARCH_DEF        "Embed the generated GPU architecture msgpack in the library" ON)
option(SKIP_CPPCHECK                   "Skip CPPCHECK." ON)

option(CODE_ANALYSIS                    "When building, don't query for tests etc." OFF)
option(CODE_COVERAGE                    "Build with code coverage flags (clang only)" OFF)

option(BUILD_SHARED_LIBS                "Build using shared libraries" ON)
option(BUILD_CLIENTS                    "Build performance testing clients" ON)
option(BUILD_TESTING                    "Build rocRoller test clients" ON)
option(BUILD_DOCS                       "Build rocRoller documentation" ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

set(YAML_BACKEND "YAML_CPP" CACHE STRING "YAML backend used for serialization")
set_property(CACHE YAML_BACKEND PROPERTY STRINGS YAML_CPP LLVM)

# For paths given when reporting errors
add_compile_options(-fmacro-prefix-map=${PROJECT_SOURCE_DIR}=.)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(COROUTINES_COMPILE_OPTION )
  set(EXTRA_COMPILE_OPTIONS )
  set(EXTRA_LINK_OPTIONS -fuse-ld=lld -frtti)
else()
  set(COROUTINES_COMPILE_OPTION -fcoroutines)
  set(EXTRA_COMPILE_OPTIONS )
  set(EXTRA_LINK_OPTIONS -frtti)
endif()

if(ROCROLLER_ENABLE_TIMERS)
  add_definitions(-DROCROLLER_ENABLE_TIMERS)
  add_compile_options(-pg -fno-omit-frame-pointer)
endif()

if(CODE_COVERAGE)
  list(PREPEND EXTRA_COMPILE_OPTIONS -fprofile-instr-generate -fcoverage-mapping)
  list(PREPEND EXTRA_LINK_OPTIONS -fprofile-instr-generate -fcoverage-mapping)
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

include(cmake/Dependencies.cmake)
# both show up in headers, system package required
rocroller_add_dependency(spdlog VERSION 1.9.2)
rocroller_add_dependency(msgpack VERSION 3.3.0)

if(YAML_BACKEND STREQUAL "YAML_CPP")
  # shows up in headers, system package required
  rocroller_add_dependency(yaml-cpp VERSION 0.7.0)
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
              "--suppressions-list=${PROJECT_SOURCE_DIR}/CppCheckSuppressions.txt"
      )
  endif()
endif()

# Git hash
execute_process(
    COMMAND git describe --always --dirty
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    OUTPUT_VARIABLE ROCROLLER_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

set(ROCROLLER_GIT_VERSION_CPP_FILE ${PROJECT_BINARY_DIR}/source/rocRoller/Utilities/Version.cpp)
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
