# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# library/cpp -- the C++ half of the dual-engine contract for library-owned
# kernel families. Every family's Python builder lives in library/kernels; its
# C++ mirror lives here, and tools/check_byte_identity.py proves the two emit
# the same LLVM IR, exactly as it does for the engine's own families.
#
#   include/rocke/   family headers, under the engine's rocke/ prefix
#   instances/       family sources (common/, gfx*/), picked up by the glob below
#   bindings/        pybind glue for rocke_library_engine
#   tests/           host-only C++ tests
#
# Not a standalone project: platform/CMakeLists.txt adds this directory after
# rocke_core (and, with ROCKE_BUILD_PYBIND, rocke_engine) exist, so it inherits
# the engine's -Werror set, sanitizers and clang-tidy opt-out. The dependency
# is one way: these targets link rocke_core; the engine never links them.
# The archive is never installed (library/ ships no product artifact); the
# pybind module installs beside rocke_engine, as that module does.

file(GLOB_RECURSE ROCKE_LIBRARY_CORE_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/instances/*.cpp")
if(NOT ROCKE_LIBRARY_CORE_SOURCES)
  return()
endif()

add_library(rocke_library_core STATIC ${ROCKE_LIBRARY_CORE_SOURCES})
target_include_directories(rocke_library_core PUBLIC
                           "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(rocke_library_core PUBLIC rocke_core)
# Beside librocke_core.a, where run_diff.py looks for it by default. PIC because
# the pybind module below links it into a shared object.
set_target_properties(rocke_library_core PROPERTIES
  OUTPUT_NAME rocke_library
  ARCHIVE_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:rocke_core>"
  POSITION_INDEPENDENT_CODE ON)

# A family header must not shadow an engine header of the same name: both
# trees resolve through the rocke/ prefix.
file(GLOB _rocke_library_headers RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/include/rocke"
     "${CMAKE_CURRENT_SOURCE_DIR}/include/rocke/*")
foreach(_h ${_rocke_library_headers})
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../platform/cpp/include/rocke/${_h}")
    message(FATAL_ERROR
            "library/cpp/include/rocke/${_h} shadows the engine header of the same "
            "name in platform/cpp/include/rocke; rename one of them")
  endif()
endforeach()

# --- rocke_library_engine: pybind module for the library families ------------
# Built beside rocke_engine, so whatever puts that module on sys.path (a .pth,
# PYTHONPATH, the install prefix) finds this one too.
file(GLOB ROCKE_LIBRARY_BINDING_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/bindings/*.cpp")
if(TARGET rocke_engine AND ROCKE_LIBRARY_BINDING_SOURCES)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(rocke_library_engine ${ROCKE_LIBRARY_BINDING_SOURCES})
  # The engine's binding glue (family_glue.hpp, builder_lifetime.hpp,
  # llvm_flavor.hpp) is shared, not copied, so both modules marshal spec dicts
  # and LLVM flavors the same way.
  target_include_directories(rocke_library_engine PRIVATE
                             "${CMAKE_CURRENT_SOURCE_DIR}/../../platform/cpp/bindings")
  find_package(Threads REQUIRED)
  target_link_libraries(rocke_library_engine PRIVATE rocke_library_core rocke_core
                        ${CMAKE_DL_LIBS} Threads::Threads)
  if(UNIX)
    target_link_libraries(rocke_library_engine PRIVATE m)
  endif()
  set_target_properties(rocke_library_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:rocke_engine>"
    RUNTIME_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:rocke_engine>")
  install(TARGETS rocke_library_engine
          LIBRARY DESTINATION "${ROCKE_PYTHON_INSTALL_DIR}"
          RUNTIME DESTINATION "${ROCKE_PYTHON_INSTALL_DIR}")
endif()

if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()
