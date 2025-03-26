################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

include(CMakeParseArguments)

if(NOT DEFINED Tensile_ROOT)
    # Compute the installation prefix relative to this file.
    get_filename_component(Tensile_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
    get_filename_component(Tensile_PREFIX "${Tensile_PREFIX}" PATH)

    if (WIN32)
        execute_process(COMMAND "${Tensile_PREFIX}/bin/TensileGetPath.exe" OUTPUT_VARIABLE Tensile_ROOT)
    else()
        execute_process(COMMAND "${Tensile_PREFIX}/bin/TensileGetPath" OUTPUT_VARIABLE Tensile_ROOT)
    endif()
endif()
list(APPEND CMAKE_MODULE_PATH "${Tensile_ROOT}/Source/cmake/")
list(APPEND CMAKE_MODULE_PATH "${Tensile_ROOT}/Source/")

if("HIP" IN_LIST Tensile_FIND_COMPONENTS)
    set(TENSILE_USE_HIP ON CACHE BOOL "Use HIP")
else()
    set(TENSILE_USE_HIP OFF CACHE BOOL "Use HIP")
endif()

if("LLVM" IN_LIST Tensile_FIND_COMPONENTS)
    set(TENSILE_USE_LLVM ON CACHE BOOL "Use LLVM")
else()
    set(TENSILE_USE_LLVM OFF CACHE BOOL "Use LLVM")
endif()

if("Client" IN_LIST Tensile_FIND_COMPONENTS)
    if(TENSILE_USE_HIP AND TENSILE_USE_LLVM)
        set(TENSILE_BUILD_CLIENT ON CACHE BOOL "Build Client")
    elseif(Tensile_FIND_REQUIRED_Client)
        message("Tensile client requires both Hip and LLVM.")
        set(Tensile_FOUND false)
    else()
        set(TENSILE_BUILD_CLIENT OFF CACHE BOOL "Build Client")
    endif()
else()
    set(TENSILE_BUILD_CLIENT OFF CACHE BOOL "Build Client")
endif()

if("STATIC_ONLY" IN_LIST Tensile_FIND_COMPONENTS)
    set(TENSILE_STATIC_ONLY ON CACHE BOOL "Disable exporting symbols from shared library.")
else()
    set(TENSILE_STATIC_ONLY OFF CACHE BOOL "Disable exporting symbols from shared library.")
endif()

add_subdirectory("${Tensile_ROOT}/Source" "Tensile")
include("${Tensile_ROOT}/Source/TensileCreateLibrary.cmake")

# Output target: ${Tensile_VAR_PREFIX}_LIBRARY_TARGET. Ensures that the libs get built in Tensile_OUTPUT_PATH/library.
function(TensileCreateLibraryFiles
         Tensile_LOGIC_PATH
         Tensile_OUTPUT_PATH
         )

  # Boolean options
  set(options
       SHORT_FILE_NAMES
       PRINT_DEBUG
       GENERATE_PACKAGE
       SEPARATE_ARCHITECTURES
       NO_LAZY_LIBRARY_LOADING
       ASAN_BUILD
       KEEP_BUILD_TMP
       NO_COMPRESS
       EXPERIMENTAL
       ENABLE_MAKRER
       )

  # Single value settings
  set(oneValueArgs
       CODE_OBJECT_VERSION
       COMPILER
       COMPILER_PATH
       EMBED_KEY
       EMBED_LIBRARY
       LIBRARY_FORMAT
       TENSILE_ROOT
       VAR_PREFIX
       CPU_THREADS
       )

  # Multi value settings
  set(multiValueArgs
       ARCHITECTURE
       )

  cmake_parse_arguments(Tensile "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(Tensile_UNPARSED_ARGUMENTS)
    message(WARNING "Unrecognized arguments: ${Tensile_UNPARSED_ARGUMENTS}")
  endif()
  if(Tensile_KEYWORDS_MISSING_VALUES)
    message(WARNING "Malformed arguments: ${Tensile_KEYWORDS_MISSING_VALUES}")
  endif()

  # Parse incoming options
  if(Tensile_TENSILE_ROOT)
    set(Script "${Tensile_TENSILE_ROOT}/bin/TensileCreateLibrary")
  else()
    set(Script "${Tensile_ROOT}/bin/TensileCreateLibrary")
  endif()

  message(STATUS "Tensile script: ${Script}")

  if(Tensile_NO_LAZY_LIBRARY_LOADING)
    set(Options ${Options} "--no-lazy-library-loading")
  endif()

  if(Tensile_ENABLE_MARKER)
    set(Options ${Options} "--enable-marker")
  endif()

  if(Tensile_KEEP_BUILD_TMP)
    set(Options ${Options} "--keep-build-tmp")
  endif()

  if(Tensile_NO_COMPRESS)
    set(Options ${Options} "--no-compress")
  endif()

  if(Tensile_EXPERIMENTAL)
    set(Options ${Options} "--experimental")
  endif()

  if(Tensile_ASAN_BUILD)
    set(Options ${Options} "--address-sanitizer")
  endif()

  if(Tensile_GENERATE_PACKAGE)
    set(Options ${Options} "--package-library")
  endif()

  if(Tensile_SHORT_FILE_NAMES)
    set(Options ${Options} "--short-file-names")
  endif()

  if(Tensile_EMBED_LIBRARY)
    set(Options ${Options} "--embed-library=${Tensile_EMBED_LIBRARY}")
  endif()

  if(Tensile_EMBED_KEY)
    set(Options ${Options} "--embed-library-key=${Tensile_EMBED_KEY}")
  endif()

  if(Tensile_CODE_OBJECT_VERSION)
    set(Options ${Options} "--code-object-version=${Tensile_CODE_OBJECT_VERSION}")
  endif()

  if(Tensile_COMPILER)
    set(Options ${Options} "--cxx-compiler=${Tensile_COMPILER}")
  endif()

  if(Tensile_COMPILER_PATH)
    set(Options ${Options} "--cmake-cxx-compiler=${Tensile_COMPILER_PATH}")
  endif()

  if(Tensile_CPU_THREADS)
    set(Options ${Options} "--jobs=${Tensile_CPU_THREADS}")
  endif()

  if(Tensile_ASM_DEBUG)
    set(Options ${Options} "--asm-debug")
  endif()

  if(Tensile_LOGIC_FILTER)
    set(Options ${Options} "--logic-filter=${Tensile_LOGIC_FILTER}")
  endif()

  if(Tensile_LIBRARY_FORMAT)
    set(Options ${Options} "--library-format=${Tensile_LIBRARY_FORMAT}")
    if(Tensile_LIBRARY_FORMAT MATCHES "yaml")
        target_compile_definitions( TensileHost PUBLIC -DTENSILE_YAML=1)
    endif()
  endif()

  if(Tensile_ARCHITECTURE)
    string (REPLACE ";" "_" archString "${Tensile_ARCHITECTURE}")
    # uses _ separator to avoid cmake ; list interpretation, either ; or _ decoded in TensileCreateLibrary
    set(Options ${Options} "--architecture=${archString}")
  endif()

  if(Tensile_BUILD_ID)
    set(Options ${Options} "--build-id=${Tensile_BUILD_ID}")
  endif()

  set(CommandLine ${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} ${Script} ${Options} ${Tensile_LOGIC_PATH} ${Tensile_OUTPUT_PATH} HIP)
  message(STATUS "Tensile_CREATE_COMMAND: ${CommandLine}")

  if(Tensile_EMBED_LIBRARY)
      set(Tensile_EMBED_LIBRARY_SOURCE "${Tensile_OUTPUT_PATH}/library/${Tensile_EMBED_LIBRARY}.cpp")
  endif()

  if($ENV{TENSILE_SKIP_LIBRARY})
      message(STATUS "Skipping build of ${Tensile_OUTPUT_PATH}")
  else()

      if(NOT Tensile_VAR_PREFIX)
          set(Tensile_VAR_PREFIX TENSILE)
      endif()

      if($ENV{ENABLE_ADDRESS_SANITIZER})
        # Must populate LD_PRELOAD with ASAN runtime if ASAN is being used.
        # Find the ASAN RT with compiler and update env for Tensile call.
        execute_process(
          COMMAND ${CMAKE_CXX_COMPILER} --print-file-name=libclang_rt.asan-x86_64.so
          OUTPUT_VARIABLE ASAN_LIB_PATH
          COMMAND_ECHO STDOUT)
        string(STRIP ${ASAN_LIB_PATH} ASAN_LIB_PATH)
        set(CommandLine env LD_PRELOAD=${ASAN_LIB_PATH} ${CommandLine})
      endif()

      add_custom_command(
        COMMENT "Generating Tensile Libraries"
        OUTPUT ${Tensile_OUTPUT_PATH}/library
        COMMAND ${CommandLine}
      )

      add_custom_target(
        "${Tensile_VAR_PREFIX}_LIBRARY_TARGET" ALL
        COMMENT "${Tensile_VAR_PREFIX}_LIBRARY_TARGET"
        DEPENDS ${Tensile_OUTPUT_PATH}/library
      )

  endif()

  if(Tensile_EMBED_LIBRARY)

    add_library(${Tensile_EMBED_LIBRARY} ${Tensile_EMBED_LIBRARY_SOURCE})
    target_link_libraries(${Tensile_EMBED_LIBRARY} PUBLIC TensileHost)

  endif()

endfunction()

function(TensileCreateExtOpLibraries OutputFolder ArchStr)
  string(REGEX MATCHALL "gfx[a-z0-9]+" Archs "${ArchStr}")
  list(REMOVE_DUPLICATES Archs)
  set(build_tmp_dir ${OutputFolder}/../build_tmp/ops)
  set(Tensile_PACKAGE_DIR ${Tensile_SOURCE_DIR}/../)
  set(cwd "${Tensile_PACKAGE_DIR}/Ops")
  set(script "${cwd}/gen_assembly.sh")
  set(ext_op_library_path ${build_tmp_dir}/hipblasltExtOpLibrary.dat)
  file(REMOVE ${ext_op_library_path})

  add_custom_command(
    OUTPUT ${OutputFolder}/hipblasltExtOpLibrary.dat
    WORKING_DIRECTORY "${cwd}"
    COMMENT "Creating ExtOp Libraries"
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${build_tmp_dir}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${build_tmp_dir}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OutputFolder}
    COMMAND bash "${script}" "\"${Archs}\"" "${build_tmp_dir}" "${VIRTUALENV_HOME_DIR}" "${Tensile_BUILD_ID}"
    COMMAND ${CMAKE_COMMAND} -E copy ${ext_op_library_path} ${build_tmp_dir}/extop_*.co ${OutputFolder}
  )

  add_custom_target(
    build_ext_op_library ALL
    WORKING_DIRECTORY "${cwd}"
    DEPENDS ${OutputFolder}/hipblasltExtOpLibrary.dat)

endfunction()
