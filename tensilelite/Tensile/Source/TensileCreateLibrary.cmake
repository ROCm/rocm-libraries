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

################################################################################
# Create A Tensile Library from LibraryLogic.yaml files
################################################################################
function(TensileCreateLibraryCmake
    Tensile_LOGIC_PATH
    Tensile_RUNTIME_LANGUAGE
    Tensile_COMPILER
    Tensile_CODE_OBJECT_VERSION
    Tensile_ARCHITECTURE
    Tensile_LIBRARY_FORMAT
    Tensile_SHORT_FILE_NAMES
    Tensile_CPU_THREADS
    Tensile_SEPARATE_ARCHITECTURES
    Tensile_LAZY_LIBRARY_LOADING,
    Tensile_ENABLE_MARKER,
    Tensile_BUILD_ID)

# make Tensile_PACKAGE_LIBRARY and optional parameter
# to avoid breaking applications which us this
  if (ARGN)
    list (GET ARGN 0 Tensile_PACKAGE_LIBRARY)
    # list (GET ARGN 1 Tensile_INCLUDE_LEGACY_CODE)
  else()
    set(Tensile_PACKAGE_LIBRARY OFF)
    # set(Tensile_INCLUDE_LEGACY_CODE ON)
  endif()

  # set(options PACKAGE_LIBRARY Tensile_INCLUDE_LEGACY_CODE)
  message(STATUS "Tensile_RUNTIME_LANGUAGE    from TensileCreateLibraryCmake : ${Tensile_RUNTIME_LANGUAGE}")
  message(STATUS "Tensile_CODE_OBJECT_VERSION from TensileCreateLibraryCmake : ${Tensile_CODE_OBJECT_VERSION}")
  message(STATUS "Tensile_COMPILER            from TensileCreateLibraryCmake : ${Tensile_COMPILER}")
  message(STATUS "Tensile_ARCHITECTURE        from TensileCreateLibraryCmake : ${Tensile_ARCHITECTURE}")
  message(STATUS "Tensile_LIBRARY_FORMAT      from TensileCreateLibraryCmake : ${Tensile_LIBRARY_FORMAT}")
  message(STATUS "Tensile_CPU_THREADS         from TensileCreateLibraryCmake : ${Tensile_CPU_THREADS}")

  #execute_process(COMMAND chmod 755 ${Tensile_ROOT}/bin/TensileCreateLibrary)
  #execute_process(COMMAND chmod 755 ${Tensile_ROOT}/bin/Tensile)
  file(COPY ${Tensile_ROOT}/bin/TensileCreateLibrary DESTINATION ${Tensile_ROOT}/bin FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  file(COPY ${Tensile_ROOT}/bin/Tensile DESTINATION ${Tensile_ROOT}/bin FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

  set(Tensile_CREATE_COMMAND "${Tensile_ROOT}/bin/TensileCreateLibrary")

  set(Tensile_SOURCE_PATH "${PROJECT_BINARY_DIR}/Tensile")
  message(STATUS "Tensile_SOURCE_PATH=${Tensile_SOURCE_PATH}")

  # TensileLibraryWriter optional arguments
  if(${Tensile_PACKAGE_LIBRARY})
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--package-library")
  endif()

  if(${Tensile_SEPARATE_ARCHITECTURES})
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--separate-architectures")
  endif()

  if(${Tensile_LAZY_LIBRARY_LOADING})
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--lazy-library-loading")
  endif()

  if(Tensile_ENABLE_MARKER)
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--enable-marker")
  endif()

  if(${Tensile_SHORT_FILE_NAMES})
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--short-file-names")
  else()
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--no-short-file-names")
  endif()

  set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--architecture=${Tensile_ARCHITECTURE}")
  set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--code-object-version=${Tensile_CODE_OBJECT_VERSION}")
  set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--cxx-compiler=${Tensile_COMPILER}")
  set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--library-format=${Tensile_LIBRARY_FORMAT}")
  set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--jobs=${Tensile_CPU_THREADS}")

  if(Tensile_BUILD_ID)
    set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND} "--build-id=${Tensile_BUILD_ID}")
  endif()

  # TensileLibraryWriter positional arguments
  set(Tensile_CREATE_COMMAND ${Tensile_CREATE_COMMAND}
    ${Tensile_LOGIC_PATH}
    ${Tensile_SOURCE_PATH}
    ${Tensile_RUNTIME_LANGUAGE}
    )

  #string( REPLACE ";" " " Tensile_CREATE_COMMAND "${Tensile_CREATE_COMMAND}")
  message(STATUS "Tensile_CREATE_COMMAND: ${Tensile_CREATE_COMMAND}")

  # execute python command
  if($ENV{TENSILE_SKIP_LIBRARY})
    message(STATUS "Skipping build of ${Tensile_OUTPUT_PATH}")
  else()
    if (WIN32)
      set(CommandLine ${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} ${Tensile_CREATE_COMMAND})
    else()
      set(CommandLine ${Tensile_CREATE_COMMAND})
    endif()
    execute_process(
      COMMAND ${CommandLine}
      RESULT_VARIABLE Tensile_CREATE_RESULT
    )
    if(Tensile_CREATE_RESULT)
      message(FATAL_ERROR "Error generating kernels")
    endif()
  endif()

endfunction()
