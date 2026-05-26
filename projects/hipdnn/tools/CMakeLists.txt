# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(hipdnn_list_engines ListEngines.cpp)
target_link_libraries(hipdnn_list_engines PRIVATE hipdnn_backend)
target_compile_definitions(hipdnn_list_engines PRIVATE HIPDNN_BACKEND_COMPILATION)
install(TARGETS hipdnn_list_engines RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Python tool bundles (shiv .pyz zipapps)
#
# Shared shiv venv across all bundle targets. Created once via a stamp file
# so parallel ninja jobs don't race to bootstrap it.
set(SHIV_VERSION "1.0.8")
set(shiv_venv_dir "${CMAKE_CURRENT_BINARY_DIR}/shiv-venv")
set(shiv_stamp "${shiv_venv_dir}/.stamp")
if(WIN32)
  set(shiv_exe "${shiv_venv_dir}/Scripts/shiv.exe")
  set(shiv_pip "${shiv_venv_dir}/Scripts/pip.exe")
else()
  set(shiv_exe "${shiv_venv_dir}/bin/shiv")
  set(shiv_pip "${shiv_venv_dir}/bin/pip")
endif()

find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_custom_command(
    OUTPUT ${shiv_stamp}
    COMMAND ${Python3_EXECUTABLE} -m venv ${shiv_venv_dir}
    COMMAND ${shiv_pip} install --quiet --disable-pip-version-check shiv==${SHIV_VERSION}
    COMMAND ${CMAKE_COMMAND} -E touch ${shiv_stamp}
    COMMENT "Bootstrapping shared shiv venv (${SHIV_VERSION})"
    VERBATIM
  )
  add_custom_target(shiv-venv DEPENDS ${shiv_stamp})
endif()

function(add_pyz_bundle target_name pkg_dir pyz_name)
  if(NOT Python3_Interpreter_FOUND)
    return()
  endif()

  set(out_dir "${CMAKE_CURRENT_BINARY_DIR}/${pkg_dir}")
  set(pyz_path "${out_dir}/${pyz_name}.pyz")

  file(GLOB_RECURSE pkg_sources CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/${pkg_dir}/src/*.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/${pkg_dir}/pyproject.toml"
  )

  if(WIN32)
    find_program(POWERSHELL_EXECUTABLE NAMES pwsh powershell)
    if(NOT POWERSHELL_EXECUTABLE)
      return()
    endif()
    set(build_cmd ${POWERSHELL_EXECUTABLE} -NoProfile -ExecutionPolicy Bypass
                  -File ${CMAKE_CURRENT_SOURCE_DIR}/${pkg_dir}/bundle/build.ps1)
  else()
    find_program(BASH_EXECUTABLE bash)
    if(NOT BASH_EXECUTABLE)
      return()
    endif()
    set(build_cmd ${BASH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${pkg_dir}/bundle/build.sh)
  endif()

  add_custom_command(
    OUTPUT ${pyz_path}
    COMMAND ${CMAKE_COMMAND} -E env
            BUNDLE_OUT_DIR=${out_dir}
            BUNDLE_VENV_DIR=${shiv_venv_dir}
            ${build_cmd}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${pkg_dir}
    DEPENDS ${pkg_sources} ${shiv_stamp}
    COMMENT "Building ${pyz_name}.pyz"
    VERBATIM
  )

  add_custom_target(${target_name} DEPENDS ${pyz_path})
endfunction()

add_pyz_bundle(dnn-benchmark-bundle dnn-benchmarking dnn-benchmark)
add_pyz_bundle(dnn-convert-shapes-bundle dnn-convert-shapes dnn-convert-shapes)
