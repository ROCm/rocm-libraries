# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(hipdnn_list_engines ListEngines.cpp)
target_link_libraries(hipdnn_list_engines PRIVATE hipdnn_backend)
target_compile_definitions(hipdnn_list_engines PRIVATE HIPDNN_BACKEND_COMPILATION)
install(TARGETS hipdnn_list_engines RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# ---------------------------------------------------------------------------
# Python tool bundles (shiv .pyz zipapps)
#
# Each bundle is a single-file Python zipapp that ships its tool plus all PyPI
# dependencies. Bundles are built on demand via custom targets — they are NOT
# part of the default build because shiv needs network access to pip-install.
#
# Targets:
#   ninja dnn-convert-shapes-bundle  -> dnn-convert-shapes/bundle/dnn-convert-shapes.pyz
#   ninja dnn-benchmark-bundle       -> dnn-benchmarking/bundle/dnn-benchmark.pyz
# ---------------------------------------------------------------------------

find_program(BASH_EXECUTABLE bash)
if(BASH_EXECUTABLE)
  # Wire a shiv .pyz bundle for the Python tool under ${script_dir}.
  #   target_name : custom target name (e.g. dnn-benchmark-bundle)
  #   script_dir  : path (relative to this CMakeLists) to the tool package
  #   output_name : informational label printed in the build log
  function(add_pyz_bundle target_name script_dir output_name)
    add_custom_target(${target_name}
      COMMAND ${BASH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${script_dir}/bundle/build.sh
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${script_dir}
      COMMENT "Building ${output_name} (shiv .pyz bundle)"
      VERBATIM
    )
  endfunction()

  add_pyz_bundle(dnn-convert-shapes-bundle dnn-convert-shapes dnn-convert-shapes.pyz)
  add_pyz_bundle(dnn-benchmark-bundle      dnn-benchmarking   dnn-benchmark.pyz)
endif()
