# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(hipdnn_list_engines ListEngines.cpp)
target_link_libraries(hipdnn_list_engines PRIVATE hipdnn_backend)
target_compile_definitions(hipdnn_list_engines PRIVATE HIPDNN_BACKEND_COMPILATION)
install(TARGETS hipdnn_list_engines RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Python tool bundles (shiv .pyz zipapps)
find_program(BASH_EXECUTABLE bash)
if(BASH_EXECUTABLE)
  add_custom_target(dnn-benchmark-bundle
    COMMAND ${BASH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/dnn-benchmarking/bundle/build.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dnn-benchmarking
    COMMENT "Building dnn-benchmark.pyz"
  )
  add_custom_target(dnn-convert-shapes-bundle
    COMMAND ${BASH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/dnn-convert-shapes/bundle/build.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dnn-convert-shapes
    COMMENT "Building dnn-convert-shapes.pyz"
  )
endif()
