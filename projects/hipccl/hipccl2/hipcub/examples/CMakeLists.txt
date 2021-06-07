# Create meta targets that build all examples for a single configuration:
foreach(cub_target IN LISTS CUB_TARGETS)
  cub_get_target_property(config_prefix ${cub_target} PREFIX)
  set(config_meta_target ${config_prefix}.examples)
  add_custom_target(${config_meta_target})
  add_dependencies(${config_prefix}.all ${config_meta_target})
endforeach()

## cub_add_example
#
# Add an example executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the example
#   target. Useful for post-processing target information per-backend.
# example_name: The name of the example minus "<config_prefix>.example." For
#   instance, examples/vector.cu will be "vector", and examples/cuda/copy.cu
#   would be "cuda.copy".
# example_src: The source file that implements the example.
# cub_target: The reference cub target with configuration information.
#
function(add_hipcub_example EXAMPLE_NAME EXAMPLE_SOURCES)
  list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
  get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
  add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
  target_include_directories(${EXAMPLE_TARGET} SYSTEM BEFORE
    PUBLIC
      ${GEXAMPLE_INCLUDE_DIRS}
  )
  target_link_libraries(${EXAMPLE_TARGET}
    PRIVATE
      ${GEXAMPLE_BOTH_LIBRARIES}
      hipcub
  )

  if(HIP_COMPILER STREQUAL "nvcc")
    set_property(TARGET ${EXAMPLE_TARGET} PROPERTY CUDA_STANDARD 14)
    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    target_link_libraries(${EXAMPLE_TARGET}
      PRIVATE
        hipcub_cub
    )
  endif()

  set_target_properties(${EXAMPLE_TARGET}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example/hipcub"
  )
  add_test(
    NAME ${EXAMPLE_NAME}
    COMMAND ${EXAMPLE_TARGET}
  )
endfunction()

add_subdirectory(block)
add_subdirectory(device)
