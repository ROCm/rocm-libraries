
message("Building client program")

# Header only, not visible to library users, system package not required
rocroller_add_dependency(CLI11 VERSION 2.1.2)

add_executable(gemm
  GraphInspector.cpp
  gemm.cpp
  GEMMParameters.cpp
  visualize.cpp
)

target_compile_options(gemm PUBLIC ${EXTRA_COMPILE_OPTIONS})
target_link_options(gemm PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(
  gemm
  PUBLIC
  rocroller_interface
  common_test_utilities
  CLI11::CLI11
)

set_target_properties(gemm
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin/client"
)

add_executable(codegen_stress
  codegen_stress_test.cpp
)

target_compile_options(codegen_stress PUBLIC ${EXTRA_COMPILE_OPTIONS})
target_link_options(codegen_stress PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(
  codegen_stress
  PUBLIC
  rocroller_interface
  CLI11::CLI11
)

set_target_properties( codegen_stress
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin/client"
)

add_custom_target(all_clients)
add_dependencies(all_clients gemm codegen_stress)
