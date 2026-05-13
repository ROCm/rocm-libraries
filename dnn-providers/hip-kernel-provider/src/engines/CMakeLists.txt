
if(ENABLE_ASM_SDPA_ENGINE)
    add_subdirectory(asm_sdpa_engine)
endif()

if(ENABLE_HIP_KERNEL_ENGINE)
    add_subdirectory(hip_kernel_engine)
endif()

# Add to the hip_kernel_provider_impl target the inlined kernels
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(KernelEmbedding)
# Generate embedded kernel sources and includes at configure time
embed_kernel_sources(
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_sources.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_sources.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_includes.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_includes.hpp
)


set(HIP_KERNEL_ENGINE_GENERATED_SOURCES 
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_sources.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/kernel_includes.cpp)

set_source_files_properties(${HIP_KERNEL_ENGINE_GENERATED_SOURCES} TARGET_DIRECTORY hip_kernel_provider_impl PROPERTIES COMPILE_OPTIONS "-Wno-overlength-strings")

target_sources(hip_kernel_provider_impl PRIVATE "${HIP_KERNEL_ENGINE_GENERATED_SOURCES}")

define_property(GLOBAL PROPERTY INLINED_KERNELS)
set_property(GLOBAL PROPERTY INLINED_KERNELS "${HIP_KERNEL_ENGINE_GENERATED_SOURCES}")


