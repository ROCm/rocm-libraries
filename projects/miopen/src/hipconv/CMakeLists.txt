cmake_minimum_required(VERSION 3.21)
project(hipconv LANGUAGES CXX HIP)

set(HIPCONV_NUM_SHARDS 8 CACHE STRING "Number of shards for parallel kernel compilation")
if(NOT HIPCONV_NUM_SHARDS MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR
        "HIPCONV_NUM_SHARDS must be a positive integer; got '${HIPCONV_NUM_SHARDS}'")
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD_REQUIRED)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    if(NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES gfx950)
    endif()
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
    find_package(hip REQUIRED)

    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GpuArch)

set(HIPCONV_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# The CRT deprecates getenv in favour of _dupenv_s.
#
# Directory-scoped, so it reaches the arch libraries and the shard generators without
# leaking to a consumer. A consumer's Windows toolchain typically sets this for its CXX
# targets and not its HIP ones, which is where the kernels compile.
if(WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# Experimental cdna4 depthwise 2D patch-Toeplitz kernel; off by default since it never
# wins the auto-pick.
#
# ON compiles it back in so its configs stay reachable via --config / the tuner. Must
# precede the arch subdirectories, since hipconv_add_arch_lib reads it.
option(HIPCONV_ENABLE_CDNA4_DW_2D_TOEPLITZ
    "Build the experimental cdna4 depthwise 2D patch-Toeplitz kernel (slower than 1D; not auto-selected)" OFF)

# Every arch in the tree is built, whatever this build targets: the set of arch
# directories is a property of the snapshot, the GPU list is not.
file(GLOB _hipconv_arch_directories RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "src/arch/*")
set(_arch_targets "")
foreach(_arch_dir IN LISTS _hipconv_arch_directories)
    # Skip tooling directories; an arch declares itself with a CMakeLists.
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_arch_dir}/CMakeLists.txt")
        continue()
    endif()
    add_subdirectory(${_arch_dir})
    get_property(_targets DIRECTORY ${_arch_dir} PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_target IN LISTS _targets)
        get_target_property(_name ${_target} HIPCONV_ARCH_NAME)
        if(_name)
            list(APPEND _arch_targets "${_target}")
        endif()
    endforeach()
endforeach()
if(NOT _arch_targets)
    message(FATAL_ERROR "hipconv: no architectures found under src/arch")
endif()

# ---- hipconv static library ----
set(_arch_registry "src/arch_registry.cpp")
hipconv_make_arch_registry("${_arch_targets}" "${_arch_registry}")

set(_hipconv_srcs
    src/conv2d_params.cpp
    src/conv_kernel.cpp
    src/kv_descriptor.cpp
    src/registry.cpp
    src/tolerance.cpp
    "${_arch_registry}"
)
if(HIPCONV_ENABLE_POINTWISE)
    find_package(hipblaslt REQUIRED)
    list(APPEND _hipconv_srcs
        src/pointwise/pointwise_conv.cpp
        src/pointwise/pointwise_fp16bf16.cpp
        src/pointwise/hipblaslt_matmul.cpp
    )
endif()

add_library(hipconv STATIC ${_hipconv_srcs} "${_reg_gen}")
set_source_files_properties(${_hipconv_srcs} "${_reg_gen}" PROPERTIES LANGUAGE HIP)
foreach(_target IN LISTS _arch_targets)
    target_sources(hipconv PRIVATE $<TARGET_OBJECTS:${_target}>)
endforeach()

target_include_directories(hipconv
    PUBLIC  include
    PRIVATE src
)
# Position-independent so the static lib can link into libMIOpen.so
set_target_properties(hipconv PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hipconv PUBLIC hip::device)
target_compile_options(hipconv PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
if(HIPCONV_ENABLE_POINTWISE)
    target_link_libraries(hipconv PUBLIC roc::hipblaslt)
    target_compile_definitions(hipconv PUBLIC HIPCONV_ENABLE_POINTWISE)
endif()
