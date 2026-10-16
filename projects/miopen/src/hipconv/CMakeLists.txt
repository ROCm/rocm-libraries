cmake_minimum_required(VERSION 3.21)

# The GPU hipconv offloads to when a build names none that hipconv serves.
#
# gfx942 is cdna3's target, and cdna3 has no kernels of its own, so every other
# architecture's entry guards discard their bodies on it. docs/architecture-guards.md says
# why the stand-in must be a GPU hipconv declares.
set(HIPCONV_BASELINE_GPU gfx942)

# Take the GPU list from GPU_TARGETS when nothing has already named the GPUs.
#
# GPU_TARGETS is ROCm's GPU list, set by the consumer or filled from amdgpu-arch by
# find_package(hip) when unset, and MIOpen enables no HIP language, so it names its
# GPUs this way alone. CMake's HIP language reads CMAKE_HIP_ARCHITECTURES and fills
# it from the build machine's own devices, so this has to precede project(). A
# consumer that sets CMAKE_HIP_ARCHITECTURES itself, as TheRock's toolchain does,
# skips this. Entries naming no offload target are dropped, hipconv serving no
# generic target. A '-strict' suffix names a specific silicon revision rather than a
# generic family, so it is kept.
if(NOT CMAKE_HIP_ARCHITECTURES AND GPU_TARGETS)
    set(_hipconv_gpus "")
    foreach(_hipconv_gpu IN LISTS GPU_TARGETS)
        string(REGEX REPLACE ":.*" "" _hipconv_logical "${_hipconv_gpu}")
        if(_hipconv_logical MATCHES "^gfx[0-9a-f]+(-strict)?$")
            list(APPEND _hipconv_gpus "${_hipconv_gpu}")
        else()
            message(STATUS
                "hipconv: skipping GPU_TARGETS entry '${_hipconv_gpu}' (not an offload target)")
        endif()
    endforeach()
    if(_hipconv_gpus)
        set(CMAKE_HIP_ARCHITECTURES "${_hipconv_gpus}")
    else()
        # Detecting the build machine's devices instead would compile kernels for GPUs
        # this build was never asked to serve.
        message(WARNING
            "hipconv: no GPU_TARGETS entry is an offload target; building for "
            "${HIPCONV_BASELINE_GPU}. "
            "Set CMAKE_HIP_ARCHITECTURES to name the GPUs hipconv should serve.")
        set(CMAKE_HIP_ARCHITECTURES ${HIPCONV_BASELINE_GPU})
    endif()
endif()

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

# The explicit-GEMM families (1x1 and patch embedding) are backed by hipBLASLt.
# Off by default so a build takes on that dependency only when it asks for it,
# and so a framework providing its own GEMM-based convolutions can drop the code.
#
# An option() for this was lost in the arch-registry rewrite (350e6ef), leaving the
# `if(HIPCONV_ENABLE_EXPLICIT_GEMM)` guards below reading an undeclared variable.
# Declaring it here restores `cmake -LH` discoverability and the misspelling check;
# the default matches what the build has actually been doing since then.
option(HIPCONV_ENABLE_EXPLICIT_GEMM
    "Build the hipBLASLt-backed explicit-GEMM convolution paths (1x1 and patch embedding)" OFF)

option(HIPCONV_LINE_TABLES "Emit DWARF line tables for device code (-gline-tables-only)" OFF)
if(HIPCONV_LINE_TABLES)
    string(APPEND CMAKE_HIP_FLAGS " -gline-tables-only")
endif()

# Device LTO costs up to 8x on the kernels this library exists to provide.
#
# Some ROCm clangs default the AMDGCN LTO mode to full, and under that pipeline the
# register allocator spills into direct_wgrad's software-pipelined row loop. The
# computation is unchanged and no test can see it. docs/rocm-device-lto-regression.md
# has the measurements; ROCm/llvm-project#4434 is the upstream report.
#
# Carried on the targets rather than in CMAKE_HIP_FLAGS because a consumer may compile
# these sources as CXX -- MIOpen does -- and then the HIP flags never reach them.
option(HIPCONV_OFFLOAD_LTO "Leave the compiler's device LTO default alone" OFF)
if(NOT HIPCONV_OFFLOAD_LTO)
    set(HIPCONV_DEVICE_COMPILE_OPTIONS -fno-offload-lto)
endif()

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
hipconv_set_stand_in_offload("${_arch_targets}")

# ---- hipconv static library ----
set(_arch_registry "src/arch_registry.cpp")
hipconv_make_arch_registry("${_arch_targets}" "${_arch_registry}")

set(_hipconv_srcs
    src/conv_params.cpp
    src/conv_kernel.cpp
    src/data_type_field.cpp
    src/direction_field.cpp
    src/kv_descriptor.cpp
    src/registry.cpp
    src/tolerance.cpp
    "${_arch_registry}"
)
if(HIPCONV_ENABLE_EXPLICIT_GEMM)
    find_package(hipblaslt REQUIRED)
    list(APPEND _hipconv_srcs
        src/explicit_gemm/explicit_gemm_algorithm.cpp
        src/explicit_gemm/pointwise/pointwise_kernel.cpp
        src/explicit_gemm/hipblaslt_matmul.cpp
        src/explicit_gemm/patch_embed/patch_embed.cpp
    )
endif()

# hipconv::version(), regenerated at build time so a new commit needs no reconfigure;
# cmake/write_version.cmake says where the version comes from.
set(_version_cpp "${CMAKE_CURRENT_BINARY_DIR}/version.cpp")
add_custom_target(hipconv_version
    COMMAND "${CMAKE_COMMAND}"
            "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DTEMPLATE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.cpp.in"
            "-DOUTPUT=${_version_cpp}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/write_version.cmake"
    BYPRODUCTS "${_version_cpp}"
    VERBATIM)
list(APPEND _hipconv_srcs "${_version_cpp}")

add_library(hipconv STATIC ${_hipconv_srcs} "${_reg_gen}")
set_source_files_properties(${_hipconv_srcs} "${_reg_gen}" PROPERTIES LANGUAGE HIP)
set_source_files_properties("${_version_cpp}" PROPERTIES GENERATED TRUE)
add_dependencies(hipconv hipconv_version)
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
target_compile_options(hipconv PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden
                                       ${HIPCONV_DEVICE_COMPILE_OPTIONS})
if(HIPCONV_ENABLE_EXPLICIT_GEMM)
    target_link_libraries(hipconv PUBLIC roc::hipblaslt)
    target_compile_definitions(hipconv PUBLIC HIPCONV_ENABLE_EXPLICIT_GEMM)
endif()
