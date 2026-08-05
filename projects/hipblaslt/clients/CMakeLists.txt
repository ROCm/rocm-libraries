# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(hipblaslt-bench)
add_library(hipblaslt-clients-common STATIC)

target_include_directories(hipblaslt-clients-common
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../tensilelite/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../tensilelite>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/amd_detail/include>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../tensilelite"
)

if(HIPBLASLT_ENABLE_COVERAGE)
    target_include_directories(hipblaslt-clients-common
        PUBLIC
            ${PROJECT_SOURCE_DIR}/library/src/amd_detail
            ${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include
            ${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/include
            ${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/rocroller/include
            ${PROJECT_SOURCE_DIR}/tensilelite/rocisa
            ${PROJECT_SOURCE_DIR}/../../shared/origami/include
    )
endif()

add_library(hipblaslt::hipblaslt-clients-common ALIAS hipblaslt-clients-common)

if(HIPBLASLT_ENABLE_BLIS)
    target_link_libraries(hipblaslt-clients-common PUBLIC BLIS::BLIS)
    target_compile_definitions(hipblaslt-clients-common PUBLIC HIPBLASLT_ENABLE_BLIS)
endif()

# LAPACK_LIBRARIES is a plain Fortran static library. With gfortran this has
# always linked cleanly, but with LLVM Flang (see invoke's Fortran-compiler
# auto-detection) the CHARACTER-argument routines it uses (e.g. zlange/clange/
# dlange) pull in Flang runtime symbols (e.g. _FortranACharacterCompareScalar1)
# that amdclang++ never links automatically, since Fortran isn't the link
# driver for these C++/HIP targets. Search every root we can derive for the
# selected Flang install (resolving symlinks, since invoke may hand us e.g.
# /opt/rocm/bin/amdflang -> .../llvm/bin/flang-N), plus HIP_CLANG_ROOT, for
# either the older split FortranRuntime+FortranDecimal or the newer merged
# flang_rt.runtime. Prefer linking the absolute path find_library resolves:
# amdclang++ isn't always installed under the same directory as flang (e.g.
# .../bin/amdclang++ vs .../llvm/bin/flang), so its default linker search
# doesn't reliably cover the runtime's directory, and a bare -l name can fail
# to link even though the file exists (see git history for a CI regression
# from trying that). Only fall back to linking by bare name when we can't
# resolve any absolute path at all, on the chance amdclang++'s default search
# happens to cover a packaging layout our probe above can't find (this is
# what several of Math CI's precheckin images need). This mirrors the
# approach used in hipsparselt's clients. Leave the working gfortran path
# untouched.
set(HIPBLASLT_FORTRAN_RUNTIME_LIBRARIES)
if(CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
    get_filename_component(_hipblaslt_fortran_real "${CMAKE_Fortran_COMPILER}" REALPATH)
    get_filename_component(_hipblaslt_fortran_bin_dir "${_hipblaslt_fortran_real}" DIRECTORY)
    get_filename_component(_hipblaslt_fortran_root_dir "${_hipblaslt_fortran_bin_dir}" DIRECTORY)
    set(_hipblaslt_fortran_search_dirs "${_hipblaslt_fortran_root_dir}/lib")
    if(HIP_CLANG_ROOT)
        list(APPEND _hipblaslt_fortran_search_dirs "${HIP_CLANG_ROOT}/lib")
    endif()
    find_library(HIPBLASLT_FORTRANRUNTIME_LIBRARY NAMES FortranRuntime
        PATHS ${_hipblaslt_fortran_search_dirs} NO_DEFAULT_PATH)
    if(HIPBLASLT_FORTRANRUNTIME_LIBRARY)
        find_library(HIPBLASLT_FORTRANDECIMAL_LIBRARY NAMES FortranDecimal
            PATHS ${_hipblaslt_fortran_search_dirs} NO_DEFAULT_PATH)
        if(HIPBLASLT_FORTRANDECIMAL_LIBRARY)
            set(HIPBLASLT_FORTRAN_RUNTIME_LIBRARIES
                ${HIPBLASLT_FORTRANRUNTIME_LIBRARY} ${HIPBLASLT_FORTRANDECIMAL_LIBRARY})
        else()
            set(HIPBLASLT_FORTRAN_RUNTIME_LIBRARIES -lFortranRuntime -lFortranDecimal)
        endif()
    else()
        # Newer LLVM Flang releases merged FortranRuntime/FortranDecimal into
        # a single flang_rt.runtime library.
        find_library(HIPBLASLT_FLANGRT_LIBRARY NAMES flang_rt.runtime
            PATHS ${_hipblaslt_fortran_search_dirs} NO_DEFAULT_PATH)
        if(HIPBLASLT_FLANGRT_LIBRARY)
            set(HIPBLASLT_FORTRAN_RUNTIME_LIBRARIES ${HIPBLASLT_FLANGRT_LIBRARY})
        else()
            set(HIPBLASLT_FORTRAN_RUNTIME_LIBRARIES -lflang_rt.runtime)
        endif()
    endif()
    unset(_hipblaslt_fortran_real)
    unset(_hipblaslt_fortran_bin_dir)
    unset(_hipblaslt_fortran_root_dir)
    unset(_hipblaslt_fortran_search_dirs)
endif()

target_link_libraries(hipblaslt-clients-common
    PUBLIC
        roc::hipblaslt
        roc::host-validation-adapters
        OpenMP::OpenMP_CXX
        ${LAPACK_LIBRARIES}
        ${HIPBLASLT_FORTRAN_RUNTIME_LIBRARIES}
    PRIVATE
        hip::device
        roc::host-validation-blas
        roc::tensilelite-host
)

if(HIPBLASLT_ENABLE_MXDATAGENERATOR)
    # PUBLIC because public headers such as testing_matmul.hpp guard
    # `<mxDataGen.hpp>` on `HIPBLASLT_ENABLE_MXDATAGENERATOR`, so downstream
    # targets (hipblaslt-test, hipblaslt-bench, ...) need to see the macro,
    # the include path, and the C++20 floor all the way through.
    target_link_libraries(hipblaslt-clients-common PUBLIC hipblaslt::mxdatagen)
endif()

if(HIPBLASLT_ENABLE_ROCROLLER)
    target_compile_definitions(hipblaslt-clients-common PRIVATE HIPBLASLT_USE_ROCROLLER)
endif()


if(HIPBLASLT_ENABLE_ASAN)
    hipblaslt_target_configure_sanitizers(hipblaslt-clients-common PUBLIC)
endif()

target_link_libraries(hipblaslt-bench
    PRIVATE
        hip::device
        hipblaslt::hipblaslt-clients-common
        OpenMP::OpenMP_CXX
)

target_include_directories(hipblaslt-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/bench/include")

if(HIPBLASLT_ENABLE_OPENMP)
    target_link_libraries(hipblaslt-bench PRIVATE OpenMP::OpenMP_CXX)
endif()

# `clients/common/CMakeLists.txt` itself only declares the mxdatagen helper
# (which the project root already added when needed); the `include` and
# `src` subdirectories populate hipblaslt-clients-common's sources, so they
# are added directly here, avoiding a second `add_subdirectory(common)` that
# would clash with the root-level one.
add_subdirectory(common/include)
add_subdirectory(common/src)
add_subdirectory(bench)

add_executable(hipblaslt-api-overhead "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_api_overhead.cpp")
target_link_libraries(hipblaslt-api-overhead PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-api-overhead PRIVATE ROCBLASLT_INTERNAL_API)
target_include_directories(hipblaslt-api-overhead PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/samples/common")

add_executable(hipblaslt-bench-groupedgemm-fixed-mk "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_groupedgemm_fixed_mk.cpp")
target_link_libraries(hipblaslt-bench-groupedgemm-fixed-mk
    PRIVATE
        hipblaslt-clients-common
        hip::device
        roc::host-validation
)
target_compile_definitions(hipblaslt-bench-groupedgemm-fixed-mk PRIVATE ROCBLASLT_INTERNAL_API)

add_executable(hipblaslt-bench-extop-layernorm "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_layernorm.cpp")
target_link_libraries(hipblaslt-bench-extop-layernorm PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-layernorm PRIVATE ROCBLASLT_INTERNAL_API)

add_executable(hipblaslt-bench-extop-matrixtransform "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_matrixtransform.cpp")
target_link_libraries(hipblaslt-bench-extop-matrixtransform PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-matrixtransform PRIVATE ROCBLASLT_INTERNAL_API)

add_executable(hipblaslt-bench-extop-softmax "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_softmax.cpp")
target_link_libraries(hipblaslt-bench-extop-softmax PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-softmax PRIVATE ROCBLASLT_INTERNAL_API)

add_executable(hipblaslt-bench-extop-amax "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_extop_amax.cpp")
target_link_libraries(hipblaslt-bench-extop-amax PRIVATE hipblaslt-clients-common hip::device)
target_compile_definitions(hipblaslt-bench-extop-amax PRIVATE ROCBLASLT_INTERNAL_API)

# CU-contention cotenant helper (clients/scripts/cotenant). A standalone HIP
# binary, compiled as CXX so it picks up GPU_TARGETS via hip::device like the
# other clients (the project does not enable the HIP language). POSIX-only
# (unistd.h, pause(), usleep(), getpid()), so it is not built on Windows.
if(NOT WIN32)
    add_executable(hipblaslt-cotenant-kernel "${CMAKE_CURRENT_SOURCE_DIR}/scripts/cotenant/hipblaslt-cotenant-kernel.hip")
    set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/cotenant/hipblaslt-cotenant-kernel.hip"
        PROPERTIES LANGUAGE CXX)
    target_link_libraries(hipblaslt-cotenant-kernel PRIVATE hip::device)
    target_compile_features(hipblaslt-cotenant-kernel PRIVATE cxx_std_17)
endif()

if(HIPBLASLT_ENABLE_YAML)
    find_package(LLVM REQUIRED)
    add_executable(hipblaslt-sequence "${CMAKE_CURRENT_SOURCE_DIR}/bench/src/client_sequence.cpp")
    target_link_libraries(hipblaslt-sequence PRIVATE hipblaslt-clients-common hip::device)
    target_compile_definitions(hipblaslt-sequence PRIVATE ROCBLASLT_INTERNAL_API)
    target_link_libraries(hipblaslt-sequence PRIVATE LLVMObjectYAML)
    target_include_directories(hipblaslt-sequence PRIVATE ${LLVM_INCLUDE_DIRS})
endif()
if(BUILD_TESTING OR HIPBLASLT_BUILD_TESTING)
    find_package(GTest CONFIG REQUIRED)
    add_executable(hipblaslt-test)
    if(amd_smi_FOUND)
        target_link_libraries(hipblaslt-test PRIVATE amd_smi)
    endif()
    target_link_libraries(hipblaslt-test
        PRIVATE
            hip::device
            hipblaslt::hipblaslt-clients-common
            roc::host-validation
            roc::tensilelite-host
            GTest::gtest
            OpenMP::OpenMP_CXX
            ${CMAKE_DL_LIBS}
    )
    target_compile_definitions(hipblaslt-test PRIVATE GOOGLE_TEST)
    if(HIPBLASLT_ENABLE_ROCROLLER)
        target_compile_definitions(hipblaslt-test PRIVATE HIPBLASLT_USE_ROCROLLER)
    endif()
    if(HIPBLASLT_ENABLE_COVERAGE)
        target_compile_definitions(hipblaslt-test PRIVATE CODE_COVERAGE)
        # For coverage testing, link against hipblaslt's object files to access internal symbols
        # This ensures we test the real library code, not stubs
        target_link_libraries(hipblaslt-test PRIVATE
            $<TARGET_OBJECTS:hipblaslt>
            roc::tensilelite-host
            ${CMAKE_DL_LIBS}
        )
        # Add roctracer if HIPBLASLT_ENABLE_MARKER is enabled
        if(HIPBLASLT_ENABLE_MARKER)
            target_link_libraries(hipblaslt-test PRIVATE ${rocTracer})
        endif()
        # Add msgpack if not using YAML
        if(NOT HIPBLASLT_ENABLE_YAML)
            target_link_libraries(hipblaslt-test PRIVATE ${hipblaslt_msgpack_target})
        endif()
        # Add rocroller dependencies if enabled
        if(HIPBLASLT_ENABLE_ROCROLLER)
            target_link_libraries(hipblaslt-test PRIVATE
                $<TARGET_OBJECTS:hipblaslt-rocroller>
                rocroller
            )
        endif()
    endif()

    add_subdirectory(tests)
endif()
if(HIPBLASLT_ENABLE_SAMPLES)
    add_subdirectory(samples)
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tests/hipblaslt_gentest.py" "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt_gentest.py" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tests/data/hipblaslt_common.yaml" "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt_common.yaml" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tests/data/hipblaslt_template.yaml" "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt_template.yaml" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/bench/sequence.yaml" "${CMAKE_CURRENT_BINARY_DIR}/sequence.yaml" COPYONLY)
if(NOT WIN32)
    # Place the launcher next to the built kernel so it auto-finds the binary in
    # the build tree (matching the installed bin/ layout) without --binary.
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/scripts/cotenant/hipblaslt-cotenant" "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt-cotenant" COPYONLY)
endif()
