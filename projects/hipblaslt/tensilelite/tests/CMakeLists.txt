# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(GTest REQUIRED)

add_executable(tensilelite-tests)
add_subdirectory(configs)

set(TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
file(MAKE_DIRECTORY "${TEST_DATA_DIR}")

foreach(FILE ${SOLUTION_LIBRARY_FILES})
    if(EXISTS "${FILE}.gz")
        execute_process(COMMAND gzip --keep --decompress --force "${FILE}.gz")
    endif()
    file(COPY ${FILE} DESTINATION "${TEST_DATA_DIR}")
endforeach()

target_sources(tensilelite-tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AlmostEqual_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/BenchmarkTimer_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CachingLibrary_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ContractionSelectionLibrary_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CuCount_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CustomKernel_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DataTypes_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DataInit_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ExactLogicLibrary_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/HardwareMonitor_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/HybridAssignmentPolicy_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../clients/common/src/hipblaslt_bench_options.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/GEMMKernelTest.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/GEMMKernelTest_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/RunGEMMKernelTest.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/RunGEMMKernelTest_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/TestData.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/TestUtils.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LaunchDimGuard_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LogReporter_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MXScalePadding_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/PciChipId_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/PersistentArgumentLayout_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/PersistentExecutionPolicySerialization_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Predicates_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/PredictionLibrary_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ReferenceFastPath_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RangeLibrary_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ResultComparison_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ReferenceMXFastPath_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionValidation_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/StreamKLaunchSummary_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/SynchronizerResidue_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/SynchronizerValidatorReporting_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TensorDescriptor_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TestData.cpp"
)

# MasterSolutionLibraryMapping_test exercises TensileLite::LoadLibraryMapping,
# which is only defined when MSGPACK is the active serialization format
# (see tensilelite/src/Tensile.cpp). The test also writes msgpack mapping
# files directly via <msgpack.hpp>, so it has no meaning in YAML builds.
if(NOT TENSILELITE_ENABLE_YAML)
    target_sources(tensilelite-tests
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/MasterSolutionLibraryMapping_test.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/CompressedLibraryLoad_test.cpp"
            # The indexed library layout is msgpack-only: its solution blob is a
            # msgpack BIN, and mapRawBytes has no YAML implementation.
            "${CMAKE_CURRENT_SOURCE_DIR}/IndexedLibraryLoad_test.cpp"
    )
endif()

if(TENSILELITE_ENABLE_YAML)
    target_include_directories(tensilelite-tests PRIVATE ${LLVM_INCLUDE_DIRS})
    target_link_libraries(tensilelite-tests PRIVATE LLVMObjectYAML)
endif()

target_include_directories(tensilelite-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(tensilelite-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../clients/common/include")
target_include_directories(tensilelite-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../client/include")

target_link_libraries(tensilelite-tests
    PUBLIC
        roc::tensilelite-host
    PRIVATE
        hip::device
        tensilelite::client-common
        ${CMAKE_DL_LIBS}
        ${hipblaslt_msgpack_target}
        ZLIB::ZLIB
)

if(HIPBLASLT_ENABLE_MXDATAGENERATOR)
    target_sources(tensilelite-tests
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/MXDataGen_test.cpp"
    )
    target_link_libraries(tensilelite-tests PRIVATE hipblaslt::mxdatagen)
endif()

target_link_libraries(tensilelite-tests PUBLIC GTest::gtest)

if(HIPBLASLT_ENABLE_YAML)
    target_link_libraries(tensilelite-tests PRIVATE LLVMObjectYAML)
    target_include_directories(tensilelite-tests PRIVATE ${LLVM_INCLUDE_DIRS})
endif()

# LaunchDimGuard_test asserts different outcomes for a cluster launch depending
# on whether SolutionAdapter::launchKernel was compiled with the cluster path,
# so it needs the same answer the library got. Reuses the cached result of the
# check in ../CMakeLists.txt rather than repeating the feature test; the
# definition stays PRIVATE on both targets.
if(HIP_HAS_CLUSTER_LAUNCH)
    target_compile_definitions(tensilelite-tests PRIVATE HIP_HAS_CLUSTER_LAUNCH)
endif()

if(HIPBLASLT_ENABLE_OPENMP)
    target_link_libraries(tensilelite-tests PRIVATE OpenMP::OpenMP_CXX)
endif()

# Optional: CTest test categorization + install wiring for TheRock, mirroring
# hipBLASLt's own clients/tests/CMakeLists.txt pattern. Skipped when shared/
# is not present (e.g. standalone hipblaslt checkout or sparse checkout in CI).
#
# tensilelite-tests installs into its own subdirectory (not top-level bin/)
# because TestData::TestData() resolves fixture data relative to the
# *executable's own location* (<exe_dir>/data, see TestData.cpp), not the
# CTest working directory. Keeping the binary, its CTestTestfile.cmake, and
# data/ together in one install subdir satisfies that lookup as-is, with no
# changes to TestData.cpp.
set(TENSILELITE_TESTS_INSTALL_SUBDIR "hipblaslt-tensilelite")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml")
    if(DEFINED ROCM_LIBRARIES_ROOT AND EXISTS "${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
        include("${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
        # Per shared/ctest/README.md, enable_testing() must run before
        # apply_test_category_labels() or its add_test() calls have no
        # effect. The unconditional enable_testing() later in this file
        # (for the cpu-gemm-driver tests) is too late for this block, so
        # call it here too; enable_testing() is idempotent.
        enable_testing()
        message(STATUS "tensilelite-tests: YAML-based test categorization")
        file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

        # Install-time CTestTestfile for TheRock: relative paths so ctest can run from installed location.
        set(TENSILELITE_TESTS_CTEST_INSTALL_FILE "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake")
        file(WRITE "${TENSILELITE_TESTS_CTEST_INSTALL_FILE}"
[=[
# This is a test file generated by tensilelite-tests for install time.
# Tests are defined with relative paths to work in the installed location.
]=]
        )
        apply_test_category_labels(
            tensilelite-tests
            "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml"
            "${PROJECT_BINARY_DIR}/staging"
            "${TENSILELITE_TESTS_CTEST_INSTALL_FILE}"
            INSTALL_EXECUTABLE "./tensilelite-tests"
        )

        # Note: rocm_install(TARGETS ...) always installs executables to the
        # standard CMAKE_INSTALL_BINDIR; its underlying rocm_install_targets()
        # has no DESTINATION keyword for TARGETS mode. Use plain install()
        # here so tensilelite-tests lands in its own subdir alongside data/
        # and CTestTestfile.cmake (see TEST_DATA_DIR comment above).
        #
        # TheRock's therock_global_post_subproject.cmake computes each
        # executable's INSTALL_RPATH from scratch after every subproject
        # configure (see cmake/therock_global_post_subproject.cmake), assuming
        # by default that executables live directly in bin/. That pass
        # replaces whatever plain CMake INSTALL_RPATH this file might set, so
        # a manual $ORIGIN/../../lib-style override here would be silently
        # discarded. Instead, tell it the real (nested) install location via
        # THEROCK_INSTALL_RPATH_ORIGIN, mirroring e.g.
        # core/post_hook_amdsmi.cmake's amdsmitst -> share/amd_smi/tests.
        set_property(TARGET tensilelite-tests PROPERTY
            THEROCK_INSTALL_RPATH_ORIGIN "${CMAKE_INSTALL_BINDIR}/${TENSILELITE_TESTS_INSTALL_SUBDIR}"
        )
        install(
            TARGETS tensilelite-tests
            RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}/${TENSILELITE_TESTS_INSTALL_SUBDIR}"
            COMPONENT tests
        )
        rocm_install(
            DIRECTORY "${TEST_DATA_DIR}/"
            DESTINATION "${CMAKE_INSTALL_BINDIR}/${TENSILELITE_TESTS_INSTALL_SUBDIR}/data"
            COMPONENT tests
        )
        # Install CTestTestfile.cmake alongside the binary+data for TheRock
        # (relocatable tests; run ctest from that dir).
        install(
            FILES "${TENSILELITE_TESTS_CTEST_INSTALL_FILE}"
            DESTINATION "${CMAKE_INSTALL_BINDIR}/${TENSILELITE_TESTS_INSTALL_SUBDIR}"
            COMPONENT tests
            RENAME "CTestTestfile.cmake"
        )
    else()
        message(STATUS "tensilelite-tests: shared/ctest not found (e.g. standalone checkout), falling back to gtest_discover_tests()")
        include(GoogleTest)
        gtest_discover_tests(tensilelite-tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} TIMEOUT 60)
    endif()
else()
    message(STATUS "tensilelite-tests: No test_categories.yaml (CTest categories not used)")
endif()

enable_testing()

#################################################
#        Tests for cpu-gemm-driver              #
#################################################

function(add_cpu_gemm_test TEST_NAME)
    cmake_parse_arguments(ARG "" "" "ARGS" ${ARGN})

    add_test(
        NAME "CPUGemm.${TEST_NAME}"
        COMMAND cpu-gemm-driver ${ARG_ARGS}
    )
endfunction()

# Default test sizes. Non-tile-aligned to stress remainder handling.
# f32, f64, and f16/bf16 use different sizes to increase coverage diversity.
set(F32_SIZES -M 133 -N 147 -K 165)
set(F64_SIZES -M 137 -N 149 -K 163)
set(F16_SIZES -M 131 -N 141 -K 151)
set(ADDITIONAL_SIZES -M 102 -N 103 -K 105)

function(get_test_sizes TYPE OUT_SIZES)
    if("${TYPE}" STREQUAL "f32")
        set(${OUT_SIZES} ${F32_SIZES} PARENT_SCOPE)
    else()
        set(${OUT_SIZES} ${F16_SIZES} PARENT_SCOPE)
    endif()
endfunction()

# Helper: convert transpose tag (NN/TN/NT/TT) to flags
macro(get_trans_flags TRANS OUT_FLAGS)
    set(${OUT_FLAGS} "")
    if("${TRANS}" MATCHES "^T")
        list(APPEND ${OUT_FLAGS} --transA)
    endif()
    if("${TRANS}" MATCHES "T$")
        list(APPEND ${OUT_FLAGS} --transB)
    endif()
endmacro()

# Run all tests against both fast and slow paths
foreach(PATH fast slow)
    if("${PATH}" STREQUAL "fast")
        set(PATH_FLAGS --tryFastPath)
    else()
        set(PATH_FLAGS "")
    endif()

    # Basic sanity, transpose combos, odd sizes, type coverage
    add_cpu_gemm_test(${PATH}_F32_Sanity
        ARGS ${F32_SIZES} --type f32 --validate ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_F32_ADDITIONAL_SIZES
        ARGS ${ADDITIONAL_SIZES} --type f32 --validate ${PATH_FLAGS}
    )
    foreach(TYPE f32 f16 bf16)
        get_test_sizes(${TYPE} SIZES)
        foreach(TRANS NN TN NT TT)
            get_trans_flags(${TRANS} TFLAGS)
            add_cpu_gemm_test(${PATH}_${TYPE}_${TRANS}
                ARGS ${SIZES} ${TFLAGS} --type ${TYPE} --validate ${PATH_FLAGS}
            )
        endforeach()
    endforeach()

    # Cross-validate: features per type
    foreach(TYPE f32 f16 bf16)
        get_test_sizes(${TYPE} SIZES)
        add_cpu_gemm_test(${PATH}_${TYPE}_Beta
            ARGS ${SIZES} --type ${TYPE} --beta 1.0 --validate ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_${TYPE}_Bias
            ARGS ${SIZES} --type ${TYPE} --validate --bias ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_${TYPE}_AllFeatures
            ARGS ${SIZES} --type ${TYPE} --validate --bias --activation relu --scaleAlphaVec ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_${TYPE}_TN_AllFeatures
            ARGS ${SIZES} --transA --type ${TYPE} --validate --bias --activation relu --scaleAlphaVec ${PATH_FLAGS}
        )
    endforeach()

    # F32-only individual feature tests
    add_cpu_gemm_test(${PATH}_f32_Relu
        ARGS ${F32_SIZES} --type f32 --validate --activation relu ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_f32_ScaleAlphaVec
        ARGS ${F32_SIZES} --type f32 --validate --scaleAlphaVec ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_f32_Beta_AllFeatures
        ARGS ${F32_SIZES} --type f32 --beta 0.5 --validate --bias --activation relu --scaleAlphaVec ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_f32_ScaleAlphaVec_ColN
        ARGS ${F32_SIZES} --type f32 --validate --scaleAlphaVec --factorDim 1 ${PATH_FLAGS}
    )

    # Edge cases: dimensions at or below tile size (BLOCK_M=32, BLOCK_N=32, BLOCK_K=8)
    foreach(CASE "M1;1;147;165" "N1;133;1;165" "K1;133;147;1" "1x1;1;1;1" "ExactBlock;32;32;8")
        list(GET CASE 0 NAME)
        list(GET CASE 1 EM)
        list(GET CASE 2 EN)
        list(GET CASE 3 EK)
        add_cpu_gemm_test(${PATH}_f32_${NAME}
            ARGS -M ${EM} -N ${EN} -K ${EK} --type f32 --validate ${PATH_FLAGS}
        )
    endforeach()

    # ScaleAB tests
    foreach(MODE Scalar Vector)
        foreach(TYPE f32 f16 bf16)
            get_test_sizes(${TYPE} SIZES)
            add_cpu_gemm_test(${PATH}_${TYPE}_ScaleAB_${MODE}
                ARGS ${SIZES} --type ${TYPE} --validate --useScaleAB ${MODE} ${PATH_FLAGS}
            )
        endforeach()
        add_cpu_gemm_test(${PATH}_f32_ScaleAB_${MODE}_AllFeatures
            ARGS ${F32_SIZES} --type f32 --beta 0.5 --validate --useScaleAB ${MODE} --bias --activation relu --scaleAlphaVec ${PATH_FLAGS}
        )
        # ScaleAB with factorDim=1 (column-dimension ScaleAlphaVec)
        add_cpu_gemm_test(${PATH}_f32_ScaleAB_${MODE}_ColN
            ARGS ${F32_SIZES} --type f32 --validate --useScaleAB ${MODE} --scaleAlphaVec --factorDim 1 ${PATH_FLAGS}
        )
    endforeach()
endforeach()

add_test(
    NAME "CPUGemm.validation_InjectedFailure_Rejected"
    COMMAND cpu-gemm-driver -M 8 -N 8 -K 8 --type f32 --validate --tryFastPath --injectValidationFailure
)
set_tests_properties("CPUGemm.validation_InjectedFailure_Rejected" PROPERTIES WILL_FAIL TRUE)

#################################################
#        FP8 tests (fast & slow)                #
#################################################
if(NOT WIN32)
    foreach(TYPE f8 bf8 f8fnuz bf8fnuz)
        foreach(PATH fast slow)
            if("${PATH}" STREQUAL "fast")
                set(PATH_FLAGS --tryFastPath)
            else()
                set(PATH_FLAGS "")
            endif()

            # Transpose combos
            foreach(TRANS NN TN NT TT)
                get_trans_flags(${TRANS} TFLAGS)
                add_cpu_gemm_test(${PATH}_${TYPE}_${TRANS}
                    ARGS ${F16_SIZES} ${TFLAGS} --type ${TYPE} --validate ${PATH_FLAGS}
                )
            endforeach()

            # Feature tests
            add_cpu_gemm_test(${PATH}_${TYPE}_Beta
                ARGS ${F16_SIZES} --type ${TYPE} --beta 1.0 --validate ${PATH_FLAGS}
            )
            add_cpu_gemm_test(${PATH}_${TYPE}_Beta_Half
                ARGS ${F16_SIZES} --type ${TYPE} --beta 0.5 --validate ${PATH_FLAGS}
            )
            add_cpu_gemm_test(${PATH}_${TYPE}_Bias
                ARGS ${F16_SIZES} --type ${TYPE} --validate --bias ${PATH_FLAGS}
            )
            add_cpu_gemm_test(${PATH}_${TYPE}_AllFeatures
                ARGS ${F16_SIZES} --type ${TYPE} --validate --bias --activation relu --scaleAlphaVec ${PATH_FLAGS}
            )
            add_cpu_gemm_test(${PATH}_${TYPE}_TN_AllFeatures
                ARGS ${F16_SIZES} --transA --type ${TYPE} --validate --bias --activation relu --scaleAlphaVec ${PATH_FLAGS}
            )

            # ScaleAB tests
            foreach(MODE Scalar Vector)
                add_cpu_gemm_test(${PATH}_${TYPE}_ScaleAB_${MODE}
                    ARGS ${F16_SIZES} --type ${TYPE} --validate --useScaleAB ${MODE} ${PATH_FLAGS}
                )
            endforeach()
        endforeach()
    endforeach()
endif()

#################################################
# Mixed-input-type FP8 tests (fast & slow path).#
# Reproduces production failure where A and B   #
# have different storage types but share a      #
# narrower MAC compute-input type (the kernel   #
# quantizes the wider operand for F8 MFMA).     #
# See: sk_f8ngemm_quick.yaml F8NH_F8NSS NN.     #
#################################################
if(NOT WIN32)
    foreach(PATH fast slow)
        if("${PATH}" STREQUAL "fast")
            set(PATH_FLAGS --tryFastPath)
        else()
            set(PATH_FLAGS "")
        endif()

        # A=F8N, B=Half, both quantized to F8N at the MAC. This is the exact
        # shape that failed in sk_f8ngemm_quick.yaml's F8NH_F8NSS NN test.
        # Pre-fix, the fast path skipped the B->F8N quantization while the slow
        # path applied it, and the two diverged.
        add_cpu_gemm_test(${PATH}_f8fnuz_x_h_macF8N
            ARGS ${F16_SIZES} --typeA f8fnuz --typeB f16 --computeInputB f8fnuz --validate ${PATH_FLAGS}
        )
        # OCP F8 counterpart (same shape, non-fnuz storage).
        # NB: BFloat8_fnuz x Half is intentionally absent - the slow path
        # has no TypedGemm dispatch for that combination.
        add_cpu_gemm_test(${PATH}_f8_x_h_macF8
            ARGS ${F16_SIZES} --typeA f8 --typeB f16 --computeInputB f8 --validate ${PATH_FLAGS}
        )
        # Mirror cases - A is the wider operand and gets quantized to F8 at
        # the MAC. Exercises the symmetric A-side quantization path on both
        # the test driver init and the GEMM implementations.
        add_cpu_gemm_test(${PATH}_h_x_f8fnuz_macF8N
            ARGS ${F16_SIZES} --typeA f16 --typeB f8fnuz --computeInputA f8fnuz --validate ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_h_x_f8_macF8
            ARGS ${F16_SIZES} --typeA f16 --typeB f8 --computeInputA f8 --validate ${PATH_FLAGS}
        )
    endforeach()
endif()

#################################################
#     F64 (double precision) tests              #
#################################################
# F64 uses double storage and accumulation.
foreach(PATH fast slow)
    if("${PATH}" STREQUAL "fast")
        set(PATH_FLAGS --tryFastPath)
    else()
        set(PATH_FLAGS "")
    endif()

    foreach(TRANS NN TN NT TT)
        get_trans_flags(${TRANS} TFLAGS)
        add_cpu_gemm_test(${PATH}_f64_${TRANS}
            ARGS ${F64_SIZES} ${TFLAGS} --type f64 --validate ${PATH_FLAGS}
        )
    endforeach()
    add_cpu_gemm_test(${PATH}_f64_Beta
        ARGS ${F64_SIZES} --type f64 --beta 1.0 --validate ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_f64_Bias
        ARGS ${F64_SIZES} --type f64 --validate --bias ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_f64_AllFeatures
        ARGS ${F64_SIZES} --type f64 --validate --bias --activation relu --scaleAlphaVec ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_f64_TN_AllFeatures
        ARGS ${F64_SIZES} --transA --type f64 --validate --bias --activation relu --scaleAlphaVec ${PATH_FLAGS}
    )
    foreach(MODE Scalar Vector)
        add_cpu_gemm_test(${PATH}_f64_ScaleAB_${MODE}
            ARGS ${F64_SIZES} --type f64 --validate --useScaleAB ${MODE} ${PATH_FLAGS}
        )
    endforeach()
endforeach()

#################################################
#     TF32 (XFloat32 math-op) tests             #
#################################################
# TF32 uses float storage with XFloat32 truncation on multiply.
foreach(PATH fast slow)
    if("${PATH}" STREQUAL "fast")
        set(PATH_FLAGS --tryFastPath)
    else()
        set(PATH_FLAGS "")
    endif()

    foreach(TRANS NN TN NT TT)
        get_trans_flags(${TRANS} TFLAGS)
        add_cpu_gemm_test(${PATH}_tf32_${TRANS}
            ARGS ${F32_SIZES} ${TFLAGS} --type tf32 --validate ${PATH_FLAGS}
        )
    endforeach()
    add_cpu_gemm_test(${PATH}_tf32_Beta
        ARGS ${F32_SIZES} --type tf32 --beta 1.0 --validate ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_tf32_Bias
        ARGS ${F32_SIZES} --type tf32 --validate --bias ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_tf32_AllFeatures
        ARGS ${F32_SIZES} --type tf32 --validate --bias --activation relu --scaleAlphaVec ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_tf32_TN_AllFeatures
        ARGS ${F32_SIZES} --transA --type tf32 --validate --bias --activation relu --scaleAlphaVec ${PATH_FLAGS}
    )
    foreach(MODE Scalar Vector)
        add_cpu_gemm_test(${PATH}_tf32_ScaleAB_${MODE}
            ARGS ${F32_SIZES} --type tf32 --validate --useScaleAB ${MODE} ${PATH_FLAGS}
        )
    endforeach()
endforeach()

add_test(
    NAME "CPUGemm.tf32_MixedA_Rejected"
    COMMAND cpu-gemm-driver -M 16 -N 16 -K 16 --typeA tf32 --typeB f32 --validate
)
set_tests_properties("CPUGemm.tf32_MixedA_Rejected" PROPERTIES WILL_FAIL TRUE)

add_test(
    NAME "CPUGemm.tf32_MixedB_Rejected"
    COMMAND cpu-gemm-driver -M 16 -N 16 -K 16 --typeA f32 --typeB tf32 --validate
)
set_tests_properties("CPUGemm.tf32_MixedB_Rejected" PROPERTIES WILL_FAIL TRUE)

#################################################
#        FP4 / MX Scale tests (fast & slow)     #
#################################################
if(NOT WIN32)
    # M/N similar to f32 tests; K a multiple of mxBlock (32), as the slow path needs.
    set(F4_SIZES_32 -M 133 -N 147 -K 160)
    set(F4_SIZES_64 -M 133 -N 147 -K 192)

    foreach(PATH fast slow)
        if("${PATH}" STREQUAL "fast")
            set(PATH_FLAGS --tryFastPath)
        else()
            set(PATH_FLAGS "")
        endif()

        # Basic F4 with MX scaling, all transpose combos
        foreach(TRANS NN TN NT TT)
            get_trans_flags(${TRANS} TFLAGS)
            add_cpu_gemm_test(${PATH}_f4_MX32_${TRANS}
                ARGS ${F4_SIZES_32} ${TFLAGS} --type f4 --mxBlockA 32 --mxBlockB 32 --validate ${PATH_FLAGS}
            )
            add_cpu_gemm_test(${PATH}_f4_MX32_K192_${TRANS}
                ARGS ${F4_SIZES_64} ${TFLAGS} --type f4 --mxBlockA 32 --mxBlockB 32 --validate ${PATH_FLAGS}
            )
        endforeach()

        # F4 without MX (mxBlock=0, the default)
        add_cpu_gemm_test(${PATH}_f4_NoMX_NN
            ARGS -M 16 -N 16 -K 16 --type f4 --validate ${PATH_FLAGS}
        )

        # F4 + MX + epilogue features
        add_cpu_gemm_test(${PATH}_f4_MX32_Bias
            ARGS ${F4_SIZES_32} --type f4 --mxBlockA 32 --mxBlockB 32 --validate --bias ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_f4_MX32_Relu
            ARGS ${F4_SIZES_32} --type f4 --mxBlockA 32 --mxBlockB 32 --validate --activation relu ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_f4_MX32_ScaleAlphaVec
            ARGS ${F4_SIZES_32} --type f4 --mxBlockA 32 --mxBlockB 32 --validate --scaleAlphaVec ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_f4_MX32_AllFeatures
            ARGS ${F4_SIZES_32} --type f4 --mxBlockA 32 --mxBlockB 32 --validate --bias --activation relu --scaleAlphaVec ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_f4_MX32_Beta
            ARGS ${F4_SIZES_32} --type f4 --mxBlockA 32 --mxBlockB 32 --beta 1.0 --validate ${PATH_FLAGS}
        )

        # F4 + MX + ScaleAB
        foreach(MODE Scalar Vector)
            add_cpu_gemm_test(${PATH}_f4_MX32_ScaleAB_${MODE}
                ARGS ${F4_SIZES_32} --type f4 --mxBlockA 32 --mxBlockB 32 --validate --useScaleAB ${MODE} ${PATH_FLAGS}
            )
        endforeach()
    endforeach()

    # Fast-path-only: mxBlock=16 (tests non-hardcoded mxBlock; 16 % BLOCK_K=8 == 0)
    add_cpu_gemm_test(fast_f4_MX16_NN
        ARGS ${F4_SIZES_32} --type f4 --mxBlockA 16 --mxBlockB 16 --validate --tryFastPath
    )

    # Batched mxfp4 (batchCount > 1) — exercises strideBatchMxsa/strideBatchMxsb
    # in the fast path's MX loop (the site of a previous .a / .b typo bug).
    foreach(PATH fast slow)
        if("${PATH}" STREQUAL "fast")
            set(PATH_FLAGS --tryFastPath)
        else()
            set(PATH_FLAGS "")
        endif()
        foreach(TRANS NN TT)
            get_trans_flags(${TRANS} TFLAGS)
            add_cpu_gemm_test(${PATH}_f4_MX32_Batched_${TRANS}
                ARGS ${F4_SIZES_32} ${TFLAGS} --type f4 --mxBlockA 32 --mxBlockB 32 --validate --batchCount 2 ${PATH_FLAGS}
            )
        endforeach()
    endforeach()

    # Asymmetric MX (mxBlockA != mxBlockB). K must be a multiple of
    # lcm(mxBlockA, mxBlockB) — here lcm(32, 64) = 64, which F4_SIZES_64
    # already satisfies (K=192). Fast path only: the production slow path
    # collapses the smaller side's per-segment scales (see driver rejection
    # in runGemm).
    add_cpu_gemm_test(fast_f4_MX_A32_B64_NN
        ARGS ${F4_SIZES_64} --type f4 --mxBlockA 32 --mxBlockB 64 --validate --tryFastPath
    )
    add_cpu_gemm_test(fast_f4_MX_A64_B32_NN
        ARGS ${F4_SIZES_64} --type f4 --mxBlockA 64 --mxBlockB 32 --validate --tryFastPath
    )

    # Negative test: slow path must reject asymmetric MX.
    add_test(
        NAME "CPUGemm.slow_f4_MX_A32_B64_Rejected"
        COMMAND cpu-gemm-driver ${F4_SIZES_64} --type f4 --mxBlockA 32 --mxBlockB 64 --validate
    )
    set_tests_properties("CPUGemm.slow_f4_MX_A32_B64_Rejected" PROPERTIES WILL_FAIL TRUE)

    # Negative test: one-sided MX (only A or only B) must be rejected.
    add_test(
        NAME "CPUGemm.f4_MX_OneSided_Rejected"
        COMMAND cpu-gemm-driver ${F4_SIZES_32} --type f4 --mxBlockA 32 --validate
    )
    set_tests_properties("CPUGemm.f4_MX_OneSided_Rejected" PROPERTIES WILL_FAIL TRUE)

    # Edge-size mxfp4 coverage: K equal to mxBlock (single MX block per row),
    # M=1 / N=1 strips, and alpha=0 short-circuit. Both fast and slow paths.
    foreach(PATH fast slow)
        if("${PATH}" STREQUAL "fast")
            set(PATH_FLAGS --tryFastPath)
        else()
            set(PATH_FLAGS "")
        endif()
        add_cpu_gemm_test(${PATH}_f4_MX32_K_eq_mxBlock
            ARGS -M 16 -N 16 -K 32 --type f4 --mxBlockA 32 --mxBlockB 32 --validate ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_f4_MX32_M1
            ARGS -M 1 -N 147 -K 160 --type f4 --mxBlockA 32 --mxBlockB 32 --validate ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_f4_MX32_N1
            ARGS -M 133 -N 1 -K 160 --type f4 --mxBlockA 32 --mxBlockB 32 --validate ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_f4_MX32_Alpha0
            ARGS ${F4_SIZES_32} --type f4 --mxBlockA 32 --mxBlockB 32 --alpha 0 --validate ${PATH_FLAGS}
        )
    endforeach()

    # F4 with odd dimensions (M*K is odd) to exercise ShadowBuffer edge cases
    add_cpu_gemm_test(fast_f4_NoMX_OddDims
        ARGS -M 3 -N 5 -K 7 --type f4 --validate --tryFastPath
    )

    # Partial last MX block (K not a multiple of mxBlockA and/or mxBlockB), as in
    # mxfp4_mxfp4_bf16_tn_act (K = 333 / 255 / 63, block 32). Fast path only: the
    # slow path's MX inner loop sums whole blocks, reading past K (the driver
    # rejects K tails without --tryFastPath, see the BadK tests below).
    foreach(TRANS NN TN NT TT)
        get_trans_flags(${TRANS} TFLAGS)
        add_cpu_gemm_test(fast_f4_MX32_KTail333_${TRANS}
            ARGS -M 133 -N 147 -K 333 ${TFLAGS} --type f4 --mxBlockA 32 --mxBlockB 32 --validate --tryFastPath
        )
        # K smaller than the block and not a multiple of the fast path's K tile
        add_cpu_gemm_test(fast_f4_MX32_KTail7_${TRANS}
            ARGS -M 33 -N 31 -K 7 ${TFLAGS} --type f4 --mxBlockA 32 --mxBlockB 32 --validate --tryFastPath
        )
    endforeach()
    foreach(K 255 63)
        add_cpu_gemm_test(fast_f4_MX32_KTail${K}_TN
            ARGS -M 133 -N 147 -K ${K} --transA --type f4 --mxBlockA 32 --mxBlockB 32 --validate --tryFastPath
        )
    endforeach()
    add_cpu_gemm_test(fast_f4_MX32_KTail_AllFeatures_TN
        ARGS -M 133 -N 147 -K 333 --transA --type f4 --mxBlockA 32 --mxBlockB 32 --validate --beta 1.0 --bias --activation relu --scaleAlphaVec --tryFastPath
    )
    add_cpu_gemm_test(fast_f4_MX32_KTail_ScaleAB_Vector_TN
        ARGS -M 133 -N 147 -K 333 --transA --type f4 --mxBlockA 32 --mxBlockB 32 --validate --useScaleAB Vector --tryFastPath
    )
    foreach(TRANS NN TT)
        get_trans_flags(${TRANS} TFLAGS)
        add_cpu_gemm_test(fast_f4_MX32_KTail_Batched_${TRANS}
            ARGS -M 133 -N 147 -K 333 ${TFLAGS} --type f4 --mxBlockA 32 --mxBlockB 32 --validate --batchCount 2 --tryFastPath
        )
    endforeach()
    add_cpu_gemm_test(fast_f4_MX16_KTail_NN
        ARGS -M 133 -N 147 -K 333 --type f4 --mxBlockA 16 --mxBlockB 16 --validate --tryFastPath
    )
    # Asymmetric blocks: K = 160 is a multiple of 32 but not of 64, so only the
    # 64 side has a partial last block; K = 333 gives both sides one.
    add_cpu_gemm_test(fast_f4_MX_A32_B64_KTailB_NN
        ARGS ${F4_SIZES_32} --type f4 --mxBlockA 32 --mxBlockB 64 --validate --tryFastPath
    )
    add_cpu_gemm_test(fast_f4_MX_A64_B32_KTailA_NN
        ARGS ${F4_SIZES_32} --type f4 --mxBlockA 64 --mxBlockB 32 --validate --tryFastPath
    )
    add_cpu_gemm_test(fast_f4_MX_A32_B64_KTail333_TN
        ARGS -M 133 -N 147 -K 333 --transA --type f4 --mxBlockA 32 --mxBlockB 64 --validate --tryFastPath
    )

    # Negative tests: K/mxBlock combos the slow path rejects should fail with a clear error
    add_test(
        NAME "CPUGemm.slow_f4_MX32_BadK_Unaligned"
        COMMAND cpu-gemm-driver -M 16 -N 16 -K 33 --type f4 --mxBlockA 32 --mxBlockB 32 --validate
    )
    set_tests_properties("CPUGemm.slow_f4_MX32_BadK_Unaligned" PROPERTIES WILL_FAIL TRUE)

    add_test(
        NAME "CPUGemm.slow_f4_MX32_BadK_TooSmall"
        COMMAND cpu-gemm-driver -M 16 -N 16 -K 16 --type f4 --mxBlockA 32 --mxBlockB 32 --validate
    )
    set_tests_properties("CPUGemm.slow_f4_MX32_BadK_TooSmall" PROPERTIES WILL_FAIL TRUE)

    # Negative test: non-power-of-2 mxBlock should be rejected
    add_test(
        NAME "CPUGemm.slow_f4_MX6_BadPow2"
        COMMAND cpu-gemm-driver -M 4 -N 4 -K 12 --type f4 --mxBlockA 6 --mxBlockB 6 --validate
    )
    set_tests_properties("CPUGemm.slow_f4_MX6_BadPow2" PROPERTIES WILL_FAIL TRUE)

    # Negative test: mxBlock with non-FP4 types should be rejected
    add_test(
        NAME "CPUGemm.slow_f32_MX32_Rejected"
        COMMAND cpu-gemm-driver -M 16 -N 16 -K 32 --type f32 --mxBlockA 32 --mxBlockB 32 --validate
    )
    set_tests_properties("CPUGemm.slow_f32_MX32_Rejected" PROPERTIES WILL_FAIL TRUE)
endif()

#################################################
#     AMAX-D output tests                       #
#################################################
# --outputAmaxD makes the reference also write AMAX-D (max |D| over every
# batch); the driver checks it against max |D| of the golden GEMM, so the
# check does not depend on which CPU path ran. The fast path does not support
# AMAX-D (isFastPathEligible rejects "amaxD"), so these run without
# --tryFastPath, on the slow path.
foreach(TYPE f32 f16 bf16)
    get_test_sizes(${TYPE} SIZES)
    add_cpu_gemm_test(slow_${TYPE}_AmaxD
        ARGS ${SIZES} --type ${TYPE} --validate --outputAmaxD
    )
    add_cpu_gemm_test(slow_${TYPE}_AmaxD_AllFeatures
        ARGS ${SIZES} --transA --type ${TYPE} --beta 0.5 --validate --bias --activation relu --scaleAlphaVec --useScaleAB Vector --outputAmaxD
    )
endforeach()
add_cpu_gemm_test(slow_f64_AmaxD
    ARGS ${F64_SIZES} --type f64 --validate --outputAmaxD
)
add_cpu_gemm_test(slow_tf32_AmaxD
    ARGS ${F32_SIZES} --type tf32 --validate --outputAmaxD
)
# Max over every batch, not only the first.
add_cpu_gemm_test(slow_f32_AmaxD_Batched
    ARGS ${F32_SIZES} --type f32 --validate --batchCount 3 --outputAmaxD
)
# All-zero D: AMAX-D must be written as 0 (the driver starts it at -1).
add_cpu_gemm_test(slow_f32_AmaxD_Alpha0
    ARGS ${F32_SIZES} --type f32 --alpha 0 --validate --outputAmaxD
)
# Fewer output elements than threads. The single D element is negative with
# the driver's fixed seed, so this also checks that AMAX-D is |D|, not D.
add_cpu_gemm_test(slow_f32_AmaxD_1x1
    ARGS -M 1 -N 1 -K 165 --type f32 --validate --outputAmaxD
)
if(NOT WIN32)
    foreach(TYPE f8 bf8)
        add_cpu_gemm_test(slow_${TYPE}_AmaxD
            ARGS ${F16_SIZES} --type ${TYPE} --validate --outputAmaxD
        )
    endforeach()
    add_cpu_gemm_test(slow_f4_MX32_AmaxD
        ARGS -M 133 -N 147 -K 160 --type f4 --mxBlockA 32 --mxBlockB 32 --validate --outputAmaxD
    )
endif()

#################################################
#     Int8 input tests (fast & slow)            #
#################################################
# A/B are Int8 drawn from [-128, 127]. Each combination is one of the
# reference's TypedGemm_I8_* ids: C/D type (--typeCD) x alpha/beta type
# (--typeAlpha), which is also the reference's epilogue type. The golden GEMM
# accumulates exactly in integers, applies the epilogue in that type and casts
# to C/D with the reference's SaturateCast rules, so D must match exactly, on
# both paths: fast and slow therefore also match each other exactly.
#
# The slow path accumulates the float-alpha types (I8_*_S) in float per K
# element, which is exact only while |sum| < 2^24. Keep K small enough:
# F16_SIZES has K = 151, so |sum| <= 151 * 128^2 ~= 2.5M.
# (The fast path accumulates in int32_t, exact at any K; its results above 2^24
# are covered by the ReferenceFastPath gtests.)
#
# Each entry: NAME;typeCD;typeAlpha;alpha;beta;alpha for ScaleAB. Alphas keep D
# in range: Half output below 65504, no int32 overflow (the golden throws on
# int32 overflow), and a mix of saturated and in-range Int8 values with
# round-half-to-even ties (I8_I8_S); 2^-16 for ScaleAB, whose scales go up to
# 100 x 100.
#
# With int alpha (>= 1) and the full data range, almost every I8_I8_I32 output
# saturates, which would hide errors that only change in-range values. The
# I8_I8_I32_Narrow tests draw A/B from [-3, 3] instead (|sum| has a standard
# deviation of about 50, so about 1% of D saturates). No ScaleAB there: the
# scales (up to 100 x 100) saturate D at any data range.
foreach(PATH fast slow)
    if("${PATH}" STREQUAL "fast")
        set(PATH_FLAGS --tryFastPath)
    else()
        set(PATH_FLAGS "")
    endif()

    foreach(COMBO
            "I8_I8_I32;i8;i32;1;2;1"
            "I8_I32_I32;i32;i32;1;2;1"
            "I8_I32_S;i32;f32;0.5;0.5;0.0000152587890625"
            "I8_I8_S;i8;f32;0.001953125;0.5;0.0000152587890625"
            "I8_H_S;f16;f32;0.015625;0.5;0.0000152587890625"
            "I8_B_S;bf16;f32;1;0.5;0.0000152587890625")
        list(GET COMBO 0 NAME)
        list(GET COMBO 1 TYPE_CD)
        list(GET COMBO 2 TYPE_ALPHA)
        list(GET COMBO 3 ALPHA)
        list(GET COMBO 4 BETA)
        list(GET COMBO 5 SCALEAB_ALPHA)
        set(I8_ARGS ${F16_SIZES} --type i8 --typeCD ${TYPE_CD} --typeAlpha ${TYPE_ALPHA} --validate ${PATH_FLAGS})

        foreach(TRANS NN TN NT TT)
            get_trans_flags(${TRANS} TFLAGS)
            add_cpu_gemm_test(${PATH}_${NAME}_${TRANS}
                ARGS ${I8_ARGS} ${TFLAGS} --alpha ${ALPHA}
            )
        endforeach()
        add_cpu_gemm_test(${PATH}_${NAME}_Beta
            ARGS ${I8_ARGS} --alpha ${ALPHA} --beta ${BETA}
        )
        add_cpu_gemm_test(${PATH}_${NAME}_Bias
            ARGS ${I8_ARGS} --alpha ${ALPHA} --bias
        )
        add_cpu_gemm_test(${PATH}_${NAME}_TN_AllFeatures
            ARGS ${I8_ARGS} --transA --alpha ${ALPHA} --beta ${BETA} --bias --activation relu --scaleAlphaVec
        )
        foreach(MODE Scalar Vector)
            add_cpu_gemm_test(${PATH}_${NAME}_ScaleAB_${MODE}
                ARGS ${I8_ARGS} --alpha ${SCALEAB_ALPHA} --useScaleAB ${MODE}
            )
        endforeach()
        add_cpu_gemm_test(${PATH}_${NAME}_Batched
            ARGS ${I8_ARGS} --alpha ${ALPHA} --batchCount 3
        )
    endforeach()

    set(I8_NARROW_ARGS ${F16_SIZES} --type i8 --typeCD i8 --typeAlpha i32 --int8Min -3 --int8Max 3 --validate ${PATH_FLAGS})
    foreach(TRANS NN TN NT TT)
        get_trans_flags(${TRANS} TFLAGS)
        add_cpu_gemm_test(${PATH}_I8_I8_I32_Narrow_${TRANS}
            ARGS ${I8_NARROW_ARGS} ${TFLAGS}
        )
    endforeach()
    add_cpu_gemm_test(${PATH}_I8_I8_I32_Narrow_Beta
        ARGS ${I8_NARROW_ARGS} --beta 2
    )
    add_cpu_gemm_test(${PATH}_I8_I8_I32_Narrow_Bias
        ARGS ${I8_NARROW_ARGS} --bias
    )
    add_cpu_gemm_test(${PATH}_I8_I8_I32_Narrow_TN_AllFeatures
        ARGS ${I8_NARROW_ARGS} --transA --beta 2 --bias --activation relu --scaleAlphaVec
    )
    add_cpu_gemm_test(${PATH}_I8_I8_I32_Narrow_Batched
        ARGS ${I8_NARROW_ARGS} --batchCount 3
    )
endforeach()

#################################################
#     Float8 output tests (fast & slow)         #
#################################################
# Float8/BFloat8 (OCP and fnuz) C/D with float alpha/beta. Each entry is one of
# the reference's Float8-output type ids (named as in TypedId.hpp):
# NAME;typeA;typeB;typeCD;compute-input flags ("-" for none);test set.
# The golden casts its float result to C/D the way the reference's
# SaturateCast does (round to nearest even, saturate to the largest finite
# value) and the driver compares Float8 outputs exactly: every K-loop sum is
# exact in float and the epilogue rounds the same operations in the same order
# (+-1 C/bias/scaleAlphaVec, integer scales; see cpu_gemm_driver.cpp), so fast,
# slow and golden must give the same F8 value.
# K = 151 with +-1 data makes every sum odd, so the sums with 16 < |sum| < 32
# (e4m3) or 8 < |sum| < 16 (e5m2) are ties between two F8 values; ScaleAB
# (scales up to 100 x 100) saturates most of D.
#
# Test sets: "full" for the four ids with the same type in A, B and C/D;
# "basic" for the cross-output and hybrid ids and Float8 x Half with a Half MAC;
# "noscale" for Float8 x Half with a Float8 MAC (--computeInput f8 on the Half
# operand). The slow path handles ScaleAB differently when the storage is wider
# than the compute-input type (Scalar: scales the Half value before quantizing
# it, Vector: drops that side's scale), for any output type, so ScaleAB is not
# tested there.
if(NOT WIN32)
    foreach(PATH fast slow)
        if("${PATH}" STREQUAL "fast")
            set(PATH_FLAGS --tryFastPath)
        else()
            set(PATH_FLAGS "")
        endif()

        foreach(COMBO
                "F8_F8_S;f8;f8;f8;-;full"
                "B8_B8_S;bf8;bf8;bf8;-;full"
                "F8N_F8N_S;f8fnuz;f8fnuz;f8fnuz;-;full"
                "B8N_B8N_S;bf8fnuz;bf8fnuz;bf8fnuz;-;full"
                "F8_B8_S;f8;f8;bf8;-;basic"
                "B8_F8_S;bf8;bf8;f8;-;basic"
                "F8B8_F8_S;f8;bf8;f8;-;basic"
                "F8B8_B8_S;f8;bf8;bf8;-;basic"
                "B8F8_F8_S;bf8;f8;f8;-;basic"
                "B8F8_B8_S;bf8;f8;bf8;-;basic"
                "F8N_B8N_S;f8fnuz;f8fnuz;bf8fnuz;-;basic"
                "B8N_F8N_S;bf8fnuz;bf8fnuz;f8fnuz;-;basic"
                "F8B8N_F8N_S;f8fnuz;bf8fnuz;f8fnuz;-;basic"
                "F8B8N_B8N_S;f8fnuz;bf8fnuz;bf8fnuz;-;basic"
                "B8F8N_F8N_S;bf8fnuz;f8fnuz;f8fnuz;-;basic"
                "B8F8N_B8N_S;bf8fnuz;f8fnuz;bf8fnuz;-;basic"
                "F8H_H_FP8_S;f8;f16;f8;--computeInputA f16;basic"
                "HF8_H_FP8_S;f16;f8;f8;--computeInputB f16;basic"
                "F8NH_H_FP8_S;f8fnuz;f16;f8fnuz;--computeInputA f16;basic"
                "HF8N_H_FP8_S;f16;f8fnuz;f8fnuz;--computeInputB f16;basic"
                "F8H_FP8_FP8_S;f8;f16;f8;--computeInputB f8;noscale"
                "HF8_FP8_FP8_S;f16;f8;f8;--computeInputA f8;noscale"
                "F8NH_FP8_FP8_S;f8fnuz;f16;f8fnuz;--computeInputB f8fnuz;noscale"
                "HF8N_FP8_FP8_S;f16;f8fnuz;f8fnuz;--computeInputA f8fnuz;noscale")
            list(GET COMBO 0 NAME)
            list(GET COMBO 1 TYPE_A)
            list(GET COMBO 2 TYPE_B)
            list(GET COMBO 3 TYPE_CD)
            list(GET COMBO 4 COMPUTE_FLAGS)
            list(GET COMBO 5 TEST_SET)
            if("${COMPUTE_FLAGS}" STREQUAL "-")
                set(COMPUTE_FLAGS "")
            endif()
            separate_arguments(COMPUTE_FLAGS)
            set(F8OUT_ARGS ${F16_SIZES} --typeA ${TYPE_A} --typeB ${TYPE_B} --typeCD ${TYPE_CD} ${COMPUTE_FLAGS} --validate ${PATH_FLAGS})

            add_cpu_gemm_test(${PATH}_${NAME}_NN
                ARGS ${F8OUT_ARGS}
            )
            # alpha 0.3: values off the F8 grid before the cast.
            add_cpu_gemm_test(${PATH}_${NAME}_TN_AllFeatures
                ARGS ${F8OUT_ARGS} --transA --alpha 0.3 --beta 0.5 --bias --activation relu --scaleAlphaVec
            )
            if(NOT "${TEST_SET}" STREQUAL "noscale")
                add_cpu_gemm_test(${PATH}_${NAME}_ScaleAB_Scalar
                    ARGS ${F8OUT_ARGS} --useScaleAB Scalar
                )
            endif()
            if("${TEST_SET}" STREQUAL "full")
                foreach(TRANS TN NT TT)
                    get_trans_flags(${TRANS} TFLAGS)
                    add_cpu_gemm_test(${PATH}_${NAME}_${TRANS}
                        ARGS ${F8OUT_ARGS} ${TFLAGS}
                    )
                endforeach()
                add_cpu_gemm_test(${PATH}_${NAME}_Beta
                    ARGS ${F8OUT_ARGS} --beta 0.5
                )
                add_cpu_gemm_test(${PATH}_${NAME}_Bias
                    ARGS ${F8OUT_ARGS} --bias
                )
                add_cpu_gemm_test(${PATH}_${NAME}_Alpha
                    ARGS ${F8OUT_ARGS} --alpha 0.3
                )
                add_cpu_gemm_test(${PATH}_${NAME}_ScaleAB_Vector
                    ARGS ${F8OUT_ARGS} --useScaleAB Vector --scaleAlphaVec --factorDim 1
                )
                add_cpu_gemm_test(${PATH}_${NAME}_Batched
                    ARGS ${F8OUT_ARGS} --batchCount 3
                )
            endif()
        endforeach()

        # A perturbed (here saturated) Float8 D element must fail validation.
        add_test(
            NAME "CPUGemm.${PATH}_F8_F8_S_InjectedFailure_Rejected"
            COMMAND cpu-gemm-driver ${F16_SIZES} --type f8 --typeCD f8 --useScaleAB Scalar --validate --injectValidationFailure ${PATH_FLAGS}
        )
        set_tests_properties("CPUGemm.${PATH}_F8_F8_S_InjectedFailure_Rejected" PROPERTIES WILL_FAIL TRUE)
    endforeach()
endif()

#################################################
#     Complex tests (fast & slow)               #
#################################################
# ComplexFloat (c, the reference's C_C_C) and ComplexDouble (z, Z_Z_Z): A, B, C,
# D and alpha/beta all of that type. --conjA/--conjB add TensorOp::ComplexConjugate
# to A/B. A, B and C have +-1 real and imaginary parts, so a missing, misplaced or
# extra conjugation changes D. Complex problems take no bias, ScaleAB,
# scaleAlphaVec or activation (the fast path rejects them; the driver too).
foreach(PATH fast slow)
    if("${PATH}" STREQUAL "fast")
        set(PATH_FLAGS --tryFastPath)
    else()
        set(PATH_FLAGS "")
    endif()

    foreach(TYPE c z)
        if("${TYPE}" STREQUAL "c")
            set(SIZES ${F32_SIZES})
        else()
            set(SIZES ${F64_SIZES})
        endif()
        foreach(TRANS NN TN NT TT)
            get_trans_flags(${TRANS} TFLAGS)
            add_cpu_gemm_test(${PATH}_${TYPE}_${TRANS}
                ARGS ${SIZES} ${TFLAGS} --type ${TYPE} --validate ${PATH_FLAGS}
            )
        endforeach()
        # Each entry: NAME;transpose;conjugate flags.
        foreach(CASE
                "ConjA_NN;NN;--conjA"
                "ConjB_NN;NN;--conjB"
                "ConjA_TN;TN;--conjA"
                "ConjB_NT;NT;--conjB"
                "ConjAConjB_TT;TT;--conjA --conjB")
            list(GET CASE 0 NAME)
            list(GET CASE 1 TRANS)
            list(GET CASE 2 CONJ_FLAGS)
            separate_arguments(CONJ_FLAGS)
            get_trans_flags(${TRANS} TFLAGS)
            add_cpu_gemm_test(${PATH}_${TYPE}_${NAME}
                ARGS ${SIZES} ${TFLAGS} ${CONJ_FLAGS} --type ${TYPE} --validate ${PATH_FLAGS}
            )
        endforeach()
        add_cpu_gemm_test(${PATH}_${TYPE}_AlphaBeta
            ARGS ${SIZES} --type ${TYPE} --alpha 0.3 --beta 0.5 --validate ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_${TYPE}_Batched_ConjA_TN
            ARGS ${SIZES} --transA --conjA --type ${TYPE} --beta 1.0 --batchCount 3 --validate ${PATH_FLAGS}
        )
    endforeach()

    add_test(
        NAME "CPUGemm.${PATH}_c_InjectedFailure_Rejected"
        COMMAND cpu-gemm-driver ${F32_SIZES} --type c --conjA --validate --injectValidationFailure ${PATH_FLAGS}
    )
    set_tests_properties("CPUGemm.${PATH}_c_InjectedFailure_Rejected" PROPERTIES WILL_FAIL TRUE)
endforeach()

#################################################
#     E output tests (fast & slow)              #
#################################################
# --useE makes the reference also write E, the value before activation (after
# bias), converted to E's type (here the C/D type) as the reference's SetValue
# does. The driver checks E against the golden GEMM's pre-activation value with
# the same casts and tolerance as D. With relu, D and E differ wherever the
# value is negative, so E written after activation (or not at all) fails.
# Bias is not combined with --batchCount (the driver's bias is one M vector).
foreach(PATH fast slow)
    if("${PATH}" STREQUAL "fast")
        set(PATH_FLAGS --tryFastPath)
    else()
        set(PATH_FLAGS "")
    endif()

    foreach(TYPE f32 f16 bf16)
        get_test_sizes(${TYPE} SIZES)
        add_cpu_gemm_test(${PATH}_${TYPE}_UseE_BiasRelu
            ARGS ${SIZES} --type ${TYPE} --beta 0.5 --bias --activation relu --useE --validate ${PATH_FLAGS}
        )
    endforeach()
    add_cpu_gemm_test(${PATH}_f32_UseE
        ARGS ${F32_SIZES} --type f32 --useE --validate ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_f32_UseE_TN_AllFeatures
        ARGS ${F32_SIZES} --transA --type f32 --beta 0.5 --bias --activation relu --scaleAlphaVec --useScaleAB Vector --useE --validate ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_f32_UseE_Relu_Batched
        ARGS ${F32_SIZES} --type f32 --beta 1.0 --activation relu --batchCount 3 --useE --validate ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_f64_UseE_BiasRelu
        ARGS ${F64_SIZES} --type f64 --beta 0.5 --bias --activation relu --useE --validate ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_tf32_UseE_BiasRelu
        ARGS ${F32_SIZES} --type tf32 --beta 0.5 --bias --activation relu --useE --validate ${PATH_FLAGS}
    )
    # Int8 inputs: Int8 E from a float epilogue (round half to even, saturate) and
    # Int32 E from an int32 epilogue, compared exactly (alphas as in the Int8 tests).
    add_cpu_gemm_test(${PATH}_I8_I8_S_UseE_BiasRelu
        ARGS ${F16_SIZES} --type i8 --typeCD i8 --typeAlpha f32 --alpha 0.001953125 --beta 0.5 --bias --activation relu --useE --validate ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_I8_I32_I32_UseE_BiasRelu
        ARGS ${F16_SIZES} --type i8 --typeCD i32 --typeAlpha i32 --alpha 1 --beta 2 --bias --activation relu --useE --validate ${PATH_FLAGS}
    )
    if(NOT WIN32)
        # Float8 E, compared exactly (see the Float8 output tests).
        add_cpu_gemm_test(${PATH}_F8_F8_S_UseE_BiasRelu
            ARGS ${F16_SIZES} --type f8 --typeCD f8 --alpha 0.3 --beta 0.5 --bias --activation relu --useE --validate ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_f4_MX32_UseE_Relu
            ARGS -M 133 -N 147 -K 160 --type f4 --mxBlockA 32 --mxBlockB 32 --activation relu --useE --validate ${PATH_FLAGS}
        )
    endif()
endforeach()

#################################################
#     Gradient tests (fast & slow)              #
#################################################
# --gradient: no bias is added; with relu, D is multiplied by relu(E), E being an
# input (+-1, +-2) with --useE, else zero. With --bias the reference writes the bias
# gradient, reduced from D (after the activation gradient; length M), A (length M) or
# B (length N) (--biasSrc d/a/b). The bias starts as NaN (integer epilogues: the
# largest value), and the driver checks it against the golden sums, so a missing,
# misplaced or wrongly ordered reduction fails. The reference reduces one batch only,
# so --bias is not combined with --batchCount here (the driver rejects it).
foreach(PATH fast slow)
    if("${PATH}" STREQUAL "fast")
        set(PATH_FLAGS --tryFastPath)
    else()
        set(PATH_FLAGS "")
    endif()

    foreach(TYPE f32 f16 bf16)
        get_test_sizes(${TYPE} SIZES)
        add_cpu_gemm_test(${PATH}_${TYPE}_Gradient_ReluE_BiasD
            ARGS ${SIZES} --type ${TYPE} --beta 0.5 --gradient --activation relu --useE --bias --validate ${PATH_FLAGS}
        )
    endforeach()
    foreach(SRC a b)
        string(TOUPPER ${SRC} SRC_NAME)
        foreach(TRANS NN TN NT TT)
            get_trans_flags(${TRANS} TFLAGS)
            add_cpu_gemm_test(${PATH}_f16_Gradient_Bias${SRC_NAME}_${TRANS}
                ARGS ${F16_SIZES} ${TFLAGS} --type f16 --beta 0.5 --gradient --bias --biasSrc ${SRC} --validate ${PATH_FLAGS}
            )
        endforeach()
    endforeach()
    add_cpu_gemm_test(${PATH}_f32_Gradient_ReluE_BiasD_TT_AllFeatures
        ARGS ${F32_SIZES} --transA --transB --type f32 --beta 0.5 --gradient --activation relu --useE --bias --scaleAlphaVec --useScaleAB Vector --validate ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_f32_Gradient_Relu_NoE
        ARGS ${F32_SIZES} --type f32 --beta 0.5 --gradient --activation relu --validate ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_f32_Gradient_ReluE_Batched
        ARGS ${F32_SIZES} --type f32 --beta 1.0 --gradient --activation relu --useE --batchCount 3 --validate ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_f64_Gradient_ReluE_BiasD
        ARGS ${F64_SIZES} --type f64 --beta 0.5 --gradient --activation relu --useE --bias --validate ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_tf32_Gradient_ReluE_BiasA
        ARGS ${F32_SIZES} --transA --type tf32 --beta 0.5 --gradient --activation relu --useE --bias --biasSrc a --validate ${PATH_FLAGS}
    )
    # Int8 inputs: float epilogue with Int8 C/D/E, and int32 epilogue (int32 bias
    # gradient) with Int32 C/D/E; compared exactly.
    add_cpu_gemm_test(${PATH}_I8_I8_S_Gradient_ReluE_BiasD
        ARGS ${F16_SIZES} --type i8 --typeCD i8 --typeAlpha f32 --alpha 0.001953125 --beta 0.5 --gradient --activation relu --useE --bias --validate ${PATH_FLAGS}
    )
    add_cpu_gemm_test(${PATH}_I8_I32_I32_Gradient_ReluE_BiasA
        ARGS ${F16_SIZES} --type i8 --typeCD i32 --typeAlpha i32 --alpha 1 --beta 2 --gradient --activation relu --useE --bias --biasSrc a --validate ${PATH_FLAGS}
    )
    if(NOT WIN32)
        add_cpu_gemm_test(${PATH}_F8_F8_S_Gradient_ReluE_BiasB
            ARGS ${F16_SIZES} --type f8 --typeCD f8 --alpha 0.3 --beta 0.5 --gradient --activation relu --useE --bias --biasSrc b --validate ${PATH_FLAGS}
        )
        add_cpu_gemm_test(${PATH}_f4_MX32_Gradient_ReluE_BiasD
            ARGS -M 133 -N 147 -K 160 --type f4 --mxBlockA 32 --mxBlockB 32 --gradient --activation relu --useE --bias --validate ${PATH_FLAGS}
        )
    endif()
endforeach()

# Option checks: these must be rejected.
foreach(CASE
        "I8_MixedA;--typeA i8 --typeB f32"
        "I8_UnsupportedCD;--type i8 --typeCD f32"
        "I8_NonIntegerAlpha;--type i8 --typeCD i32 --typeAlpha i32 --alpha 0.5"
        "I8_Int8RangeOutOfBounds;--type i8 --int8Min -129"
        "I8_Int8RangeEmpty;--type i8 --int8Min 3 --int8Max 2"
        "F32_Int8Range;--type f32 --int8Max 3"
        "F32_TypeCD;--type f32 --typeCD f16"
        "F32_F8CD;--type f32 --typeCD f8"
        "F8_F8CDOtherFamily;--type f8 --typeCD f8fnuz"
        "F8_F8CDTypeAlpha;--type f8 --typeCD f8 --typeAlpha f64"
        "B8H_B8CD;--typeA bf8 --typeB f16 --computeInputA f16 --typeCD bf8"
        "F8H_F8CDNoTypeId;--typeA f8 --typeB f16 --typeCD f8"
        "C_MixedZ;--typeA c --typeB z"
        "C_MixedF32;--typeA c --typeB f32"
        "C_TypeCD;--type c --typeCD f32"
        "C_ComputeInput;--type c --computeInputA f32"
        "C_Bias;--type c --bias"
        "Z_ScaleAB;--type z --useScaleAB Scalar"
        "C_UseE;--type c --useE"
        "C_Gradient;--type c --gradient"
        "F32_BiasSrcWithoutGradient;--type f32 --bias --biasSrc a"
        "F32_UnknownBiasSrc;--type f32 --gradient --bias --biasSrc c"
        "F32_GradientBiasBatched;--type f32 --gradient --bias --batchCount 2"
        "F4_GradientBiasSrcA;--type f4 --gradient --bias --biasSrc a"
        "F32_ConjA;--type f32 --conjA")
    list(GET CASE 0 NAME)
    list(GET CASE 1 FLAGS)
    separate_arguments(FLAGS)
    add_test(
        NAME "CPUGemm.${NAME}_Rejected"
        COMMAND cpu-gemm-driver -M 16 -N 16 -K 16 ${FLAGS} --validate
    )
    set_tests_properties("CPUGemm.${NAME}_Rejected" PROPERTIES WILL_FAIL TRUE)
endforeach()
