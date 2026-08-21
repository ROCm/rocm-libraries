# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(GTest REQUIRED)
include(GoogleTest)

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
        "${CMAKE_CURRENT_SOURCE_DIR}/CachingLibrary_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ContractionSelectionLibrary_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CuCount_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CustomKernel_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DataTypes_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DataInit_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ExactLogicLibrary_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/HardwareMonitor_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/GEMMKernelTest.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/GEMMKernelTest_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/RunGEMMKernelTest.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/RunGEMMKernelTest_impl.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/TestData.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/TestUtils.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MXScalePadding_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/PciChipId_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Predicates_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ReferenceFastPath_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RangeLibrary_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ResultComparison_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ReferenceMXFastPath_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionValidation_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TensorDescriptor_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TestData.cpp"
)

# MasterSolutionLibraryMapping_test exercises TensileLite::LoadLibraryMapping,
# which is only defined when MSGPACK is the active serialization format
# (see tensilelite/src/Tensile.cpp). The test also writes msgpack mapping
# files directly via <msgpack.hpp>, so it has no meaning in YAML builds.
if(NOT HIPBLASLT_ENABLE_YAML)
    target_sources(tensilelite-tests
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/MasterSolutionLibraryMapping_test.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/CompressedLibraryLoad_test.cpp"
    )
endif()

target_include_directories(tensilelite-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
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

gtest_discover_tests(tensilelite-tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} TIMEOUT 60)

target_link_libraries(tensilelite-tests PUBLIC GTest::gtest)

if(HIPBLASLT_ENABLE_OPENMP)
    target_link_libraries(tensilelite-tests PRIVATE OpenMP::OpenMP_CXX)
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
    # M/N similar to f32 tests; K must be a multiple of mxBlock (32).
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

    # Negative tests: invalid K/mxBlock combos should fail with clear error
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
