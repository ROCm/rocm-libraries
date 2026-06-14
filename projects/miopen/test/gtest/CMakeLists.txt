find_package(GTest REQUIRED)

include(GoogleTest)
find_package(Python 3 REQUIRED COMPONENTS Interpreter)

# Dapper currently requires all tests to be built. This may change prior to moving to TheRock CI,
# so leave the logic and hardwire them on temporarily.
set(MIOPEN_TEST_DISCRETE 1)
set(MIOPEN_TEST_SINGLE_GTEST 1)

message(STATUS "------------------------------------ CMAKE_CURRENT_LIST_DIR: ${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "------------------------------------ CMAKE_SOURCE_DIR:       ${CMAKE_SOURCE_DIR}")
message(STATUS "------------------------------------ CMAKE_BINARY_DIR:       ${CMAKE_BINARY_DIR}")

set(MIOPEN_DAPPER_SRC_DIR "${CMAKE_BINARY_DIR}/../script/dependency-parser")
set(MIOPEN_DAPPER_OUT_DIR "${CMAKE_BINARY_DIR}")
set(SHARDS_FILE ${MIOPEN_DAPPER_OUT_DIR}/miopen_gtest_shards.txt)
set(BUILD_NINJA "${MIOPEN_DAPPER_OUT_DIR}/build.ninja")
set(MAPPING_JSON "${MIOPEN_DAPPER_OUT_DIR}/miopen_dapper_mapping.json")
set(FIXTURES_JSON "${MIOPEN_DAPPER_OUT_DIR}/miopen_dapper_fixtures.json")
set(TESTS_JSON "${MIOPEN_DAPPER_OUT_DIR}/miopen_dapper_tests.json")
set(PY_MAIN "/${MIOPEN_DAPPER_SRC_DIR}/main.py")
set(PY_FIXTURES "/${MIOPEN_DAPPER_SRC_DIR}/src/extract_gtest_fixtures.py")
set(MIOPEN_GTEST_RUNNER "/${MIOPEN_DAPPER_SRC_DIR}/src/miopen_gtest_runner.py")
set(MIOPEN_DAPPER_DIFF ${MIOPEN_DAPPER_SRC_DIR}/src/dapper_diff.py)

# Generate fixtures
add_custom_command(
    OUTPUT ${FIXTURES_JSON}
    COMMENT "Generating ${FIXTURES_JSON}"
    COMMAND ${Python_EXECUTABLE} ${PY_FIXTURES}
    WORKING_DIRECTORY ${MIOPEN_DAPPER_OUT_DIR}
    VERBATIM
)

# Generate mapping
add_custom_command(
    OUTPUT ${MAPPING_JSON}
    COMMENT "Generating ${MAPPING_JSON}"
    COMMAND ${Python_EXECUTABLE} ${PY_MAIN} parse ${BUILD_NINJA}
    WORKING_DIRECTORY ${MIOPEN_DAPPER_OUT_DIR}
    DEPENDS tests
    VERBATIM
)

# Generate TESTS_JSON file
add_custom_command(
    OUTPUT "${TESTS_JSON}"
    COMMENT "Generating ${TESTS_JSON}"
    COMMAND ${Python_EXECUTABLE} "${PY_MAIN}" select ${MAPPING_JSON}  --fixturemap=${FIXTURES_JSON} --shardsfile=${SHARDS_FILE}
    WORKING_DIRECTORY ${MIOPEN_DAPPER_OUT_DIR}
    DEPENDS ${MAPPING_JSON} ${FIXTURES_JSON}
    VERBATIM
)
add_custom_target(dapper_tests DEPENDS "${TESTS_JSON}")
add_dependencies(check dapper_tests)

add_custom_target(
    diff_check
    COMMENT "Running filtered gtests..."
    COMMAND ${Python_EXECUTABLE} "${MIOPEN_GTEST_RUNNER}" "${MIOPEN_DAPPER_OUT_DIR}/bin/miopen_gtest" "${TESTS_JSON}"
    WORKING_DIRECTORY ${MIOPEN_DAPPER_OUT_DIR}
    DEPENDS miopen_gtest tests "${TESTS_JSON}"
    VERBATIM
)

add_test(NAME miopen_gtest_sharded_dapper
    COMMAND ${Python_EXECUTABLE} ${MIOPEN_DAPPER_DIFF} "${TESTS_JSON}" "${MIOPEN_DEBUG_CATEGORY}" "${MIOPEN_DAPPER_FILTER}"
    WORKING_DIRECTORY ${MIOPEN_DAPPER_OUT_DIR}
)

# TRJS dapper_diff may be redundant; 'miopen_gtest_sharded_dapper' is ran automatically by ctest
add_custom_target(dapper_diff
    COMMAND ${Python_EXECUTABLE} ${MIOPEN_DAPPER_DIFF} "${TESTS_JSON}" "${MIOPEN_DEBUG_CATEGORY}" "${MIOPEN_DAPPER_FILTER}"
    WORKING_DIRECTORY ${MIOPEN_DAPPER_OUT_DIR}
    DEPENDS check
    VERBATIM
)

if(MIOPEN_USE_HIPBLASLT)
  find_package(hipblaslt REQUIRED PATHS /opt/rocm $ENV{HIP_PATH})
endif()

if(MIOPEN_USE_ROCBLAS)
  find_package(rocblas REQUIRED PATHS /opt/rocm $ENV{HIP_PATH})
endif()

set(SOURCES
    gtest_common.cpp
    log.cpp
    platform.cpp
    conv_common.cpp
    unit_conv_solver.cpp
    )


# We use a special entry point for HIP backend testing so we can detect and flag any HIP errors that tests don't
# clean up themselves
if(MIOPEN_BACKEND_HIP)
  list(APPEND SOURCES main_hip.cpp)
endif()

if(MIOPEN_BACKEND_OPENCL)
  list(APPEND SKIP_TESTS dumpTensorTest.cpp)
endif()

# Add files to SKIP_TESTS when ASAN packaging is enabled.  These are temporary until a fix is found for
# the hang that occurs with CK kernels.
if(THEROCK_SANITIZER STREQUAL "ASAN")
    # Disable tests temporarily due to a bug that cause a hang when using CK kernels while Address Sanitizer is enabled.
    list(APPEND SKIP_TESTS bad_fusion_plan.cpp cba_find2_infer.cpp cba_infer.cpp ck_builder_shared.cpp ck_builder_xdl.cpp conv_activ_infer.cpp conv_ai_3d_kernel_tuning_utils.cpp conv_ck_igemm_fwd_v6r1_dlops_nchw.cpp conv_hip_igemm_xdlops.cpp conv_igemm_mlir_xdlops_bwd_wrw.cpp conv_igemm_mlir_xdlops_fwd.cpp find_2_conv.cpp find_db.cpp fused_conv_bias_res_add_activ.cpp graphapi_conv_bias_res_add_activ_fwd.cpp group_conv_deterministic_split_k.cpp group_conv2d_fwd.cpp group_conv2d_bwd.cpp group_conv2d_wrw.cpp group_conv3d_fwd.cpp group_conv3d_bwd.cpp group_conv3d_wrw.cpp find_mode_trust_verify.cpp kernel_tuning_net.cpp miopendriver_conv_immed.cpp miopendriver_conv2d_trans.cpp miopendriver_gemm.cpp miopendriver_regression_big_tensor.cpp miopendriver_regression_big_tensor.cpp miopendriver_regression_half_gfx9.cpp miopendriver_regression_half.cpp perf_config_HipImplicitGemm3DGroupFwdXdlops.cpp smoke_solver_ConvCkIgemmFwdV6r1DlopsNchw.cpp unit_conv_solver_ConvAsmImplicitGemmGTCDynamicBwdXdlops.cpp unit_conv_solver_ConvAsmImplicitGemmGTCDynamicFwdXdlops.cpp unit_conv_solver_ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC.cpp unit_conv_solver_ConvAsmImplicitGemmGTCDynamicWrwXdlops.cpp unit_conv_solver_ConvCkGroupedConvFwd.cpp unit_conv_solver_ConvHipImplicitGemm3DGroupBwdXdlops.cpp unit_conv_solver_ConvHipImplicitGemm3DGroupFwdXdlops.cpp unit_conv_solver_ConvHipImplicitGemm3DGroupWrwXdlops.cpp unit_conv_solver_ConvHipImplicitGemmBwdDataV1R1Xdlops.cpp unit_conv_solver_ConvHipImplicitGemmBwdDataV4R1Xdlops.cpp unit_conv_solver_ConvHipImplicitGemmFwdXdlops.cpp unit_conv_solver_ConvHipImplicitGemmGroupBwdXdlops.cpp unit_conv_solver_ConvHipImplicitGemmForwardV4R4Xdlops.cpp unit_conv_solver_ConvHipImplicitGemmForwardV4R5Xdlops.cpp unit_conv_solver_ConvHipImplicitGemmForwardV4R4Xdlops_Padded_Gemm.cpp unit_conv_solver_ConvHipImplicitGemmWrwV4R4Xdlops.cpp unit_conv_solver_ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm.cpp unit_conv_solver_ConvHipImplicitGemmBwdXdlops.cpp unit_conv_solver_ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC.cpp unit_conv_solver_ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC.cpp unit_conv_solver_ConvHipImplicitGemmGroupFwdXdlops.cpp unit_conv_solver_ConvHipImplicitGemmGroupWrwXdlops.cpp unit_implicitgemm_ck_util.cpp smoke_solver_ConvAsmImplicitGemmGTCDynamicXdlopsNHWC_fp32_fp16.cpp smoke_solver_ConvAsmImplicitGemmGTCDynamicXdlopsNHWC_bf16.cpp)
    # temporarily disable due to test failures
    list(APPEND SKIP_TESTS unit_conv_solver_ConvWinoRageRxS.cpp)
endif()

function(add_gtest_negative_filter NEGATIVE_FILTER_TO_ADD)
    set (TMP_FILTER ${MIOPEN_GTEST_FILTER_NEGATIVE})

    if (NOT TMP_FILTER )
        set (TMP_FILTER "-")
    else()
        set (TMP_FILTER "${TMP_FILTER}:")
    endif()
    set (TMP_FILTER ${TMP_FILTER}${NEGATIVE_FILTER_TO_ADD})

    set (MIOPEN_GTEST_FILTER_NEGATIVE ${TMP_FILTER} PARENT_SCOPE)
endfunction()

function(add_gtest TEST_NAME TEST_CPP)
  list(JOIN TEST_CPP "  " TESTS_LIST)
  message("Adding Test: " ${TEST_NAME} " : " ${TESTS_LIST})
  add_executable(${TEST_NAME} ${TEST_CPP})
  if(WIN32)
    target_compile_definitions(${TEST_NAME} PRIVATE NOMINMAX)
    set(DAPPER_PP_FLAG "/P")
  else()
    set(DAPPER_PP_FLAG "-E")
  endif()

  add_dependencies(tests ${TEST_NAME})
  add_dependencies(check ${TEST_NAME})

  target_compile_options(${TEST_NAME} PRIVATE -Wno-global-constructors -Wno-undef)
  target_include_directories(${TEST_NAME} PRIVATE ../ ../../src/kernels)

  target_link_libraries(${TEST_NAME} miopen_gtest_common)
  if(MIOPEN_ENABLE_AI_KERNEL_TUNING)
    target_link_libraries(${TEST_NAME} frugally-deep::fdeep Eigen3::Eigen)
  endif()

  # Need to add hipblaslt since tests directly use gemm_v2.cpp
  if(hipblaslt_FOUND)
    target_link_libraries( ${TEST_NAME} roc::hipblaslt )
  endif()
  # Need to add rocBLAS since static builds of MIOpen require rocBLAS linkage
  if(rocblas_FOUND)
    target_link_libraries( ${TEST_NAME} $<BUILD_INTERFACE:roc::rocblas> )
  endif()

  # Workaround : change in rocm-cmake was causing linking error so had to add ${CMAKE_DL_LIBS}
  #               We can remove ${CMAKE_DL_LIBS} once root cause is identified.
  # MIOpen_with_plugins ensures CK plugin .so's are built alongside the test
  target_link_libraries(${TEST_NAME} ${CMAKE_DL_LIBS} GTest::gtest GTest::gtest_main GTest::gmock MIOpen_with_plugins hip::host )
  if(NOT MIOPEN_EMBED_DB STREQUAL "")
      target_link_libraries(${TEST_NAME} $<BUILD_INTERFACE:miopen_data>)
  endif()
# Individual test binaries are only added if single gtest is not being created.
  if(NOT MIOPEN_TEST_SINGLE_GTEST)
    string(CONCAT TEST_ENVIRONMENT_VARIABLES
    "ENVIRONMENT;MIOPEN_USER_DB_PATH=${CMAKE_CURRENT_BINARY_DIR};"
    "ENVIRONMENT;MIOPEN_INVOKED_FROM_CTEST=1;")

    set(MIOPEN_GTEST_PREFIX)
    set(MIOPEN_GTEST_FILTER_NEGATIVE)

    if (MIOPEN_TEST_ALL)
      set(MIOPEN_GTEST_PREFIX "")
    else()
      set(MIOPEN_GTEST_PREFIX "Smoke")
    endif()

    if (NOT MIOPEN_TEST_DBSYNC)
        add_gtest_negative_filter("*DBSync*")
    endif()

    if (NOT MIOPEN_TEST_DEEPBENCH)
        add_gtest_negative_filter("*DeepBench*")
    endif()

    if (NOT MIOPEN_TEST_CONV)
        add_gtest_negative_filter("*MIOpenTestConv*")
    endif()

    if(MIOPEN_NO_GPU)
        add_gtest_negative_filter("*GPU*")
    endif()

    # NONE datatype is explicitly enabled for all the state, all the NONE tests take 4 seconds to complete.
    set(MIOPEN_GTEST_FILTER "${MIOPEN_GTEST_PREFIX}*${MIOPEN_GTEST_SUFFIX}*:${MIOPEN_GTEST_PREFIX}*NONE*${MIOPEN_GTEST_FILTER_NEGATIVE}")

    #it prints the message only once
    set(TMPFNAME _dummy_static_variable_${name})
    if (NOT COMMAND ${TMPFNAME})
      message(STATUS "MIOPEN_GTEST_FILTER ${MIOPEN_GTEST_FILTER}")
      # Define a function so next time it will exist
      function(${TMPFNAME})
      endfunction()
    endif()

    # Enable CMake to discover the test binary
    # Note: Due to the following cmake issue with gtest_discover_tests https://gitlab.kitware.com/cmake/cmake/-/issues/17812 you cannot pass PROPERTIES as a list.
    #       To work around this limitation, we are passing the environment variables in the format ENVIRONMENT;value1=${value1};ENVIRONMENT;value2=${value2}.
    # bn tests are run sequentially since running tests in parallel was causing large tensor case fail with insufficient memory error.
    if("${TEST_NAME}" STREQUAL "test_bn_bwd_serial_run" OR "${TEST_NAME}" STREQUAL "test_bn_fwd_train_serial_run" OR "${TEST_NAME}" STREQUAL "test_bn_infer_serial_run")
      gtest_discover_tests(${TEST_NAME} DISCOVERY_TIMEOUT 300 DISCOVERY_MODE PRE_TEST WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/${DATABASE_INSTALL_DIR} PROPERTIES RUN_SERIAL TRUE ${TEST_ENVIRONMENT_VARIABLES} TEST_FILTER "${MIOPEN_GTEST_FILTER}")
    else()
      gtest_discover_tests(${TEST_NAME} DISCOVERY_TIMEOUT 300 DISCOVERY_MODE PRE_TEST WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/${DATABASE_INSTALL_DIR} PROPERTIES ${TEST_ENVIRONMENT_VARIABLES} TEST_FILTER "${MIOPEN_GTEST_FILTER}")
    endif()
  endif()

  if(MIOPEN_TEST_SINGLE_GTEST AND TEST_NAME STREQUAL "miopen_gtest")
    set(MIOPEN_GTEST_PREFIX)
    set(MIOPEN_GTEST_FILTER_NEGATIVE)

    if (MIOPEN_TEST_ALL)
      set(MIOPEN_GTEST_PREFIX "")
    else()
      set(MIOPEN_GTEST_PREFIX "Smoke")
    endif()

    if (NOT MIOPEN_TEST_DBSYNC)
        add_gtest_negative_filter("*DBSync*")
    endif()

    if (NOT MIOPEN_TEST_DEEPBENCH)
        add_gtest_negative_filter("*DeepBench*")
    endif()

    if (NOT MIOPEN_TEST_CONV)
        add_gtest_negative_filter("*MIOpenTestConv*")
    endif()

    if(MIOPEN_NO_GPU)
        add_gtest_negative_filter("*GPU*")
    endif()

    if(MIOPEN_TEST_GFX115X)
        add_gtest_negative_filter("Full/GPU_BNFWDTrainLargeFusedActivation2D_FP32.BnV2LargeFWD_TrainCKfp32Activation/NCHW_BNSpatial_testBNAPIV1_Dim_2_test_id_32") # Temporarily disabled until gfx1151 CI nodes have fw 31 or higher installed
        add_gtest_negative_filter("Full/GPU_BNFWDTrainLarge2D_FP32.BnV2LargeFWD_TrainCKfp32/NCHW_BNSpatial_testBNAPIV2_Dim_2_test_id_64") # Temporarily disabled until gfx1151 CI nodes have fw 31 or higher installed
        add_gtest_negative_filter("*SerialRun3D*") # These FP32 SerialRun3D tests use so much memory that they have a risk of timing out the machine during tests
    endif()

    if(MIOPEN_TEST_GFX94X)
        add_gtest_negative_filter("Full/GPU_LstmMSRnn_FP32.TestFloat/1095") # Disable flaky test ALMIOPEN-1926
    endif()

    # when building the single gtest, add them in a sharded manner
    # loop over GTEST_PARALLEL_LEVEL
    set(MIOPEN_GTEST_FILTER "${MIOPEN_GTEST_PREFIX}*${MIOPEN_GTEST_SUFFIX}*:${MIOPEN_GTEST_PREFIX}*NONE*${MIOPEN_GTEST_FILTER_NEGATIVE}")

# TODO Fix Dapper default minimal run if no work to do. It's in calc_union_filter but in mici this is ran by dapper_diff which does not run until after check

    set(MIOPEN_CATEGORY_QUICK "Smoke/GPU_BNCKFWDTrainLarge2D_FP16*:Smoke/GPU_BNOCLFWDTrainLarge2D_FP16*:Smoke/GPU_BNOCLFWDTrainLarge3D_FP16*:Smoke/GPU_BNCKFWDTrainLarge2D_BFP16*:Smoke/GPU_BNOCLFWDTrainLarge2D_BFP16*:Smoke/GPU_BNOCLFWDTrainLarge3D_BFP16*:Smoke/GPU_UnitTestConvSolverImplicitGemmFwdXdlops_FP16*:Smoke/GPU_UnitTestConvSolverImplicitGemmFwdXdlops_BFP16*-Smoke/GPU_BNOCLFWDTrainLarge2D_BFP16*:Smoke/GPU_BNOCLInferLarge2D_BFP16*:*/GPU_MIOpenDriver*:*GPU_TestMhaFind20_FP32*:*GPU_TestMhaFind20_FP16*:Smoke/GPU_BNOCLInferLarge2D_BFP16*:Smoke/GPU_BNOCLFWDTrainLarge2D_BFP16*")
    set(MIOPEN_CATEGORY_STANDARD "*Fusion*:*/GPU_BNBWD*_*:*/GPU_BNOCLBWD*_*:*/GPU_BNFWD*_*:*/GPU_BNOCLFWD*_*:*/GPU_BNInfer*_*:*/GPU_BNActivInfer_*:*/GPU_BNOCLInfer*_*:*/GPU_bn_infer*_*:CPU_*:*/CPU_*:*/GPU_Cat_*:*/GPU_ConvBiasActiv*:*/GPU_Conv*:*/GPU_conv*:*/GPU_UnitTestConv*:*/GPU_GetitemBwd*:*/GPU_GLU_*:*/GPU_GroupConv*:*/GPU_GroupNorm_*:*/GPU_GRUExtra_*:*/GPU_TestActivation*:*/GPU_HipBLASLtGEMMTest*:*/GPU_KernelTuningNetTestConv*:*/GPU_Kthvalue_*:*/GPU_LayerNormTest*:*/GPU_LayoutTransposeTest_*:*/GPU_Lrn*:*/GPU_lstm_extra*:*/GPU_MultiMarginLoss_*:*/GPU_ConvNonpack*:*/GPU_PerfConfig_HipImplicitGemm*:*/GPU_AsymPooling2d_*:*/GPU_WidePooling2d_*:*/GPU_PReLU_*:*/GPU_Reduce*:*/GPU_reduce_custom_*:*/GPU_regression_issue_*:*/GPU_RNNExtra_*:*/GPU_RoPE*:*/GPU_SoftMarginLoss*:*/GPU_T5LayerNormTest_*:*/GPU_Op4dTensorGenericTest_*:*/GPU_TernaryTensorOps_*:*/GPU_unaryTensorOps_*:*/GPU_Transformers*:*/GPU_TunaNetTest_*:*/")
    if(DEFINED MIOPEN_DEBUG_CATEGORY)
      string(TOUPPER "${MIOPEN_DEBUG_CATEGORY}" MIOPEN_DEBUG_CATEGORY_UPPER)
      message(STATUS "User-specified category='${MIOPEN_DEBUG_CATEGORY_UPPER}'")

      if(MIOPEN_DEBUG_CATEGORY_UPPER STREQUAL "QUICK")
          set(MIOPEN_DAPPER_FILTER "${MIOPEN_DEBUG_DAPPER_FILTER}:${MIOPEN_CATEGORY_QUICK}")
      elseif(MIOPEN_DEBUG_CATEGORY_UPPER STREQUAL "STANDARD")
          set(MIOPEN_DAPPER_FILTER "${MIOPEN_DEBUG_DAPPER_FILTER}:${MIOPEN_CATEGORY_STANDARD}")
      else()
          message(FATAL_ERROR "Aborting. Unknown MIOPEN_DEBUG_CATEGORY: '{MIOPEN_DEBUG_CATEGORY_UPPER}'")
      endif()
      message(STATUS "Applying Category '${MIOPEN_DEBUG_CATEGORY_UPPER}' with user filters '${MIOPEN_DEBUG_DAPPER_FILTER}'")
      if(MIOPEN_TEST_ENABLE_DAPPER)
          set(MIOPEN_GTEST_FILTER "${MIOPEN_DAPPER_FILTER}")
      endif()
    endif()

    message(STATUS "MIOPEN_GTEST_FILTER ${MIOPEN_GTEST_FILTER}")

    # when building the single gtest, add them in a sharded manner
    # loop over GTEST_PARALLEL_LEVEL
    if(NOT DEFINED GTEST_PARALLEL_LEVEL)
        set(GTEST_PARALLEL_LEVEL 4)
    endif()

    file(REMOVE "${SHARDS_FILE}")
    file(TOUCH "${SHARDS_FILE}")
    set(MIOPEN_GTEST_SHARDS "")

    foreach(i RANGE 1 ${GTEST_PARALLEL_LEVEL})
      set(MIOPEN_GTEST_SHARD ${TEST_NAME}_shard${i})
      set(MIOPEN_GTEST_SHARD_JSON ${TEST_NAME}_shard${i}.json)
      add_test(NAME ${MIOPEN_GTEST_SHARD}
        COMMAND ${TEST_NAME} --gtest_filter=${MIOPEN_GTEST_FILTER} --gtest_output=json:${MIOPEN_DAPPER_OUT_DIR}/${MIOPEN_GTEST_SHARD_JSON}
        WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/${DATABASE_INSTALL_DIR}"
      )
      file(APPEND ${SHARDS_FILE} "${MIOPEN_GTEST_SHARD_JSON}\n")
      list(APPEND MIOPEN_GTEST_SHARDS ${MIOPEN_GTEST_SHARD})
      math(EXPR SHARD_INDEX "${i} - 1")
      set_tests_properties(${MIOPEN_GTEST_SHARD} PROPERTIES
        ENVIRONMENT "MIOPEN_USER_DB_PATH=${CMAKE_CURRENT_BINARY_DIR};MIOPEN_INVOKED_FROM_CTEST=1;GTEST_TOTAL_SHARDS=${GTEST_PARALLEL_LEVEL};GTEST_SHARD_INDEX=${SHARD_INDEX}"
      )
    endforeach()

    set_tests_properties(miopen_gtest_sharded_dapper PROPERTIES DEPENDS "${MIOPEN_GTEST_SHARDS}")
  endif()
  target_link_libraries(${TEST_NAME} BZip2::BZip2)
  if(WIN32)
    # Refer to https://en.cppreference.com/w/cpp/language/types for details.
    target_compile_options(${TEST_NAME} PRIVATE $<BUILD_INTERFACE:$<$<CXX_COMPILER_ID:Clang>:-U__LP64__>>)
  endif()
  if(HAS_LIB_STD_FILESYSTEM)
    target_link_libraries(${TEST_NAME} stdc++fs)
  endif()
endfunction()

file(GLOB TESTS *.cpp)
# Remove files that do not contain tests
foreach(SOURCE ${SOURCES})
    list(REMOVE_ITEM TESTS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE})
endforeach()

foreach(SOURCE ${SKIP_TESTS})
    list(REMOVE_ITEM TESTS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE})
endforeach()

# Common static library speeds up test assembly.
# Otherwise, all files in ${SOURCES} are rebuilt for each test.
add_library(miopen_gtest_common STATIC ${SOURCES})
target_include_directories(miopen_gtest_common PRIVATE ../ ../.. ../../src/kernels)
target_link_libraries(miopen_gtest_common PRIVATE GTest::gtest GTest::gmock MIOpen)
if(WIN32)
  # Refer to https://en.cppreference.com/w/cpp/language/types for details.
  target_compile_options(miopen_gtest_common PRIVATE $<BUILD_INTERFACE:$<$<CXX_COMPILER_ID:Clang>:-U__LP64__>>)
endif()

if( MIOPEN_TEST_DISCRETE )
  foreach(TEST ${TESTS})

    get_filename_component(BASE_NAME ${TEST} NAME_WE)
    add_gtest(test_${BASE_NAME} ${BASE_NAME}.cpp)
  endforeach()
endif()
if(MIOPEN_TEST_SINGLE_GTEST)
  foreach(TEST ${TESTS})
    get_filename_component(BASE_NAME ${TEST} NAME)
    list(APPEND TESTS_CPP ${BASE_NAME})
  endforeach()

  add_gtest(miopen_gtest "${TESTS_CPP}")

  if(MIOPEN_USE_COMPOSABLEKERNEL)
    if(MIOPEN_BUILD_CK )
        # The include directories also dont propagate correctly when using the non aliased targets.
        # Manually including them fixes this problem for inline builds.
        target_include_directories(miopen_gtest SYSTEM PRIVATE
            ../
            ${MIOPEN_CK_INCLUDE_DIR}
            ${MIOPEN_CK_BUILD_INCLUDE_DIR}
            ${MIOPEN_CK_LIBRARY_INCLUDE_DIR})
    else()
        # While building on TheRock we are missing CK include headers for miopen_gtest target
        # But we need them, for example, for unit_implicitgemm_ck_util.cpp gtest, which is part of miopen_gtest target
        find_package(composable_kernel 1.0.0 COMPONENTS utility)
        target_link_libraries(miopen_gtest composable_kernel::utility)
    endif()
  endif()

  add_custom_command(
    OUTPUT test_list
    COMMAND miopen_gtest --gtest_list_tests > test_list
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/check_names.py --list test_list
    DEPENDS miopen_gtest ${CMAKE_CURRENT_SOURCE_DIR}/check_names.py
    COMMENT "Checking test names"
    VERBATIM
  )

  add_custom_target(
    miopen_gtest_check
    DEPENDS test_list
  )

  if( NOT ENABLE_ASAN_PACKAGING )
    install(TARGETS miopen_gtest
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT clients)
  endif()
endif()

message(STATUS "gtest env: MIOPEN_USER_DB_PATH=${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "gtest env: MIOPEN_TEST_COMPOSABLEKERNEL=${MIOPEN_TEST_COMPOSABLEKERNEL}")

# Create install-time test file for distribution
set(INSTALL_TEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake")
file(WRITE "${INSTALL_TEST_FILE}"
[=[
# This is a test file generated by MIOpen for install time.
# Tests are defined with relative paths to work in the installed location.
]=]
)

# Enable the test categories in test_categories.yaml with 'standard' default
# TODO: probably not needed (just create them all)
if (NOT MIOPEN_TEST_CATEGORY)
  set(MIOPEN_TEST_CATEGORY standard)
endif()
set(MIOPEN_TEST_CATEGORIES_CMAKE ${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake)
include(${MIOPEN_TEST_CATEGORIES_CMAKE})
message(STATUS "YAML-based test categorization:")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml")
        message(STATUS "YAML-based test categorization")
        message(STATUS "apply_test_category_labels(miopen_gtest ${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml ${PROJECT_BINARY_DIR}/${DATABASE_INSTALL_DIR} ${INSTALL_TEST_FILE})")
        apply_test_category_labels(miopen_gtest "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml" "${PROJECT_BINARY_DIR}/${DATABASE_INSTALL_DIR}" "${INSTALL_TEST_FILE}")
else()
        message(STATUS "No test categorization (${MIOPEN_TEST_CATEGORIES_CMAKE} missing)")
endif()

list(POP_BACK CMAKE_MESSAGE_INDENT)

# Install CTestTestfile.cmake to bin/MIOpen/ subdirectory for TheRock distribution
if( NOT ENABLE_ASAN_PACKAGING )
  install(
      FILES "${INSTALL_TEST_FILE}"
      DESTINATION "${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}"
      COMPONENT tests
      RENAME "CTestTestfile.cmake"
  )
endif()
