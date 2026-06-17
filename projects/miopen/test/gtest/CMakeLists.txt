find_package(GTest REQUIRED)

include(GoogleTest)

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

# Append a pattern to the accumulated gtest negative (exclusion) filter.
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

# Build a gtest executable and register its (optionally sharded) ctest entries.
function(add_gtest TEST_NAME TEST_CPP)
  # cmake-lint: disable=R0912,R0915
  message("Adding Test: " ${TEST_NAME} " : " ${TEST_CPP})
  add_executable(${TEST_NAME} ${TEST_CPP})
  if(WIN32)
    target_compile_definitions(${TEST_NAME} PRIVATE NOMINMAX)
  endif()

  add_dependencies(miopen-tests ${TEST_NAME})
  add_dependencies(miopen-check ${TEST_NAME})

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

  if(MIOPEN_TEST_DISCRETE)
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
      gtest_discover_tests(${TEST_NAME} DISCOVERY_TIMEOUT 300 DISCOVERY_MODE PRE_TEST WORKING_DIRECTORY ${KERNELS_BINARY_DIR} PROPERTIES RUN_SERIAL TRUE ${TEST_ENVIRONMENT_VARIABLES} TEST_FILTER "${MIOPEN_GTEST_FILTER}")
    else()
      gtest_discover_tests(${TEST_NAME} DISCOVERY_TIMEOUT 300 DISCOVERY_MODE PRE_TEST WORKING_DIRECTORY ${KERNELS_BINARY_DIR} PROPERTIES ${TEST_ENVIRONMENT_VARIABLES} TEST_FILTER "${MIOPEN_GTEST_FILTER}")
    endif()
  else()
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

    # Excluded from MIOpen CI: the FP32 case OOMs when shards run concurrently on a single GPU.
    add_gtest_negative_filter("*UnitTestConvSolver*LargeStride*")

    # when building the single gtest, add them in a sharded manner
    # loop over GTEST_PARALLEL_LEVEL
    set(MIOPEN_GTEST_FILTER "${MIOPEN_GTEST_PREFIX}*${MIOPEN_GTEST_SUFFIX}*:${MIOPEN_GTEST_PREFIX}*NONE*${MIOPEN_GTEST_FILTER_NEGATIVE}")

    message(STATUS "MIOPEN_GTEST_FILTER ${MIOPEN_GTEST_FILTER}")

    # cmake-lint: disable=E1120
    foreach(i RANGE 1 ${GTEST_PARALLEL_LEVEL})
      math(EXPR SHARD_INDEX "${i} - 1")
      add_test(NAME ${TEST_NAME}_shard${i}
        COMMAND ${TEST_NAME}
          --gtest_filter=${MIOPEN_GTEST_FILTER}
          --gtest_output=xml:${PROJECT_BINARY_DIR}/test_results/${TEST_NAME}_shard${i}.xml
        WORKING_DIRECTORY "${KERNELS_BINARY_DIR}"
      )
      set_tests_properties(${TEST_NAME}_shard${i} PROPERTIES
        ENVIRONMENT "MIOPEN_USER_DB_PATH=${CMAKE_CURRENT_BINARY_DIR};MIOPEN_INVOKED_FROM_CTEST=1;GTEST_TOTAL_SHARDS=${GTEST_PARALLEL_LEVEL};GTEST_SHARD_INDEX=${SHARD_INDEX}"
      )
    endforeach()

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
target_include_directories(miopen_gtest_common PRIVATE ../ ../../src/kernels)
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

else()
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
    COMMENT "Checking gtest test names"
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

#Enable the test categories in test_categories.yaml
include(${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake)
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml")
        message(STATUS "YAML-based test categorization")
        apply_test_category_labels(miopen_gtest "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml" "${KERNELS_BINARY_DIR}" "${INSTALL_TEST_FILE}")
else()
        message(STATUS "No test categorization (YAML missing)")
endif()

# Install CTestTestfile.cmake to bin/MIOpen/ subdirectory for TheRock distribution
if( NOT ENABLE_ASAN_PACKAGING )
  install(
      FILES "${INSTALL_TEST_FILE}"
      DESTINATION "${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}"
      COMPONENT tests
      RENAME "CTestTestfile.cmake"
  )
endif()
