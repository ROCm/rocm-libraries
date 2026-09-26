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

# Add files to SKIP_TESTS when ASAN packaging is enabled.  These are temporary until a fix is found for
# the hang that occurs with CK kernels.
if(THEROCK_SANITIZER STREQUAL "ASAN")
    # Disable tests that currently hang or crash under ASAN with CK kernels.
    list(APPEND SKIP_TESTS
        group_conv3d_bwd.cpp
        group_conv3d_fwd.cpp
        group_conv3d_wrw.cpp
        miopendriver_conv2d_trans.cpp
        miopendriver_conv_immed.cpp
        miopendriver_regression_half_gfx9.cpp
        unit_conv_solver_ConvCkGroupedConvFwd.cpp
        unit_conv_solver_ConvDepthwiseBwdData2D_test.cpp
        unit_conv_solver_ConvHipImplicitGemm3DGroupBwdXdlops.cpp
        unit_conv_solver_ConvHipImplicitGemm3DGroupFwdXdlops.cpp
        unit_conv_solver_ConvHipImplicitGemm3DGroupWrwXdlops.cpp
        unit_conv_solver_ConvHipImplicitGemmGroupBwdXdlops.cpp
        unit_conv_solver_ConvHipImplicitGemmGroupWrwXdlops.cpp
    )
    # Disable tests that are inert on gfx942 configuration (unsupported arch or
    # test_drive<> framework disabled).  These did not actually launch CK kernels
    # during gfx942 testing but are kept skipped pending more extensive testing.
    list(APPEND SKIP_TESTS
        conv_hip_igemm_xdlops.cpp
        conv_igemm_mlir_xdlops_bwd_wrw.cpp
        conv_igemm_mlir_xdlops_fwd.cpp
        smoke_solver_ConvAsmImplicitGemmGTCDynamicXdlopsNHWC_bf16.cpp
        smoke_solver_ConvAsmImplicitGemmGTCDynamicXdlopsNHWC_fp32_fp16.cpp
        unit_conv_solver_ConvHipImplicitGemmBwdDataV4R1Xdlops.cpp
        unit_conv_solver_ConvHipImplicitGemmForwardV4R4Xdlops.cpp
        unit_conv_solver_ConvHipImplicitGemmForwardV4R4Xdlops_Padded_Gemm.cpp
        unit_conv_solver_ConvHipImplicitGemmForwardV4R5Xdlops.cpp
        unit_conv_solver_ConvHipImplicitGemmWrwV4R4Xdlops.cpp
        unit_conv_solver_ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm.cpp
    )
    # temporarily disable due to test failures
    list(APPEND SKIP_TESTS unit_conv_solver_ConvWinoRageRxS.cpp)
    # temporarily disable: LeakSanitizer reports a memory leak at exit
    # (assertions pass but the process leaks).
    list(APPEND SKIP_TESTS find_2_conv.cpp)
    # temporarily disable: these fail under ASan even when run individually
    # on gfx942.
    list(APPEND SKIP_TESTS
        miopendriver_gemm.cpp
        miopendriver_regression_half.cpp
        unit_conv_solver_ConvDepthwiseFwd3D.cpp
    )
endif()

# Tests added for the hipDNN wrapper work build only with the flag on, so a flag-off build
# registers exactly the tests it did before that work.
if(NOT MIOPEN_ENABLE_HIPDNN_WRAPPER)
    list(APPEND SKIP_TESTS
        hipdnn_shim_conv.cpp
        version_api.cpp
    )
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
  if(MIOPEN_ENABLE_HIPDNN_WRAPPER)
    # Tests that assert on logged/traced API names need to know which library
    # layout they are running against: when MIOpen is built as a public wrapper
    # over a private implementation, the public entry points are renamed with an
    # _impl suffix and MIOPEN_LOG_FUNCTION runs inside the renamed definition,
    # so the logged name carries that suffix.
    target_compile_definitions(${TEST_NAME} PRIVATE MIOPEN_ENABLE_HIPDNN_WRAPPER)
    # src/private is deliberately not on the include path: wrapper_routing.cpp is
    # the only test that needs the routing header and reaches it by relative
    # path. Adding it would also expose miopen_private_rename.h, which rewrites
    # the public miopenFoo names to their _impl form -- a name collision waiting
    # to happen in tests that have no business seeing it.
  endif()

  add_dependencies(miopen-tests ${TEST_NAME})
  add_dependencies(miopen-check ${TEST_NAME})

  target_compile_options(${TEST_NAME} PRIVATE -Wno-global-constructors -Wno-undef)
  target_include_directories(${TEST_NAME} PRIVATE ../ ../../src/kernels)

  if(MIOPEN_USE_HIPCONV)
    target_compile_definitions(${TEST_NAME} PRIVATE MIOPEN_USE_HIPCONV=1)
  endif()

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

  # CK setup for any test target that compiles CK-including sources. This must
  # be applied to every test executable so the test objects use the same CK
  # macro set as MIOpen and the CK plugins.
  if(MIOPEN_USE_COMPOSABLEKERNEL)
    if(MIOPEN_BUILD_CK)
      # The include directories also don't propagate correctly when using the
      # non-aliased targets. Manually including them fixes this for inline builds.
      target_include_directories(${TEST_NAME} SYSTEM PRIVATE
          ${MIOPEN_CK_INCLUDE_DIR}
          ${MIOPEN_CK_BUILD_INCLUDE_DIR}
          ${MIOPEN_CK_LIBRARY_INCLUDE_DIR}
      )
    else()
      # While building on TheRock we are missing CK include headers; pull them in
      # via the utility component (needed by CK utility gtests).
      if(NOT TARGET composable_kernel::utility)
        find_package(composable_kernel 1.0.0 REQUIRED COMPONENTS utility)
      endif()
      target_link_libraries(${TEST_NAME} composable_kernel::utility)
    endif()
    miopen_apply_ck_compile_definitions(${TEST_NAME})
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
    if("${TEST_NAME}" STREQUAL "test_bn_bwd_serial_run" OR "${TEST_NAME}" STREQUAL "test_bn_fwd_train_serial_run" OR "${TEST_NAME}" STREQUAL "test_bn_infer_serial_run" OR "${TEST_NAME}" STREQUAL "test_hip_graph_exist")
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

    # HipGraphExist OOMs (LLVM JIT) when shards share one GPU; filtered here, re-added below as RUN_SERIAL.
    add_gtest_negative_filter("*HipGraphExist*")

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

    # HipGraphExist was filtered out of the shards above (LLVM JIT OOM under concurrent GPU memory
    # pressure). Register it as a single RUN_SERIAL entry so ctest runs it alone, never alongside
    # another test on the GPU. Skipped when there is no GPU, since these are *GPU* tests.
    if(NOT MIOPEN_NO_GPU)
      add_test(NAME ${TEST_NAME}_hip_graph_serial
        COMMAND ${TEST_NAME}
          --gtest_filter=*HipGraphExist*
          --gtest_output=xml:${PROJECT_BINARY_DIR}/test_results/${TEST_NAME}_shard_hip_graph_serial.xml
        WORKING_DIRECTORY "${KERNELS_BINARY_DIR}"
      )
      set_tests_properties(${TEST_NAME}_hip_graph_serial PROPERTIES
        RUN_SERIAL TRUE
        ENVIRONMENT "MIOPEN_USER_DB_PATH=${CMAKE_CURRENT_BINARY_DIR};MIOPEN_INVOKED_FROM_CTEST=1"
      )
    endif()

  endif()

  # Among discrete binaries only hipdnn_shim_conv.cpp has shim tests; any other would replay
  # nothing, which the comparison rejects. No GPU: the shim tests are all GPU tests.
  if(MIOPEN_ENABLE_HIPDNN_WRAPPER AND NOT MIOPEN_NO_GPU AND (NOT MIOPEN_TEST_DISCRETE OR TEST_CPP STREQUAL "hipdnn_shim_conv.cpp"))
    miopen_forwarding_parity_entries(BUILD ${TEST_NAME})
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

# libMIOpen.so does not export the routing seam's symbols, so a test cannot link
# against them. Compile the source in here instead.
#
# Each test executable therefore carries its own copy of the seam, with its own
# GetForwardingMode() cache. These tests cover the seam's logic, not whether the
# wrapper stubs actually consult it.
#
# No include directory is needed: routing.cpp includes its own header by relative
# path, and adding src/private here would expose miopen_private_rename.h to every
# test-common source.
if(MIOPEN_ENABLE_HIPDNN_WRAPPER)
  target_sources(miopen_gtest_common PRIVATE ${PROJECT_SOURCE_DIR}/src/private/routing.cpp)
endif()
if(WIN32)
  # Refer to https://en.cppreference.com/w/cpp/language/types for details.
  target_compile_options(miopen_gtest_common PRIVATE $<BUILD_INTERFACE:$<$<CXX_COMPILER_ID:Clang>:-U__LP64__>>)
endif()

# Registers one forwarding parity entry. MODE is BUILD for the build tree or INSTALL for the
# packaged test list; the two differ only inside the branches below.
function(miopen_forwarding_parity_entry MODE TEST_NAME ENTRY_NAME LABELS)
  # Shard variables pinned: a sharded CI run exports them, and a replay honouring them would
  # compare only one shard's slice of the shim surface.
  set(PARITY_ENVIRONMENT "MIOPEN_INVOKED_FROM_CTEST=1;GTEST_TOTAL_SHARDS=1;GTEST_SHARD_INDEX=0")
  if(MODE STREQUAL "BUILD")
    add_test(NAME ${ENTRY_NAME}
      COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/script/run_forwarding_parity.py
        --gtest $<TARGET_FILE:${TEST_NAME}>
        --filter ${MIOPEN_FORWARDING_PARITY_FILTER}
        --output-dir ${PROJECT_BINARY_DIR}/test_results
        --lib-dir $<TARGET_FILE_DIR:MIOpen>
      WORKING_DIRECTORY "${KERNELS_BINARY_DIR}"
    )
    # RUN_SERIAL: unlike the shard entries, nothing else keeps this one off a busy GPU.
    set_tests_properties(${ENTRY_NAME} PROPERTIES
      RUN_SERIAL TRUE
      LABELS "${LABELS}"
      TIMEOUT ${MIOPEN_FORWARDING_PARITY_TIMEOUT}
      ENVIRONMENT "MIOPEN_USER_DB_PATH=${CMAKE_CURRENT_BINARY_DIR};${PARITY_ENVIRONMENT}"
    )
  elseif(MODE STREQUAL "INSTALL")
    # No --output-dir, --lib-dir or MIOPEN_USER_DB_PATH: the script's defaults fit the
    # installed layout, and the DB path names a build-tree directory. python3 because the
    # configure machine's interpreter path need not exist where the package is tested.
    file(APPEND "${INSTALL_TEST_FILE}"
"add_test(${ENTRY_NAME} python3 \"../run_forwarding_parity.py\" \"--gtest\" \"../${TEST_NAME}\" \"--filter\" \"${MIOPEN_FORWARDING_PARITY_FILTER}\")
set_tests_properties(${ENTRY_NAME} PROPERTIES RUN_SERIAL TRUE TIMEOUT ${MIOPEN_FORWARDING_PARITY_TIMEOUT} LABELS \"${LABELS}\" ENVIRONMENT \"${PARITY_ENVIRONMENT}\")

")
  else()
    message(FATAL_ERROR "miopen_forwarding_parity_entry: MODE must be BUILD or INSTALL, not '${MODE}'")
  endif()
endfunction()

# A bare entry plus one per ex_gpu_* label, because no single entry survives both forms of
# the runner's architecture filter (`-L ^ex_gpu_<arch>$` and `-LE ex_gpu`). `forwarding_parity`
# goes on the bare entry only, so selecting by it replays the shim surface once.
function(miopen_forwarding_parity_entries MODE TEST_NAME)
  miopen_forwarding_parity_entry(${MODE} ${TEST_NAME}
    "${TEST_NAME}_forwarding_parity" "forwarding_parity;${MIOPEN_FORWARDING_PARITY_TIER_LABELS}")
  foreach(PARITY_GPU_LABEL ${MIOPEN_FORWARDING_PARITY_GPU_LABELS})
    miopen_forwarding_parity_entry(${MODE} ${TEST_NAME}
      "${TEST_NAME}_forwarding_parity_${PARITY_GPU_LABEL}"
      "${MIOPEN_FORWARDING_PARITY_TIER_LABELS};${PARITY_GPU_LABEL}")
  endforeach()
endfunction()

# Set before the test executables are registered, because add_gtest() uses all of it.
# test/gtest/README.md describes the harness and why it is set up this way.
if(MIOPEN_ENABLE_HIPDNN_WRAPPER)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  set(MIOPEN_FORWARDING_PARITY_FILTER "*HipdnnShim*" CACHE STRING
    "gtest filter selecting the tests the forwarding parity harness replays")
  set(MIOPEN_FORWARDING_PARITY_TIMEOUT 3600 CACHE STRING
    "Timeout in seconds for one forwarding parity entry: two replays and a comparison")
  mark_as_advanced(MIOPEN_FORWARDING_PARITY_FILTER MIOPEN_FORWARDING_PARITY_TIMEOUT)

  # The tiers of TIER_ORDER in shared/ctest/parse_ctest_categories.py. Not ffm-quick/ffm-full,
  # which run a fixed list of patterns on a tight time budget.
  set(MIOPEN_FORWARDING_PARITY_TIER_LABELS "quick;standard;comprehensive;full")

  set(MIOPEN_TEST_CATEGORIES_YAML "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml")
  set(MIOPEN_TEST_CATEGORIES_PARSER "${ROCM_LIBRARIES_ROOT}/shared/ctest/parse_test_categories.py")
  include(${CMAKE_CURRENT_SOURCE_DIR}/ForwardingParityGpuLabels.cmake)

  # ENVIRONMENT_MODIFICATION rather than ENVIRONMENT, so each directory is joined with the
  # platform's path separator. No bytecode: the imported modules sit in the source tree.
  add_test(NAME test_forwarding_parity_scripts
    COMMAND ${Python3_EXECUTABLE} -m unittest -v
            test_compare_forwarding_runs
            test_run_forwarding_parity
            test_forwarding_parity_gpu_labels
            test_check_wrapper_abi
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  set_tests_properties(test_forwarding_parity_scripts PROPERTIES
    LABELS "${MIOPEN_FORWARDING_PARITY_TIER_LABELS}"
    TIMEOUT 300
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
    ENVIRONMENT_MODIFICATION
      "PYTHONPATH=path_list_prepend:${CMAKE_CURRENT_SOURCE_DIR};PYTHONPATH=path_list_prepend:${PROJECT_SOURCE_DIR}/script")

  add_test(NAME wrapper_abi_check
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/script/check_wrapper_abi.py
      --lib-dir $<TARGET_FILE_DIR:MIOpen>
      --baseline ${PROJECT_SOURCE_DIR}/test/public_abi/public_symbols.baseline
      --excluded ${PROJECT_SOURCE_DIR}/test/public_abi/wrapper_excluded_symbols.txt
      --public-header ${PROJECT_SOURCE_DIR}/include/miopen/miopen.h
  )
  set_tests_properties(wrapper_abi_check PROPERTIES
    LABELS "${MIOPEN_FORWARDING_PARITY_TIER_LABELS}"
    TIMEOUT 300)
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
  # CK include dirs / utility linkage / compile definitions for miopen_gtest
  # are applied inside add_gtest() (shared with discrete test executables).

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
    # GPU-free dbsync (StaticFDBSync) rocjitsu runner. Ships in the MIOpen dist so TheRock's
    # `miopen-dbsync` test component can invoke it from share/miopen/bin/ (pulled via the --miopen
    # artifact) and run dbsync GPU-free on a CPU runner. See the script header for the mechanism.
    install(FILES dbsync/run_dbsync_rocjitsu.py
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        DESTINATION ${DATA_INSTALL_DIR}/bin
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

# The packaged test list gets none of the add_test() calls above, so the harness entries are
# written into it here. No MIOPEN_NO_GPU condition: that describes the build machine, and
# packages are often built without a GPU and tested elsewhere.
if(MIOPEN_ENABLE_HIPDNN_WRAPPER AND NOT MIOPEN_TEST_DISCRETE)
  miopen_forwarding_parity_entries(INSTALL miopen_gtest)
endif()

# No MIOPEN_TEST_DISCRETE condition needed: a discrete build adds this directory
# EXCLUDE_FROM_ALL and installs nothing from it. No --public-header: it checks two source
# files, which the build-tree entry already covers.
if(MIOPEN_ENABLE_HIPDNN_WRAPPER)
  file(APPEND "${INSTALL_TEST_FILE}"
"add_test(wrapper_abi_check python3 \"../check_wrapper_abi.py\" \"--baseline\" \"../public_symbols.baseline\" \"--excluded\" \"../wrapper_excluded_symbols.txt\")
set_tests_properties(wrapper_abi_check PROPERTIES TIMEOUT 300 LABELS \"${MIOPEN_FORWARDING_PARITY_TIER_LABELS}\")

")
endif()

# Install CTestTestfile.cmake to bin/MIOpen/ subdirectory for TheRock distribution
if( NOT ENABLE_ASAN_PACKAGING )
  install(
      FILES "${INSTALL_TEST_FILE}"
      DESTINATION "${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}"
      COMPONENT tests
      RENAME "CTestTestfile.cmake"
  )

  # NOT MIOPEN_TEST_DISCRETE is for the parity scripts; a discrete build installs nothing
  # from this directory anyway, so it does not take the ABI check's files away.
  if(MIOPEN_ENABLE_HIPDNN_WRAPPER AND NOT MIOPEN_TEST_DISCRETE)
    install(
        PROGRAMS
          "${PROJECT_SOURCE_DIR}/script/check_wrapper_abi.py"
          "${PROJECT_SOURCE_DIR}/script/check_public_abi.py"
          "${PROJECT_SOURCE_DIR}/script/miopen_wrapper_libs.py"
          "${PROJECT_SOURCE_DIR}/script/run_forwarding_parity.py"
          "${PROJECT_SOURCE_DIR}/script/compare_forwarding_runs.py"
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT tests
    )
    install(
        FILES
          "${PROJECT_SOURCE_DIR}/test/public_abi/public_symbols.baseline"
          "${PROJECT_SOURCE_DIR}/test/public_abi/wrapper_excluded_symbols.txt"
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT tests
    )
  endif()
endif()
