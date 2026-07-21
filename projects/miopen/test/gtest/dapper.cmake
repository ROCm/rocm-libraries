# Restore the pre-Dapper single-gtest default that _dapper_native_init() would otherwise
# force. Used when Dapper does not run the native init (TheRock, or native mode=off).
macro(_dapper_default_single_gtest)
    if(NOT DEFINED MIOPEN_TEST_SINGLE_GTEST)
        if(MIOPEN_TEST_DISCRETE)
            set(MIOPEN_TEST_SINGLE_GTEST OFF)
        else()
            set(MIOPEN_TEST_SINGLE_GTEST ON)
        endif()
    endif()
endmacro()

# Dapper entry point. Selects the mode, wires up the native or TheRock pipeline, and (for an
# enabled native build) flips MIOPEN_ENABLE_DAPPER_NATIVE ON in the caller's scope.
#
# Dapper master switch (read by both the native/Jenkins flow and TheRock):
#   off      : Dapper disabled (no analysis / no shard-file / no dapper ctest tests;
#              the single gtest and its shard tests still build and run)
#   validate : native shard + dapper_diff coverage validation (Jenkins/MICI default)
#   union    : ACTIVE -- the reduced subtractive union filter actually runs (TheRock default)
macro(dapper_init)
    # Dapper's build-time tooling (Python + nm / C preprocessor) is not Windows-ready yet
    # (see script/dependency-parser/DAPPER.md "Known limitations"). Force the mode off on
    # Windows; the rest of this
    # macro then takes the 'off' path -- no python, no dapper wiring -- so the build falls
    # back to the normal full-category / non-dapper test flow.
    if(WIN32 OR CMAKE_HOST_WIN32)
        set(MIOPEN_DAPPER_MODE "off" CACHE STRING
            "Dapper mode: off | validate | union" FORCE)
        message(STATUS "Dapper: disabled on Windows (build-time tooling not yet ported)")
    endif()
    if(MIOPEN_BUILD_IN_THEROCK)
        set(_MIOPEN_DAPPER_MODE_DEFAULT "union")
    else()
        set(_MIOPEN_DAPPER_MODE_DEFAULT "validate")
    endif()
    set(MIOPEN_DAPPER_MODE "${_MIOPEN_DAPPER_MODE_DEFAULT}" CACHE STRING
        "Dapper mode: off | validate | union")
    set_property(CACHE MIOPEN_DAPPER_MODE PROPERTY STRINGS off validate union)
    set(MIOPEN_DAPPER_BASE_REF "origin/develop" CACHE STRING
        "Git ref to compute the Dapper impact diff against")
    # Additive attribution bridges run during 'parse' (see dependency-parser/main.py).
    # Default 'symbol' (nm-based, correctness-dominant); set to "" to disable all bridges.
    # Applies to both native and TheRock. The bridge module must exist on the current branch.
    set(MIOPEN_DAPPER_BRIDGES "symbol" CACHE STRING
        "Comma-separated dapper attribution bridges to run during 'parse' (symbol)")
    message(STATUS "Dapper: MIOPEN_DAPPER_MODE=${MIOPEN_DAPPER_MODE} (TheRock=${MIOPEN_BUILD_IN_THEROCK})")

    if(MIOPEN_BUILD_IN_THEROCK)
        _dapper_default_single_gtest()
        # The TheRock impact JSON + CTestTestfile finalize are produced later, once the install
        # CTestTestfile exists, via dapper_therock_generate_json() (called from CMakeLists.txt
        # after apply_test_category_labels). Here we only ensure the Python interpreter is found.
        if(NOT MIOPEN_DAPPER_MODE STREQUAL "off")
            find_package(Python 3 REQUIRED COMPONENTS Interpreter)
        endif()
    else()
        # Native/Jenkins build. 'validate'/'union' set up the native pipeline; 'off' disables it.
        # (Full native check->union activation is a follow-up; today 'union' here is exercised via
        # the existing `diff_check` target.)
        if(NOT MIOPEN_DAPPER_MODE STREQUAL "off")
            set(MIOPEN_ENABLE_DAPPER_NATIVE ON)
            find_package(Python 3 REQUIRED COMPONENTS Interpreter)
            _dapper_native_init()
        else()
            _dapper_default_single_gtest()
        endif()
    endif()
endmacro()

# Native/Jenkins Dapper setup: single-gtest mapping targets (shas/fixtures/mapping) plus the
# diff_check convenience target. dapper_dev_filters()/dapper_add_sharded_test() add the ctest
# analysis tests later, once the shard tests are registered.
macro(_dapper_native_init)
    set(MIOPEN_TEST_SINGLE_GTEST 1)
    # Dapper no longer forces the discrete test build. The dependency mapping is now
    # derived from the single aggregated miopen_gtest (per-source synthetic bin/test_<stem>
    # keys in enhanced_ninja_parser), so the ~280 discrete test binaries are unnecessary.
    # Respect whatever MIOPEN_TEST_DISCRETE the user/CI set (default off) instead of forcing it.

    # TRJS
    message(STATUS "------------------------------------ CMAKE_CURRENT_LIST_DIR: ${CMAKE_CURRENT_LIST_DIR}")
    message(STATUS "------------------------------------ CMAKE_SOURCE_DIR:       ${CMAKE_SOURCE_DIR}")
    message(STATUS "------------------------------------ CMAKE_BINARY_DIR:       ${CMAKE_BINARY_DIR}")

    set(MIOPEN_DAPPER_SRC_DIR "${CMAKE_BINARY_DIR}/../script/dependency-parser")
    set(MIOPEN_DAPPER_OUT_DIR "${CMAKE_BINARY_DIR}")
    set(SHARDS_FILE ${MIOPEN_DAPPER_OUT_DIR}/miopen_gtest_shards.txt)
    set(BUILD_NINJA "${MIOPEN_DAPPER_OUT_DIR}/build.ninja")
    set(SHAS_JSON "${MIOPEN_DAPPER_OUT_DIR}/miopen_dapper_shas.txt")
    set(MAPPING_JSON "${MIOPEN_DAPPER_OUT_DIR}/miopen_dapper_mapping.json")
    set(FIXTURES_JSON "${MIOPEN_DAPPER_OUT_DIR}/miopen_dapper_fixtures.json")
    set(TESTS_JSON "${MIOPEN_DAPPER_OUT_DIR}/miopen_dapper_tests.json")
    set(PY_MAIN "/${MIOPEN_DAPPER_SRC_DIR}/main.py")
    set(PY_FIXTURES "/${MIOPEN_DAPPER_SRC_DIR}/src/extract_gtest_fixtures.py")
    set(MIOPEN_GTEST_RUNNER "/${MIOPEN_DAPPER_SRC_DIR}/src/miopen_gtest_runner.py")
    set(MIOPEN_DAPPER_DIFF ${MIOPEN_DAPPER_SRC_DIR}/src/dapper_diff.py)

    # shas: git merge-base and HEAD SHAs; needed by both parse and select.
    add_custom_target(dapper_shas
        COMMENT "Generating ${SHAS_JSON}"
        COMMAND ${Python_EXECUTABLE} ${PY_MAIN} shas
        WORKING_DIRECTORY ${MIOPEN_DAPPER_OUT_DIR}
        VERBATIM
    )

    # fixtures: test-name -> gtest fixture mapping; reads compile_commands.json (always present).
    add_custom_target(dapper_fixtures
        COMMENT "Generating ${FIXTURES_JSON}"
        COMMAND ${Python_EXECUTABLE} ${PY_FIXTURES}
        WORKING_DIRECTORY ${MIOPEN_DAPPER_OUT_DIR}
        VERBATIM
    )
    add_dependencies(dapper_fixtures dapper_shas)

    # mapping: file -> test-executable mapping; needs tests built so build.ninja is final.
    add_custom_target(dapper_mapping
        COMMENT "Generating ${MAPPING_JSON}"
        COMMAND ${Python_EXECUTABLE} ${PY_MAIN} parse ${BUILD_NINJA} --bridges=${MIOPEN_DAPPER_BRIDGES}
        WORKING_DIRECTORY ${MIOPEN_DAPPER_OUT_DIR}
        VERBATIM
    )
    add_dependencies(dapper_mapping dapper_shas miopen-tests)

    # Drive all three pre-test files into the build graph before ctest runs.
    # TESTS_JSON is intentionally excluded — it is generated by the dapper_tests_generate
    # ctest test after the shard tests have written their XML output.
    add_custom_target(dapper_prebuild)
    add_dependencies(dapper_prebuild dapper_fixtures dapper_mapping)
    add_dependencies(miopen-check dapper_prebuild)

    add_custom_target(
        diff_check
        COMMENT "Running filtered gtests..."
        COMMAND ${Python_EXECUTABLE} "${MIOPEN_GTEST_RUNNER}" "${MIOPEN_DAPPER_OUT_DIR}/bin/miopen_gtest" "${TESTS_JSON}"
        WORKING_DIRECTORY ${MIOPEN_DAPPER_OUT_DIR}
        DEPENDS miopen_gtest miopen-tests
        VERBATIM
    )
endmacro()

macro(dapper_dev_filters)
# TODO: This is a development feature that will be removed following phase 3.

    message(STATUS "================== DAPPER DEVELOPMENT FILTERS")

    set(MIOPEN_DAPPER_DEV_FILTER_SHORT "*GPU_TestMhaFind20_FP32*:*GPU_TestMhaFind20_FP16*")
    set(MIOPEN_DAPPER_DEV_FILTER_LONGER "Smoke/GPU_BNCKFWDTrainLarge2D_FP16*:Smoke/GPU_BNOCLFWDTrainLarge2D_FP16*:Smoke/GPU_BNOCLFWDTrainLarge3D_FP16*:Smoke/GPU_BNCKFWDTrainLarge2D_BFP16*:Smoke/GPU_BNOCLFWDTrainLarge2D_BFP16*:Smoke/GPU_BNOCLFWDTrainLarge3D_BFP16*:Smoke/GPU_UnitTestConvSolverImplicitGemmFwdXdlops_FP16*:Smoke/GPU_UnitTestConvSolverImplicitGemmFwdXdlops_BFP16*-Smoke/GPU_BNOCLFWDTrainLarge2D_BFP16*:Smoke/GPU_BNOCLInferLarge2D_BFP16*:*/GPU_MIOpenDriver*:*GPU_TestMhaFind20_FP32*:*GPU_TestMhaFind20_FP16*:Smoke/GPU_BNOCLInferLarge2D_BFP16*:Smoke/GPU_BNOCLFWDTrainLarge2D_BFP16*")
    set(MIOPEN_DAPPER_DEV_FILTER_LONG "*Fusion*:*/GPU_BNBWD*_*:*/GPU_BNOCLBWD*_*:*/GPU_BNFWD*_*:*/GPU_BNOCLFWD*_*:*/GPU_BNInfer*_*:*/GPU_BNActivInfer_*:*/GPU_BNOCLInfer*_*:*/GPU_bn_infer*_*:CPU_*:*/CPU_*:*/GPU_Cat_*:*/GPU_ConvBiasActiv*:*/GPU_Conv*:*/GPU_conv*:*/GPU_UnitTestConv*:*/GPU_GetitemBwd*:*/GPU_GLU_*:*/GPU_GroupConv*:*/GPU_GroupNorm_*:*/GPU_GRUExtra_*:*/GPU_TestActivation*:*/GPU_HipBLASLtGEMMTest*:*/GPU_KernelTuningNetTestConv*:*/GPU_Kthvalue_*:*/GPU_LayerNormTest*:*/GPU_LayoutTransposeTest_*:*/GPU_Lrn*:*/GPU_lstm_extra*:*/GPU_MultiMarginLoss_*:*/GPU_ConvNonpack*:*/GPU_PerfConfig_HipImplicitGemm*:*/GPU_AsymPooling2d_*:*/GPU_WidePooling2d_*:*/GPU_PReLU_*:*/GPU_Reduce*:*/GPU_reduce_custom_*:*/GPU_regression_issue_*:*/GPU_RNNExtra_*:*/GPU_RoPE*:*/GPU_SoftMarginLoss*:*/GPU_T5LayerNormTest_*:*/GPU_Op4dTensorGenericTest_*:*/GPU_TernaryTensorOps_*:*/GPU_unaryTensorOps_*:*/GPU_Transformers*:*/GPU_TunaNetTest_*:*/")

    if(DEFINED MIOPEN_DEBUG_DAPPER_FILTER)
        if(MIOPEN_DEBUG_DAPPER_FILTER MATCHES "-")
            string(REGEX REPLACE "-.*$" "" MIOPEN_DEBUG_DAPPER_FILTER "${MIOPEN_DEBUG_DAPPER_FILTER}")
            message(WARNING "MIOPEN_DEBUG_DAPPER_FILTER is a positive-only filter; negative portion stripped. Using: '${MIOPEN_DEBUG_DAPPER_FILTER}'")
        endif()
    endif()

    if(DEFINED MIOPEN_DEBUG_DEV_FILTER)
        set(MIOPEN_DEBUG_DEV_FILTER_IN "${MIOPEN_DEBUG_DEV_FILTER}")
        string(TOUPPER "${MIOPEN_DEBUG_DEV_FILTER_IN}" MIOPEN_DEBUG_DEV_FILTER)
        message(STATUS "User-specified development category='${MIOPEN_DEBUG_DEV_FILTER}'")

        if(MIOPEN_DEBUG_DEV_FILTER STREQUAL "SHORT")
            set(MIOPEN_DEV_FILTER "${MIOPEN_DEBUG_DAPPER_FILTER}:${MIOPEN_DAPPER_DEV_FILTER_SHORT}")
        elseif(MIOPEN_DEBUG_DEV_FILTER STREQUAL "LONGER")
            set(MIOPEN_DEV_FILTER "${MIOPEN_DEBUG_DAPPER_FILTER}:${MIOPEN_DAPPER_DEV_FILTER_LONGER}")
        elseif(MIOPEN_DEBUG_DEV_FILTER STREQUAL "LONG")
            set(MIOPEN_DEV_FILTER "${MIOPEN_DEBUG_DAPPER_FILTER}:${MIOPEN_DAPPER_DEV_FILTER_LONG}")
        else()
            message(FATAL_ERROR "Aborting. Unknown MIOPEN_DEBUG_DEV_FILTER: '{MIOPEN_DEBUG_DEV_FILTER}'")
        endif()

        set(MIOPEN_CATEGORY "${MIOPEN_DEBUG_DEV_FILTER}")
        message(STATUS "Applying Category '${MIOPEN_DEBUG_DEV_FILTER}' with user filters '${MIOPEN_DEBUG_DAPPER_FILTER}'")
        set(MIOPEN_GTEST_FILTER "${MIOPEN_DEV_FILTER}")
        message(STATUS "gtest_filter=MIOPEN_DEV_FILTER: ${MIOPEN_DEV_FILTER}")
    elseif(DEFINED MIOPEN_DEBUG_DAPPER_FILTER)
        message(STATUS "Applying user filters '${MIOPEN_DEBUG_DAPPER_FILTER}'")
        set(MIOPEN_GTEST_FILTER "${MIOPEN_DEBUG_DAPPER_FILTER}:${MIOPEN_GTEST_FILTER}")
    endif()
endmacro()

macro(dapper_add_sharded_test)
    if(NOT DEFINED MIOPEN_CATEGORY)
        set(MIOPEN_CATEGORY "NONE")
    endif()

    # Run 'select' as a ctest test so it executes after the shard tests (which write the XML
    # output files) and before the dapper analysis test.
    add_test(NAME dapper_tests_generate
        COMMAND ${Python_EXECUTABLE} "${PY_MAIN}" select ${MAPPING_JSON}
            --fixturemap=${FIXTURES_JSON} --shardsfile=${SHARDS_FILE}
        WORKING_DIRECTORY ${MIOPEN_DAPPER_OUT_DIR}
    )
    set_tests_properties(dapper_tests_generate PROPERTIES
        DEPENDS "${MIOPEN_GTEST_SHARDS}"
        FIXTURES_SETUP dapper_tests_fixture
    )

    add_test(NAME miopen_gtest_sharded_dapper
        COMMAND ${Python_EXECUTABLE} ${MIOPEN_DAPPER_DIFF} "${TESTS_JSON}" "${MIOPEN_CATEGORY}" "${MIOPEN_GTEST_FILTER}"
        WORKING_DIRECTORY ${MIOPEN_DAPPER_OUT_DIR}
    )
    set_tests_properties(miopen_gtest_sharded_dapper PROPERTIES
        FIXTURES_REQUIRED dapper_tests_fixture
        RUN_SERIAL TRUE
    )

    # Force miopen_gtest_sharded_dapper to run dead last so its analysis summary prints at the very
    # end of the ctest output, not interleaved with (or buried under) the buffered output of
    # long-running tests that are still finishing (test_conv3d, test_immed_conv3d, ...).
    #
    # We use the DEPENDS test property rather than a FIXTURES_CLEANUP fixture on purpose. DEPENDS
    # only orders tests that are *already* selected to run; it never pulls extra tests into a run.
    # A FIXTURES_REQUIRED-on-every-test scheme would instead auto-pull the dapper setup/shard chain
    # into any subset run (e.g. `ctest -R conv_transposed_wrw`), which is slow and can fail because
    # dapper_tests_generate would run without shard XML present.
    #
    # dapper must run after every other test. The shard tests are in this directory scope
    # (MIOPEN_GTEST_SHARDS); the remaining tests are defined in the parent test/ directory, which is
    # fully processed before this subdirectory is added (add_subdirectory(gtest) is its last
    # statement), so the parent's TESTS directory property is complete and safe to read here.
    # Also depend on the tests registered in THIS directory so far (the shards and the
    # separately-registered ${TEST_NAME}_hip_graph_serial), not just the parent directory --
    # otherwise hip_graph_serial can run after the dapper analysis and bury its summary.
    # (TheRock-only category suites are added later by apply_test_category_labels and do not
    # run in MICI, so they are intentionally not required here.)
    get_property(_dapper_parent_tests DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/.. PROPERTY TESTS)
    get_property(_dapper_curdir_tests DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)
    set(_dapper_predecessors
        ${_dapper_parent_tests} ${_dapper_curdir_tests} ${MIOPEN_GTEST_SHARDS})
    # dapper_tests_generate is already ordered before dapper via dapper_tests_fixture; exclude it and
    # dapper itself from the dependency list.
    list(REMOVE_ITEM _dapper_predecessors miopen_gtest_sharded_dapper dapper_tests_generate)
    if(_dapper_predecessors)
        set_property(TEST miopen_gtest_sharded_dapper APPEND PROPERTY
            DEPENDS "${_dapper_predecessors}")
    endif()
    unset(_dapper_parent_tests)
    unset(_dapper_curdir_tests)
    unset(_dapper_predecessors)

    # CMake target equivalent to miopen_gtest_sharded_dapper
    add_custom_target(dapper_diff
        COMMAND ${Python_EXECUTABLE} ${MIOPEN_DAPPER_DIFF} "${TESTS_JSON}" "${MIOPEN_CATEGORY}" "${MIOPEN_GTEST_FILTER}"
        WORKING_DIRECTORY ${MIOPEN_DAPPER_OUT_DIR}
        DEPENDS miopen-check test_immed_conv3d test_tensor_vec test_conv3d_find2
        VERBATIM
    )

    # Full dapper pipeline using only existing shard and fixtures outputs — no rebuild.
    # Errors immediately if miopen_dapper_fixtures.json is missing (run dapper_fix_diff instead).
    add_custom_target(dapper_only_diff
        COMMENT "Running full dapper pipeline on existing shard outputs (no rebuild)..."
        COMMAND ${Python_EXECUTABLE} -c
            "import sys, pathlib; f=pathlib.Path('${FIXTURES_JSON}'); f.exists() or (print(f'Error: {f.name} not found. Run dapper_fix_diff to regenerate it via the preprocessor, or copy a valid file into: {f.parent}'), sys.exit(1))"
        COMMAND ${Python_EXECUTABLE} ${PY_MAIN} shas
        COMMAND ${Python_EXECUTABLE} ${PY_MAIN} parse ${BUILD_NINJA} --bridges=${MIOPEN_DAPPER_BRIDGES}
        COMMAND ${Python_EXECUTABLE} ${PY_MAIN} select ${MAPPING_JSON}
            --fixturemap=${FIXTURES_JSON} --shardsfile=${SHARDS_FILE}
        COMMAND ${Python_EXECUTABLE} ${MIOPEN_DAPPER_DIFF}
            ${TESTS_JSON} ${MIOPEN_CATEGORY} ${MIOPEN_GTEST_FILTER}
        WORKING_DIRECTORY ${MIOPEN_DAPPER_OUT_DIR}
        VERBATIM
    )

    # Same as dapper_only_diff but regenerates miopen_dapper_fixtures.json via the C preprocessor.
    # Use this when fixture mappings may be stale or the file is missing.
    add_custom_target(dapper_fix_diff
        COMMENT "Running full dapper pipeline, regenerating fixtures (no rebuild)..."
        COMMAND ${Python_EXECUTABLE} ${PY_MAIN} shas
        COMMAND ${Python_EXECUTABLE} ${PY_FIXTURES}
        COMMAND ${Python_EXECUTABLE} ${PY_MAIN} parse ${BUILD_NINJA} --bridges=${MIOPEN_DAPPER_BRIDGES}
        COMMAND ${Python_EXECUTABLE} ${PY_MAIN} select ${MAPPING_JSON}
            --fixturemap=${FIXTURES_JSON} --shardsfile=${SHARDS_FILE}
        COMMAND ${Python_EXECUTABLE} ${MIOPEN_DAPPER_DIFF}
            ${TESTS_JSON} ${MIOPEN_CATEGORY} ${MIOPEN_GTEST_FILTER}
        WORKING_DIRECTORY ${MIOPEN_DAPPER_OUT_DIR}
        VERBATIM
    )
endmacro()

# Build-time production of the Dapper artifacts for a single-gtest TheRock build. GPU-less
# (git diff + ninja deps + nm + fixture extraction). Produces, in one build-time command:
#   - miopen_dapper_tests.json : dapper_filter + fallback_mode, plus per dapper category
#     category_<NAME>_filter (original) and category_<NAME>_union (effective) -- the
#     downloadable record.
#   - a finalized CTestTestfile: for each Dapper-enabled category the existing '<name>_suite'
#     runs the subtractive union (burned directly into the add_test, exactly as develop's
#     direct-binary invocation), and a '<name>_unfiltered_suite' is added retaining the full
#     filter. Nothing dapper runs at ctest time; no runner/helper is installed.
#
# Called from CMakeLists.txt AFTER apply_test_category_labels (so install_ctest_file exists).
# Args: install_ctest_file = the configure-generated install CTestTestfile; test_yaml = the
# category yaml (for enable_dapper). Uses source-tree script paths (PROJECT_SOURCE_DIR) since
# TheRock builds out-of-source, and --source-dir so git runs in the MIOpen source worktree.
macro(dapper_therock_generate_json install_ctest_file test_yaml)
    set(_dapper_src "${PROJECT_SOURCE_DIR}/script/dependency-parser")
    set(_dapper_out "${CMAKE_BINARY_DIR}")
    set(_dapper_tests_json "${_dapper_out}/miopen_dapper_tests.json")
    set(_dapper_mapping_json "${_dapper_out}/miopen_dapper_mapping.json")
    set(_dapper_fixtures_json "${_dapper_out}/miopen_dapper_fixtures.json")
    set(_dapper_build_ninja "${_dapper_out}/build.ninja")
    set(_dapper_ctest_final "${_dapper_out}/dapper_CTestTestfile.cmake")

    add_custom_command(
        OUTPUT ${_dapper_tests_json} ${_dapper_ctest_final}
        COMMENT "Dapper: impact JSON + burning union into CTestTestfile (mode=${MIOPEN_DAPPER_MODE}, bridges=${MIOPEN_DAPPER_BRIDGES})"
        COMMAND ${Python_EXECUTABLE} ${_dapper_src}/main.py shas
            --base-ref ${MIOPEN_DAPPER_BASE_REF} --source-dir ${PROJECT_SOURCE_DIR}
        COMMAND ${Python_EXECUTABLE} ${_dapper_src}/src/extract_gtest_fixtures.py
        COMMAND ${Python_EXECUTABLE} ${_dapper_src}/main.py parse ${_dapper_build_ninja}
            --bridges=${MIOPEN_DAPPER_BRIDGES}
        COMMAND ${Python_EXECUTABLE} ${_dapper_src}/main.py select ${_dapper_mapping_json}
            --fixturemap=${_dapper_fixtures_json} --source-dir ${PROJECT_SOURCE_DIR}
            --output ${_dapper_tests_json}
        COMMAND ${Python_EXECUTABLE} ${_dapper_src}/main.py finalize-ctest
            --ctest-in ${install_ctest_file} --ctest-out ${_dapper_ctest_final}
            --yaml ${test_yaml} --dapper-json ${_dapper_tests_json}
        WORKING_DIRECTORY ${_dapper_out}
        DEPENDS miopen_gtest ${install_ctest_file}
        VERBATIM
    )
    add_custom_target(dapper_therock_json ALL
        DEPENDS ${_dapper_tests_json} ${_dapper_ctest_final})

    if(NOT ENABLE_ASAN_PACKAGING)
        # Install the reference JSON and the finalized CTestTestfile (with union burned in).
        install(FILES ${_dapper_tests_json}
            DESTINATION "${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}"
            COMPONENT tests)
        install(FILES ${_dapper_ctest_final}
            DESTINATION "${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}"
            COMPONENT tests
            RENAME "CTestTestfile.cmake")
    endif()
endmacro()
