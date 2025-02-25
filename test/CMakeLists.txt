
option(ROCROLLER_TESTS_SKIP_GUIDEPOSTS "Skip compiling guidepost tests." ON)
option(ROCROLLER_TESTS_SKIP_SLOW       "Skip slow tests (currently only Python tests)" ON)

if(CODE_COVERAGE)
  set(ROCROLLER_TESTS_SKIP_GUIDEPOSTS ON)
  set(ROCROLLER_TESTS_SKIP_SLOW OFF)
endif()

include(CTest)

# Header only/static, not visible to library users, so not necessary to use system packages
rocroller_add_dependency(googletest VERSION 1.12.1)
rocroller_add_dependency(Catch2 VERSION 3.7.0)
rocroller_add_dependency(mxDataGenerator HASH 05fd3a47fa83db1ccc81178ba360a8a7ade9bd49)

# Component stand-alone test
add_executable(ComponentTest standalone/ComponentTest.cpp)
target_compile_options(ComponentTest PUBLIC ${EXTRA_COMPILE_OPTIONS})
target_link_options(ComponentTest PUBLIC ${EXTRA_LINK_OPTIONS} -pthread)
target_link_libraries(ComponentTest
  rocroller_interface
  msgpackc-cxx
)
target_include_directories(ComponentTest PRIVATE
  ${PROJECT_SOURCE_DIR}/lib/include
                          )

add_library(common_test_utilities OBJECT
  common/CommonGraphs.cpp
  common/Utilities.cpp
  common/mxDataGen.cpp
)

target_compile_options(common_test_utilities PUBLIC ${COROUTINES_COMPILE_OPTION} ${EXTRA_COMPILE_OPTIONS})
target_link_options(common_test_utilities PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(common_test_utilities
  rocroller_interface
  openblas
  mxDataGenerator
)

target_include_directories(common_test_utilities PUBLIC
                           common/
                           )


set(
  rocRollerTests_SOURCES
  unit/ContextFixture.cpp
  unit/GPUContextFixture.cpp
  unit/GenericContextFixture.cpp

  unit/AddStreamKTest.cpp
  unit/AllocatingObserverTest.cpp
  unit/ArgumentLoaderTest.cpp
  unit/ArgumentValueTest.cpp
  unit/ArithmeticTest.cpp
  unit/AssemblerTest.cpp
  unit/AssertTest.cpp
  unit/BranchGeneratorTest.cpp
  unit/BufferDescriptorTest.cpp
  unit/CommandTest.cpp
  unit/CommonSubexpressionElimTest.cpp
  unit/ComponentTest.cpp
  unit/ControlGraphTest.cpp
  unit/ConversionInstructions/Convert_F16_F32.cpp
  unit/ConversionTest.cpp
  unit/CoordinateGraphTest.cpp
  unit/CopyGeneratorTest.cpp
  unit/CostTest.cpp
  unit/DataTypesTest.cpp
  unit/DependencyTest.cpp
  unit/ErrorTest.cpp
  unit/F6Test.cpp
  unit/F8Test.cpp
  unit/FP4Test.cpp
  unit/FastDivisionTest.cpp
  unit/LowerExponentialTest.cpp
  unit/FastMultiplicationTest.cpp
  unit/FileWritingObserverTest.cpp
  unit/GEMMFusion.cpp
  unit/GEMMTest.cpp
  unit/GPUArchitectureTest.cpp
  unit/GPUInstructionInfoTest.cpp
  unit/GeneratorTest.cpp
  unit/HalfPrecisionTest.cpp
  unit/HypergraphTest.cpp
  unit/InstructionTest.cpp
  unit/KernelGraphTest.cpp
  unit/KernelGraphTest/ConstantPropagation.cpp
  unit/KernelGraphTest/KernelGraphUtils.cpp
  unit/KernelGraphTest/Simplify.cpp
  unit/KernelGraphTest/UpdateParameters.cpp
  unit/KernelIndexTest.cpp
  unit/KernelOptionsTest.cpp
  unit/KernelTest.cpp
  unit/LDSTest.cpp
  unit/LabelTest.cpp
  unit/LDSTest.cpp
  unit/LDSTileCopyTest.cpp
  unit/LockTest.cpp
  unit/LowerExponentialTest.cpp
  unit/MFMAHazardObserverTest.cpp
  unit/MFMAUnitObserverTest.cpp
  unit/MatrixMultiplyTest.cpp
  unit/MemoryInstructionsTest.cpp
  unit/MetaObserverTest.cpp
  unit/MixedArithmeticTest.cpp
  unit/OperationTagTest.cpp
  unit/RegisterAllocatorTest.cpp
  unit/RandomTest.cpp
  unit/RegisterTest.cpp
  unit/RegisterLivenessObserverTest.cpp
  unit/SchedulerTest.cpp
  unit/ScopeTest.cpp
  unit/SourceMatcherTest.cpp
  unit/SupportedInstructionObserverTest.cpp
  unit/TileTransposeAdd.cpp
  unit/TimerTest.cpp
  unit/TransposeLoadsTest.cpp
  unit/UtilsTest.cpp
  unit/VariantTest.cpp
  unit/VectorAddBenchmark.cpp
  unit/WaitCountObserverTest.cpp
  unit/WaitCountTest.cpp
)

if (NOT ROCROLLER_TESTS_SKIP_GUIDEPOSTS)
  list(APPEND rocRollerTests_SOURCES
    unit/GemmGuidePost/GemmGuidePost.cpp
    unit/GemmGuidePost/GemmGuidePost_Minimal.cpp
    unit/GemmGuidePost/GemmGuidePost_Optimized.cpp
    unit/GemmGuidePost/HGemmGuidePost_Minimal.cpp
    unit/GemmGuidePost/HGemmGuidePost_Optimized.cpp
  )
endif()

add_executable(
  rocRollerTests
  ${rocRollerTests_SOURCES}
)

find_package(OpenMP)
target_compile_options(rocRollerTests PUBLIC ${EXTRA_COMPILE_OPTIONS})
target_link_options(rocRollerTests PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(
  rocRollerTests
  common_test_utilities
  msgpackc-cxx
  gtest_main
  gmock
  OpenMP::OpenMP_CXX
)

if(YAML_BACKEND STREQUAL "YAML_CPP")
  target_compile_definitions(rocRollerTests PUBLIC ROCROLLER_TESTS_USE_YAML_CPP)
endif()

set(
  rocRollerTests_catch_SOURCES

  catch/CustomAssertions.cpp
  catch/TestKernels.cpp

  catch/BinaryExpressionTest.cpp
  catch/CommandArgumentValueTest.cpp
  catch/CommandTest.cpp
  catch/CustomAssertionTest.cpp
  catch/CustomMatcherTest.cpp
  catch/ExpressionContainsSubExpressionTest.cpp
  catch/ExpressionIdenticalTest.cpp
  catch/ExpressionSerializationTest.cpp
  catch/ExpressionTest.cpp
  catch/ExpressionToStringTest.cpp
  catch/ExpressionTransformationTest.cpp
  catch/FastArithmeticTest.cpp
  catch/IdentifyParallelDimensionsTest.cpp
  catch/KernelArgumentsTest.cpp
  catch/KernelGraphConstraintsTest.cpp
  catch/RandomNumberExpressionTest.cpp
  catch/LargerLDSTest.cpp
  catch/MultipleLDSAllocationTest.cpp
  catch/ScalarExpressionTest.cpp
  catch/SettingsTest.cpp
  catch/TernaryExpressionTest.cpp
  catch/TestContextTest.cpp
  catch/UnrollLoopsTest.cpp
  catch/UtilsTest.cpp
  catch/RandomNumberGenerationTest.cpp
  catch/mxDataGeneratorTest.cpp
  catch/GlobalLoadStoreTest.cpp
  catch/HazardObserverTest.cpp
)

add_executable(rocRollerTests_catch ${rocRollerTests_catch_SOURCES})
target_compile_options(rocRollerTests_catch PUBLIC ${EXTRA_COMPILE_OPTIONS})

target_link_options(rocRollerTests_catch PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(
  rocRollerTests_catch
  common_test_utilities
  msgpackc-cxx
  Catch2::Catch2WithMain
  OpenMP::OpenMP_CXX
)

if(YAML_BACKEND STREQUAL "YAML_CPP")
  target_compile_definitions(rocRollerTests_catch PUBLIC ROCROLLER_TESTS_USE_YAML_CPP)
endif()

# ArchGen unit tests
add_executable(
  ArchGenTests
  unit/GPUArchitectureGeneratorTest.cpp
)

target_compile_options(ArchGenTests PUBLIC  ${COROUTINES_COMPILE_OPTION} ${EXTRA_COMPILE_OPTIONS})
target_link_options(ArchGenTests PUBLIC ${EXTRA_LINK_OPTIONS})
target_include_directories(ArchGenTests PRIVATE
                           ${PROJECT_SOURCE_DIR}/GPUArchitectureGenerator/include
                           )

target_link_libraries(
  ArchGenTests PUBLIC
  common_test_utilities
  rocroller_interface
  gtest_main
  gmock
  msgpackc-cxx
  isa_spec_manager
)


if (NOT CODE_ANALYSIS)
  set(TEST_REPORT_DIR "${PROJECT_BINARY_DIR}/test_report")

  include(GoogleTest)
  gtest_discover_tests(rocRollerTests
    XML_OUTPUT_DIR ${TEST_REPORT_DIR}
    TEST_FILTER "-*GPU_*"
    DISCOVERY_MODE PRE_TEST
  )
  gtest_discover_tests(rocRollerTests
    XML_OUTPUT_DIR ${TEST_REPORT_DIR}
    TEST_FILTER "*GPU_*"
    PROPERTIES "LABELS" "GPU"
    DISCOVERY_MODE PRE_TEST
  )
  gtest_discover_tests(ArchGenTests
    XML_OUTPUT_DIR ${TEST_REPORT_DIR}
    DISCOVERY_MODE PRE_TEST
  )

  # path to catch scripts added to CMAKE_MODULE_PATH when Catch2 is added
  include(Catch)
  catch_discover_tests(rocRollerTests_catch
      REPORTER JUnit
      OUTPUT_DIR ${TEST_REPORT_DIR}
      OUTPUT_SUFFIX ".xml"
      DISCOVERY_MODE PRE_TEST
      EXTRA_ARGS "--reporter console::out=-::colour-mode=ansi"
      TEST_SPEC "[gpu]"
      PROPERTIES
      "LABELS" "CATCH"
      "LABELS" "GPU"
      SKIP_REGULAR_EXPRESSION "[1-9][0-9]* skipped"
  )

  include(Catch)
  catch_discover_tests(rocRollerTests_catch
      REPORTER JUnit
      OUTPUT_DIR ${TEST_REPORT_DIR}
      OUTPUT_SUFFIX ".xml"
      DISCOVERY_MODE PRE_TEST
      EXTRA_ARGS "--reporter console::out=-::colour-mode=ansi"
      TEST_SPEC "~[gpu]"
      PROPERTIES
      "LABELS" "CATCH"
      SKIP_REGULAR_EXPRESSION "[1-9][0-9]* skipped"
  )

  # Add script linting
  add_test(NAME PythonLint
           COMMAND flake8
           WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/scripts)
  set_tests_properties(PythonLint
                       PROPERTIES
                        ENVIRONMENT "ROCROLLER_BUILD_DIR=${PROJECT_BINARY_DIR}"
                        "LABELS" "PYTHON")

  if (ROCROLLER_TESTS_SKIP_SLOW)
    set(PYTHON_TEST_SKIP "-m not slow")
  endif()

  # Add rrperf unit tests
  add_test(NAME PythonTest
           COMMAND pytest -s --cov-report html:python_cov_html --cov=rrperf scripts ${PYTHON_TEST_SKIP} --junit-xml=${TEST_REPORT_DIR}/python_tests.xml
           WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  set_tests_properties(PythonTest
                       PROPERTIES
                        ENVIRONMENT "ROCROLLER_BUILD_DIR=${PROJECT_BINARY_DIR};PYTHONPATH=${PROJECT_SOURCE_DIR}/scripts/lib"
                        "LABELS" "PYTHON")

  add_test(NAME GraupelInstall
           COMMAND npm install
           WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/utils/graupel")

  set_tests_properties(GraupelInstall
                        PROPERTIES
                        FIXTURES_SETUP Graupel
                        "LABELS" "JAVASCRIPT")

  add_test(NAME GraupelTests
           COMMAND npm test
           WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/utils/graupel")

  set_tests_properties(GraupelTests PROPERTIES
                       ENVIRONMENT "JEST_JUNIT_OUTPUT_FILE=${TEST_REPORT_DIR}/graupel_tests.xml"
                       FIXTURES_REQUIRED Graupel
                       "LABELS" "JAVASCRIPT")
endif()
