
cmake_minimum_required(VERSION 3.16)

project(rocRoller VERSION 1.0)

option(ROCROLLER_ENABLE_TIMERS         "Enable rocRoller timer code." OFF)
option(ROCROLLER_TESTS_SKIP_GUIDEPOSTS "Skip compiling guidepost tests." OFF)
option(ROCROLLER_TESTS_USE_YAML_CPP    "Use yaml-cpp in testing." ON)
option(ROCROLLER_USE_HIP               "Use HIP within the rocRoller." ON)
option(SKIP_CPPCHECK                    "Skip CPPCHECK." OFF)

option(CODE_COVERAGE                    "Build with code coverage flags (clang only)" OFF)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  option(ROCROLLER_USE_LLVM      "Use LLVM for generating metadata." OFF)
  option(ROCROLLER_USE_YAML_CPP  "Use yaml-cpp for generating metadata." ON)

  set(COROUTINES_COMPILE_OPTION -fcoroutines-ts)
  set(EXTRA_COMPILE_OPTIONS -stdlib=libc++)
  set(EXTRA_LINK_OPTIONS -lc++ -lm -fuse-ld=lld)
  set(CMAKE_CXX_FLAGS -stdlib=libc++)
  set_source_files_properties(lib/source/Assembler.cpp PROPERTIES COMPILE_FLAGS "-stdlib=libstdc++")
else()
  option(ROCROLLER_USE_LLVM      "Use LLVM for generating metadata." ON)
  option(ROCROLLER_USE_YAML_CPP  "Use yaml-cpp for generating metadata." OFF)

  set(COROUTINES_COMPILE_OPTION -fcoroutines)
  set(EXTRA_COMPILE_OPTIONS )
  set(EXTRA_LINK_OPTIONS )
endif()

if (ROCROLLER_ENABLE_TIMERS)
  add_definitions(-DROCROLLER_ENABLE_TIMERS)
  add_compile_options(-pg -fno-omit-frame-pointer)
endif()

if(CODE_COVERAGE)
  add_definitions(-DROCROLLER_CODE_COVERAGE)
  set(EXTRA_COMPILE_OPTIONS -fprofile-instr-generate -fcoverage-mapping ${EXTRA_COMPILE_OPTIONS})
  set(EXTRA_LINK_OPTIONS -fprofile-instr-generate -fcoverage-mapping ${EXTRA_LINK_OPTIONS})
endif()

if(ROCROLLER_USE_LLVM AND ROCROLLER_USE_YAML_CPP)
  message(FATAL_ERROR "Use either LLVM or yaml-cpp backend for serialization, not both.")
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/e2239ee6043f73722e7aa812a459f54a28552929.zip
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  libdivide
  GIT_REPOSITORY https://github.com/ridiculousfish/libdivide.git
  GIT_TAG 5.0
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)

FetchContent_GetProperties(asio)
if(NOT libdivide_POPULATED)
  FetchContent_Populate(libdivide)
endif()

# TODO Switch to pre-built spdlog when build bugs are fixed, v1.9.3 will have fixed fmt lib

# FetchContent_Declare(
#   spdlog
#   GIT_REPOSITORY https://github.com/gabime/spdlog.git
#   GIT_TAG v1.9.2
# )
# if(NOT spdlog_POPULATED)
#   FetchContent_Populate(spdlog)
#   add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR})
#   set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)
# endif()

# FetchContent_Declare(
#   fmt
#   GIT_REPOSITORY https://github.com/fmtlib/fmt.git
#   GIT_TAG 8.1.1
# )
# FetchContent_MakeAvailable(fmt)

add_definitions(-DMSGPACK_NO_BOOST)
set(MSGPACK_USE_BOOST OFF CACHE INTERNAL "")
FetchContent_Declare(
  msgpack
  GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
  GIT_TAG cpp-4.1.1
)
FetchContent_MakeAvailable(msgpack)

if(ROCROLLER_USE_YAML_CPP OR ROCROLLER_TESTS_USE_YAML_CPP)
FetchContent_Declare(
  yaml-cpp
  URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.7.0.zip
)
FetchContent_MakeAvailable(yaml-cpp)
endif()

target_compile_options(gtest PUBLIC ${EXTRA_COMPILE_OPTIONS})
target_compile_options(gmock PUBLIC ${EXTRA_COMPILE_OPTIONS})

if(NOT SKIP_CPPCHECK)
  find_program(CMAKE_CXX_CPPCHECK NAMES cppcheck)
  if (CMAKE_CXX_CPPCHECK)
      list(
          APPEND CMAKE_CXX_CPPCHECK
              "--enable=warning"
              "--error-exitcode=10"
              "--inconclusive"
              "--force"
              "--inline-suppr"
              "-q"
              "--library=googletest"
              "--suppressions-list=${CMAKE_SOURCE_DIR}/CppCheckSuppressions.txt"
      )
  endif()
endif()

enable_testing()
include(CTest)

find_package(LLVM 13.0.1 REQUIRED)

llvm_map_components_to_libnames(llvm_libs all)
find_package(LLD REQUIRED CONFIG HINTS ${LLVM_DIR}/../lib/cmake/lld/)

if(ROCROLLER_USE_HIP)
  if(NOT DEFINED HIP_PATH)
      if(NOT DEFINED ENV{HIP_PATH})
          set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
      else()
          set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
      endif()
  endif()
  set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
  find_package(HIP)
endif()

if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/extern/spdlog)
  execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} submodule update --init -- extern/spdlog)
endif ()
include(ExternalProject)
ExternalProject_Add(spdlog
 PREFIX spdlog
 SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/spdlog
 CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
 -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
 -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
 -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
 -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/spdlog
 -DSPDLOG_BUILD_SHARED=OFF
)

# ArchitectureGenerator
add_executable(
  GPUArchitectureGenerator
  GPUArchitectureGenerator/source/GPUArchitectureGenerator.cpp
  lib/source/Error.cpp
  lib/source/GPUArchitecture.cpp
)
add_dependencies(GPUArchitectureGenerator spdlog)
target_compile_features(GPUArchitectureGenerator PUBLIC cxx_std_20)
target_compile_options(GPUArchitectureGenerator PUBLIC  ${COROUTINES_COMPILE_OPTION} ${EXTRA_COMPILE_OPTIONS})
target_link_options(GPUArchitectureGenerator PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(
  GPUArchitectureGenerator PUBLIC
  ${llvm_libs}
  lldELF
  msgpackc-cxx
)
target_include_directories(GPUArchitectureGenerator PUBLIC
                           lib/include
                           GPUArchitectureGenerator/include
                           lib/include/rocRoller
                           )

target_include_directories(GPUArchitectureGenerator SYSTEM PUBLIC
                           ${CMAKE_CURRENT_BINARY_DIR}/spdlog/include/
                           ${LLVM_INCLUDE_DIRS}
                           ${LLD_INCLUDE_DIRS})

if(ROCROLLER_USE_HIP)
  target_include_directories(GPUArchitectureGenerator SYSTEM PUBLIC ${HIP_ROOT_DIR}/include)
  target_compile_definitions(GPUArchitectureGenerator PUBLIC ROCROLLER_USE_HIP __HIP_PLATFORM_AMD__)
endif()

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/source/rocRoller)
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/source/rocRoller/GPUArchitecture_def.yaml
  COMMAND ${CMAKE_BINARY_DIR}/GPUArchitectureGenerator ${CMAKE_BINARY_DIR}/source/rocRoller/GPUArchitecture_def.yaml -Y
  DEPENDS GPUArchitectureGenerator
)
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/source/rocRoller/GPUArchitecture_def.msgpack
  COMMAND ${CMAKE_BINARY_DIR}/GPUArchitectureGenerator ${CMAKE_BINARY_DIR}/source/rocRoller/GPUArchitecture_def.msgpack
  DEPENDS GPUArchitectureGenerator
)
ADD_CUSTOM_TARGET(
  GPUArchitecture_def
  ALL
  DEPENDS
  ${CMAKE_BINARY_DIR}/source/rocRoller/GPUArchitecture_def.yaml
  ${CMAKE_BINARY_DIR}/source/rocRoller/GPUArchitecture_def.msgpack
)

# rocRoller library
set(rocroller_src
 lib/source/Arithmetic/Arithmetic.cpp
 lib/source/Arithmetic/Double.cpp
 lib/source/Arithmetic/Float.cpp
 lib/source/Arithmetic/Int32.cpp
 lib/source/Arithmetic/Int64.cpp
 lib/source/Arithmetic/MatrixMultiply.cpp

 lib/source/CodeGen/LowerFromKernelGraph.cpp
 lib/source/CodeGen/Arithmetic/Add.cpp
 lib/source/CodeGen/Arithmetic/ArithmeticGenerator.cpp
 lib/source/CodeGen/Arithmetic/BitwiseOr.cpp
 lib/source/CodeGen/Arithmetic/Convert.cpp
 lib/source/CodeGen/Arithmetic/Divide.cpp
 lib/source/CodeGen/Arithmetic/Equal.cpp
 lib/source/CodeGen/Arithmetic/GreaterThan.cpp
 lib/source/CodeGen/Arithmetic/GreaterThanEqual.cpp
 lib/source/CodeGen/Arithmetic/LessThan.cpp
 lib/source/CodeGen/Arithmetic/LessThanEqual.cpp
 lib/source/CodeGen/Arithmetic/Modulo.cpp
 lib/source/CodeGen/Arithmetic/Multiply.cpp
 lib/source/CodeGen/Arithmetic/MultiplyHigh.cpp
 lib/source/CodeGen/Arithmetic/BitwiseAnd.cpp
 lib/source/CodeGen/Arithmetic/BitwiseXor.cpp
 lib/source/CodeGen/Arithmetic/ShiftL.cpp
 lib/source/CodeGen/Arithmetic/ShiftR.cpp
 lib/source/CodeGen/Arithmetic/SignedShiftR.cpp
 lib/source/CodeGen/Arithmetic/Subtract.cpp
 lib/source/CodeGen/Buffer.cpp
 lib/source/CodeGen/MemoryInstructions.cpp

 lib/source/KernelGraph/ControlGraph/ControlEdge.cpp
 lib/source/KernelGraph/ControlGraph/ControlGraph.cpp
 lib/source/KernelGraph/CoordGraph/Dimension.cpp
 lib/source/KernelGraph/CoordGraph/Transformer.cpp
 lib/source/KernelGraph/CoordinateTransform/Dimension.cpp
 lib/source/KernelGraph/CoordinateTransform/HyperGraph.cpp
 lib/source/KernelGraph/CoordinateTransform/Transformer.cpp
 lib/source/KernelGraph/Transformations/CleanGraph.cpp
 lib/source/KernelGraph/Transformations/KernelGraph_loop.cpp
 lib/source/KernelGraph/Transformations/KernelGraph_unrollKernel.cpp
 lib/source/KernelGraph/Transformations/LowerLinear.cpp
 lib/source/KernelGraph/Transformations/LowerTile.cpp
 lib/source/KernelGraph/Transformations/UpdateParameters.cpp
 lib/source/KernelGraph/KernelGraph.cpp
 lib/source/KernelGraph/KernelHypergraph.cpp
 lib/source/KernelGraph/LowerFromCommand.cpp
 lib/source/KernelGraph/TagType.cpp
 lib/source/KernelGraph/Utils.cpp

 lib/source/Assembler.cpp
 lib/source/AssemblyKernel.cpp
 lib/source/AssemblyKernel_serialization.cpp
 lib/source/Context.cpp
 lib/source/DataTypes.cpp
 lib/source/Error.cpp
 lib/source/ExecutableKernel.cpp
 lib/source/Expression.cpp
 lib/source/Expression_evaluate.cpp
 lib/source/Expression_generate.cpp
 lib/source/Expression_serialization.cpp
 lib/source/ExpressionTransformations/FastArithmetic.cpp
 lib/source/ExpressionTransformations/FastDivision.cpp
 lib/source/ExpressionTransformations/FastMultiplication.cpp
 lib/source/ExpressionTransformations/LaunchTimeSubExpressions.cpp
 lib/source/ExpressionTransformations/Simplify.cpp
 lib/source/ExpressionTransformations/Fuse.cpp
 lib/source/GPUArchitecture.cpp
 lib/source/GPUArchitectureLibrary.cpp
 lib/source/KernelArguments.cpp
 lib/source/KernelOptions.cpp
 lib/source/LDSAllocator.cpp
 lib/source/Logging.cpp
 lib/source/Register.cpp
 lib/source/ScheduledInstructions.cpp
 lib/source/Scheduler.cpp
 lib/source/Timer.cpp
)

if(ROCROLLER_USE_HIP)
  set(rocroller_src
   ${rocroller_src}
   lib/source/HIPTimer.cpp
  )
endif()

add_library(rocroller SHARED
 ${rocroller_src}
)

add_dependencies(rocroller spdlog)
ADD_DEPENDENCIES(rocroller GPUArchitecture_def)

target_compile_features(rocroller PUBLIC cxx_std_20)
target_compile_options(rocroller PUBLIC ${COROUTINES_COMPILE_OPTION} ${EXTRA_COMPILE_OPTIONS})
target_link_options(rocroller PUBLIC ${EXTRA_LINK_OPTIONS})
#target_compile_options(rocroller PUBLIC -fconcepts-diagnostics-depth=2)
target_include_directories(rocroller PUBLIC
                           lib/include
                           lib/include/rocRoller
                           )

# Using SYSTEM include paths for third party libraries speeds up cppcheck a lot
target_include_directories(rocroller SYSTEM PUBLIC
                           ${libdivide_SOURCE_DIR}/
                           ${CMAKE_CURRENT_BINARY_DIR}/spdlog/include/)
# target_link_libraries(rocroller
#   PRIVATE spdlog::spdlog
#   PUBLIC fmt::fmt
# )

if(ROCROLLER_USE_YAML_CPP)
  target_compile_definitions(rocroller PUBLIC ROCROLLER_USE_YAML_CPP)
  target_link_libraries(rocroller PUBLIC yaml-cpp)
  set_property(TARGET yaml-cpp PROPERTY POSITION_INDEPENDENT_CODE ON)

  target_compile_definitions(GPUArchitectureGenerator PUBLIC ROCROLLER_USE_YAML_CPP)
  target_link_libraries(GPUArchitectureGenerator PUBLIC yaml-cpp)
endif()
if(ROCROLLER_USE_LLVM)
  target_compile_definitions(rocroller PUBLIC ROCROLLER_USE_LLVM)
  target_compile_definitions(GPUArchitectureGenerator PUBLIC ROCROLLER_USE_LLVM)
endif()

target_link_libraries(rocroller PRIVATE ${llvm_libs} lldELF PUBLIC msgpackc-cxx)
target_include_directories(rocroller SYSTEM PUBLIC
                            ${LLVM_INCLUDE_DIRS} ${LLD_INCLUDE_DIRS})

if(ROCROLLER_USE_HIP)
  target_link_directories(rocroller PUBLIC ${HIP_ROOT_DIR}/lib)
  target_include_directories(rocroller SYSTEM PUBLIC ${HIP_ROOT_DIR}/include)
  target_link_libraries(rocroller PUBLIC amdhip64)
  target_compile_definitions(rocroller PUBLIC ROCROLLER_USE_HIP __HIP_PLATFORM_AMD__)
endif()

# Component stand-alone test
add_executable(ComponentTest test/standalone/ComponentTest.cpp)
target_compile_options(ComponentTest PUBLIC ${EXTRA_COMPILE_OPTIONS})
target_link_options(ComponentTest PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(ComponentTest
  rocroller
  msgpackc-cxx
)
target_include_directories(ComponentTest PUBLIC
                            lib/include
                            )

# add_executable(CoroutineTest test/standalone/coroutine_test.cpp)
# target_link_libraries(CoroutineTest rocRoller)
# target_compile_options(CoroutineTest PUBLIC ${EXTRA_COMPILE_OPTIONS})
# target_link_options(CoroutineTest PUBLIC ${EXTRA_LINK_OPTIONS})
# target_link_libraries(CoroutineTest rocRoller)
# target_include_directories(CoroutineTest PUBLIC
#                             lib/include
#                             )

# rocRoller unit tests
set(
  rocRollerTests_SOURCES
  test/unit/ContextFixture.cpp
  test/unit/GenericContextFixture.cpp
  test/unit/GPUContextFixture.cpp

  test/unit/ArgumentLoaderTest.cpp
  test/unit/ArgumentValueTest.cpp
  test/unit/ArithmeticTest.cpp
  test/unit/AssemblerTest.cpp
  test/unit/BranchGeneratorTest.cpp
  test/unit/CommandTest.cpp
  test/unit/ComponentTest.cpp
  test/unit/ControlHypergraphTest.cpp
  test/unit/CoordinateTransformTest.cpp
  test/unit/CoordinateHypergraphTest.cpp
  test/unit/CopyGeneratorTest.cpp
  test/unit/DataTypesTest.cpp
  test/unit/DependencyTest.cpp
  test/unit/ErrorTest.cpp
  test/unit/ExpressionTest.cpp
  test/unit/FastDivisionTest.cpp
  test/unit/FastMultiplicationTest.cpp
  test/unit/FileWritingObserverTest.cpp
  test/unit/GPUArchitectureTest.cpp
  test/unit/GPUInstructionInfoTest.cpp
  test/unit/GEMMTest.cpp
  test/unit/GeneratorTest.cpp
  test/unit/HalfPrecisionTest.cpp
  test/unit/HypergraphTest.cpp
  test/unit/InstructionTest.cpp
  test/unit/KernelArgumentsTest.cpp
  test/unit/KernelGraphTest.cpp
  test/unit/KernelHypergraphTest.cpp
  test/unit/KernelIndexTest.cpp
  test/unit/KernelOptionsTest.cpp
  test/unit/KernelTest.cpp
  test/unit/LabelTest.cpp
  test/unit/LDSTest.cpp
  test/unit/LockTest.cpp
  test/unit/MemoryInstructionsTest.cpp
  test/unit/MatrixMultiplyTest.cpp
  test/unit/MetaObserverTest.cpp
  test/unit/MFMAObserverTest.cpp
  test/unit/MixedArithmeticTest.cpp
  test/unit/RandomTest.cpp
  test/unit/RegisterAllocatorTest.cpp
  test/unit/RegisterTest.cpp
  test/unit/SchedulerTest.cpp
  test/unit/SettingsTest.cpp
  test/unit/SimplifyTest.cpp
  test/unit/SourceMatcherTest.cpp
  test/unit/TileTransposeAdd.cpp
  test/unit/TimerTest.cpp
  test/unit/Utilities.cpp
  test/unit/UtilsTest.cpp
  test/unit/VariantTest.cpp
  test/unit/VectorAddBenchmark.cpp
  test/unit/WaitCountObserverTest.cpp
  test/unit/WaitCountTest.cpp
)

if (NOT ROCROLLER_TESTS_SKIP_GUIDEPOSTS)
  set(
    rocRollerTests_SOURCES
    ${rocRollerTests_SOURCES}
    test/unit/GemmGuidePost/GemmGuidePost.cpp
    test/unit/GemmGuidePost/GemmGuidePost_Minimal.cpp
    test/unit/GemmGuidePost/GemmGuidePost_Optimized.cpp
    test/unit/GemmGuidePost/HGemmGuidePost_Minimal.cpp
    test/unit/GemmGuidePost/HGemmGuidePost_Optimized.cpp
  )
endif()

add_executable(
  rocRollerTests
  ${rocRollerTests_SOURCES}
)

find_package(OpenMP)
target_compile_options(rocRollerTests PUBLIC ${EXTRA_COMPILE_OPTIONS})
target_include_directories(rocRollerTests PUBLIC
                           lib/include
                           )
target_link_options(rocRollerTests PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(
  rocRollerTests
  rocroller
  msgpackc-cxx
  gtest_main
  gmock
  openblas
  OpenMP::OpenMP_CXX
)
if(ROCROLLER_TESTS_USE_YAML_CPP)
  target_compile_definitions(rocRollerTests PUBLIC ROCROLLER_TESTS_USE_YAML_CPP)
  target_link_libraries(rocRollerTests yaml-cpp)
endif()

# ArchGen unit tests
add_executable(
  ArchGenTests
  test/unit/GPUArchitectureGeneratorTest.cpp
  lib/source/Error.cpp
  lib/source/GPUArchitecture.cpp
)
add_dependencies(ArchGenTests spdlog)
target_compile_features(ArchGenTests PUBLIC cxx_std_20)
target_compile_options(ArchGenTests PUBLIC  ${COROUTINES_COMPILE_OPTION} ${EXTRA_COMPILE_OPTIONS})
target_link_options(ArchGenTests PUBLIC ${EXTRA_LINK_OPTIONS})
target_include_directories(ArchGenTests PUBLIC
                           GPUArchitectureGenerator/include
                           lib/include
                           lib/include/rocRoller
                           )

target_include_directories(ArchGenTests SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/spdlog/include/
                           ${LLVM_INCLUDE_DIRS}
                           ${LLD_INCLUDE_DIRS})

if(ROCROLLER_USE_HIP)
  target_include_directories(ArchGenTests SYSTEM PUBLIC ${HIP_ROOT_DIR}/include)
  target_compile_definitions(ArchGenTests PUBLIC ROCROLLER_USE_HIP __HIP_PLATFORM_AMD__)
endif()

target_link_libraries(
  ArchGenTests PUBLIC
  ${llvm_libs}
  lldELF
  gtest_main
  gmock
  msgpackc-cxx
)

if(ROCROLLER_USE_YAML_CPP)
  target_compile_definitions(ArchGenTests PUBLIC ROCROLLER_USE_YAML_CPP)
  target_link_libraries(ArchGenTests PUBLIC yaml-cpp)
endif()
if(ROCROLLER_USE_LLVM)
  target_compile_definitions(ArchGenTests PUBLIC ROCROLLER_USE_LLVM)
endif()

string(LENGTH "${CMAKE_CURRENT_SOURCE_DIR}" PATH_PREFIX_LENGTH)
target_compile_definitions(rocroller PUBLIC ROCROLLER_PATH_PREFIX_LENGTH=${PATH_PREFIX_LENGTH})
target_compile_definitions(ArchGenTests PUBLIC ROCROLLER_PATH_PREFIX_LENGTH=${PATH_PREFIX_LENGTH})
target_compile_definitions(GPUArchitectureGenerator PUBLIC ROCROLLER_PATH_PREFIX_LENGTH=${PATH_PREFIX_LENGTH})

include(GoogleTest)
gtest_discover_tests(rocRollerTests
  XML_OUTPUT_DIR test_report
  TEST_FILTER "-*GPU_*"
)
gtest_discover_tests(rocRollerTests
  XML_OUTPUT_DIR test_report
  TEST_FILTER "*GPU_*"
  PROPERTIES "LABELS" "GPU"
)
gtest_discover_tests(ArchGenTests
  XML_OUTPUT_DIR test_report
)

# Add docs directory
add_subdirectory(docs)

# Add client directory
add_subdirectory(client)

# Add rrperf unit tests
add_test(NAME RRPerfPYTest
         COMMAND pytest-3 scripts
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(RRPerfPYTest
                     PROPERTIES ENVIRONMENT "ROCROLLER_BUILD_DIR=${CMAKE_BINARY_DIR}")
