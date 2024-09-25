
cmake_minimum_required(VERSION 3.16)

project(rocRoller VERSION 1.0)

option(ROCROLLER_ENABLE_TIMERS         "Enable rocRoller timer code." OFF)
option(ROCROLLER_TESTS_SKIP_GUIDEPOSTS "Skip compiling guidepost tests." ON)
option(ROCROLLER_TESTS_SKIP_SLOW       "Skip slow tests (currently only Python tests)" ON)
option(ROCROLLER_USE_HIP               "Use HIP within the rocRoller." ON)
option(SKIP_CPPCHECK                   "Skip CPPCHECK." ON)

option(CODE_ANALYSIS                    "When building, don't query for tests etc." OFF)
option(CODE_COVERAGE                    "Build with code coverage flags (clang only)" OFF)

option(BUILD_SHARED_LIBS                "Build using shared libraries" ON)

set(YAML_BACKEND "YAML_CPP" CACHE STRING "YAML backend used for serialization")
set_property(CACHE YAML_BACKEND PROPERTY STRINGS YAML_CPP LLVM)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(COROUTINES_COMPILE_OPTION )
  set(EXTRA_COMPILE_OPTIONS )
  set(EXTRA_LINK_OPTIONS -fuse-ld=lld)
else()
  set(COROUTINES_COMPILE_OPTION -fcoroutines)
  set(EXTRA_COMPILE_OPTIONS )
  set(EXTRA_LINK_OPTIONS )
endif()

if(ROCROLLER_ENABLE_TIMERS)
  add_definitions(-DROCROLLER_ENABLE_TIMERS)
  add_compile_options(-pg -fno-omit-frame-pointer)
endif()

if(CODE_COVERAGE)
  set(ROCROLLER_TESTS_SKIP_GUIDEPOSTS ON)
  set(ROCROLLER_TESTS_SKIP_SLOW OFF)
  set(EXTRA_COMPILE_OPTIONS -fprofile-instr-generate -fcoverage-mapping ${EXTRA_COMPILE_OPTIONS})
  set(EXTRA_LINK_OPTIONS -fprofile-instr-generate -fcoverage-mapping ${EXTRA_LINK_OPTIONS})
endif()

if(ROCROLLER_USE_HIP)
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip)
  find_package(hip)

  find_package(LLVM REQUIRED HINTS /opt/rocm/llvm/)

  llvm_map_components_to_libnames(llvm_libs all)
  find_package(LLD REQUIRED CONFIG HINTS ${LLVM_DIR}/../lld/)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
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

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.0 # or a later release
)

FetchContent_MakeAvailable(Catch2)
target_compile_options(Catch2 PRIVATE ${EXTRA_COMPILE_OPTIONS})
target_compile_options(Catch2WithMain PRIVATE ${EXTRA_COMPILE_OPTIONS})
target_link_options(Catch2 PRIVATE ${EXTRA_LINK_OPTIONS})
target_link_options(Catch2WithMain PRIVATE ${EXTRA_LINK_OPTIONS})

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

if(YAML_BACKEND STREQUAL "YAML_CPP")
  FetchContent_Declare(
    yaml-cpp
    URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.7.0.zip
  )
  FetchContent_MakeAvailable(yaml-cpp)

  target_compile_options(yaml-cpp PUBLIC ${EXTRA_COMPILE_OPTIONS})
  target_link_options(yaml-cpp PUBLIC ${EXTRA_LINK_OPTIONS})
  set_property(TARGET yaml-cpp PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET yaml-cpp APPEND PROPERTY COMPILE_OPTIONS "-Wno-shadow")

  add_definitions(-DROCROLLER_USE_YAML_CPP)
elseif(YAML_BACKEND STREQUAL "LLVM")
  add_definitions(-DROCROLLER_USE_LLVM)
endif()

target_compile_options(gtest PUBLIC ${EXTRA_COMPILE_OPTIONS})
target_compile_options(gmock PUBLIC ${EXTRA_COMPILE_OPTIONS})

if(SKIP_CPPCHECK)
  unset(CMAKE_CXX_CPPCHECK CACHE)
else()
  find_program(CMAKE_CXX_CPPCHECK NAMES cppcheck)
  if (CMAKE_CXX_CPPCHECK)
      list(
          APPEND CMAKE_CXX_CPPCHECK
              "--enable=warning"
              "--enable=performance"
              "--enable=portability"
              "--enable=style"
              "--error-exitcode=10"
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

set(ROCROLLER_DEFAULT_ARCHITECTURE_FILE_PATH "${CMAKE_BINARY_DIR}/source/rocRoller/GPUArchitecture_def.msgpack")
set(ROCROLLER_DEFAULT_ARCHITECTURE_CPP_FILE ${CMAKE_BINARY_DIR}/source/rocRoller/DefaultArchitectureFilePath.cpp)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lib/source/DefaultArchitectureFilePath.cpp.in ${ROCROLLER_DEFAULT_ARCHITECTURE_CPP_FILE})

# ArchitectureGenerator
add_executable(
  GPUArchitectureGenerator
  GPUArchitectureGenerator/source/GPUArchitectureGenerator.cpp

  lib/source/GPUArchitecture.cpp
  lib/source/GPUArchitectureLibrary.cpp

  lib/source/Assemblers/Assembler.cpp
  lib/source/Costs/Cost.cpp
  lib/source/DataTypes_F8_Utils.cpp
  lib/source/DataTypes_BF16_Utils.cpp
  lib/source/Error.cpp
  lib/source/Logging.cpp
  lib/source/Scheduler.cpp
  ${ROCROLLER_DEFAULT_ARCHITECTURE_CPP_FILE}
)
add_dependencies(GPUArchitectureGenerator spdlog)
target_compile_features(GPUArchitectureGenerator PUBLIC cxx_std_20)
target_compile_options(GPUArchitectureGenerator PUBLIC  ${COROUTINES_COMPILE_OPTION} ${EXTRA_COMPILE_OPTIONS})
target_link_options(GPUArchitectureGenerator PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(
  GPUArchitectureGenerator PUBLIC
  ${llvm_libs}
  msgpackc-cxx
)
target_include_directories(GPUArchitectureGenerator PUBLIC
                           ${CMAKE_CURRENT_SOURCE_DIR}/lib/include
                           GPUArchitectureGenerator/include
                           ${CMAKE_CURRENT_SOURCE_DIR}/lib/include/rocRoller
                           )

target_include_directories(GPUArchitectureGenerator SYSTEM PUBLIC
                           ${CMAKE_CURRENT_BINARY_DIR}/spdlog/include/
                           ${libdivide_SOURCE_DIR}/
                           ${LLVM_INCLUDE_DIRS}
                           ${LLD_INCLUDE_DIRS})

if(YAML_BACKEND STREQUAL "YAML_CPP")
  target_link_libraries(GPUArchitectureGenerator PUBLIC yaml-cpp)
endif()

if(ROCROLLER_USE_HIP)
  target_link_libraries(GPUArchitectureGenerator PUBLIC hip::host)
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

 lib/source/AssemblyKernel.cpp
 lib/source/AssemblyKernel_serialization.cpp
 lib/source/AssemblyKernel.cpp
 lib/source/AssemblyKernelArgument.cpp
 lib/source/AssemblyKernel_serialization.cpp
 lib/source/AssertOpKinds.cpp
 lib/source/CommandSolution.cpp
 lib/source/CommonSubexpressionElim.cpp
 lib/source/Context.cpp
 lib/source/DataTypes.cpp
 lib/source/DataTypes_F8_Utils.cpp
 lib/source/DataTypes_BF16_Utils.cpp
 lib/source/Error.cpp
 lib/source/ExecutableKernel.cpp
 lib/source/Expression.cpp
 lib/source/Expression_evaluate.cpp
 lib/source/Expression_generate.cpp
 lib/source/Expression_serialization.cpp
 lib/source/GPUArchitecture.cpp
 lib/source/GPUArchitectureLibrary.cpp
 lib/source/GPUInstructionInfo.cpp
 lib/source/KernelArguments.cpp
 lib/source/KernelOptions.cpp
 lib/source/LDSAllocator.cpp
 lib/source/Logging.cpp
 lib/source/Random.cpp
 lib/source/Register.cpp
 lib/source/ScheduledInstructions.cpp
 lib/source/Scheduler.cpp
 lib/source/Timer.cpp

 lib/source/Assemblers/Assembler.cpp
 lib/source/Assemblers/InProcessAssembler.cpp
 lib/source/Assemblers/InProcessAssembler_assemble.cpp
 lib/source/Assemblers/SubprocessAssembler.cpp

 lib/source/ExpressionTransformations/DataFlowTagPropagation.cpp
 lib/source/ExpressionTransformations/FastArithmetic.cpp
 lib/source/ExpressionTransformations/LowerExponential.cpp
 lib/source/ExpressionTransformations/FastDivision.cpp
 lib/source/ExpressionTransformations/FastMultiplication.cpp
 lib/source/ExpressionTransformations/FuseAssociative.cpp
 lib/source/ExpressionTransformations/FuseTernary.cpp
 lib/source/ExpressionTransformations/Identity.cpp
 lib/source/ExpressionTransformations/LaunchTimeSubExpressions.cpp
 lib/source/ExpressionTransformations/RestoreCommandArguments.cpp
 lib/source/ExpressionTransformations/Simplify.cpp

 lib/source/CodeGen/Arithmetic/Add.cpp
 lib/source/CodeGen/Arithmetic/AddShiftL.cpp
 lib/source/CodeGen/Arithmetic/ArithmeticGenerator.cpp
 lib/source/CodeGen/Arithmetic/ArithmeticShiftR.cpp
 lib/source/CodeGen/Arithmetic/BitwiseNegate.cpp
 lib/source/CodeGen/Arithmetic/BitwiseAnd.cpp
 lib/source/CodeGen/Arithmetic/BitwiseOr.cpp
 lib/source/CodeGen/Arithmetic/BitwiseXor.cpp
 lib/source/CodeGen/Arithmetic/Conditional.cpp
 lib/source/CodeGen/Arithmetic/Convert.cpp
 lib/source/CodeGen/Arithmetic/Divide.cpp
 lib/source/CodeGen/Arithmetic/Equal.cpp
 lib/source/CodeGen/Arithmetic/Exponential2.cpp
 lib/source/CodeGen/Arithmetic/NotEqual.cpp
 lib/source/CodeGen/Arithmetic/GreaterThan.cpp
 lib/source/CodeGen/Arithmetic/GreaterThanEqual.cpp
 lib/source/CodeGen/Arithmetic/LessThan.cpp
 lib/source/CodeGen/Arithmetic/LessThanEqual.cpp
 lib/source/CodeGen/Arithmetic/LogicalShiftR.cpp
 lib/source/CodeGen/Arithmetic/LogicalAnd.cpp
 lib/source/CodeGen/Arithmetic/LogicalOr.cpp
 lib/source/CodeGen/Arithmetic/LogicalNot.cpp
 lib/source/CodeGen/Arithmetic/MatrixMultiply.cpp
 lib/source/CodeGen/Arithmetic/Modulo.cpp
 lib/source/CodeGen/Arithmetic/Multiply.cpp
 lib/source/CodeGen/Arithmetic/MultiplyAdd.cpp
 lib/source/CodeGen/Arithmetic/MultiplyHigh.cpp
 lib/source/CodeGen/Arithmetic/Negate.cpp
 lib/source/CodeGen/Arithmetic/ShiftL.cpp
 lib/source/CodeGen/Arithmetic/ShiftLAdd.cpp
 lib/source/CodeGen/Arithmetic/Subtract.cpp
 lib/source/CodeGen/Arithmetic/Utility.cpp
 lib/source/CodeGen/ArgumentLoader.cpp
 lib/source/CodeGen/Buffer.cpp
 lib/source/CodeGen/CrashKernelGenerator.cpp
 lib/source/CodeGen/Instruction.cpp
 lib/source/CodeGen/LoadStoreTileGenerator.cpp
 lib/source/CodeGen/LowerFromKernelGraph.cpp
 lib/source/CodeGen/MemoryInstructions.cpp
 lib/source/CodeGen/WaitCount.cpp

 lib/source/Costs/Cost.cpp
 lib/source/Costs/LinearWeightedCost.cpp
 lib/source/Costs/MinNopsCost.cpp
 lib/source/Costs/NoneCost.cpp
 lib/source/Costs/UniformCost.cpp
 lib/source/Costs/WaitCntNopCost.cpp

 lib/source/KernelGraph/ControlFlowRWTracer.cpp
 lib/source/KernelGraph/ControlToCoordinateMapper.cpp
 lib/source/KernelGraph/GlobalConstraints.cpp
 lib/source/KernelGraph/KernelGraph.cpp
 lib/source/KernelGraph/KernelGraph_serialization.cpp
 lib/source/KernelGraph/LastRWTracer.cpp
 lib/source/KernelGraph/LowerFromCommand.cpp
 lib/source/KernelGraph/Reindexer.cpp
 lib/source/KernelGraph/ScopeManager.cpp
 lib/source/KernelGraph/Utils.cpp

 lib/source/KernelGraph/ControlGraph/ControlGraph.cpp
 lib/source/KernelGraph/ControlGraph/Operation.cpp

 lib/source/KernelGraph/CoordinateGraph/CoordinateGraph.cpp
 lib/source/KernelGraph/CoordinateGraph/Dimension.cpp
 lib/source/KernelGraph/CoordinateGraph/Transformer.cpp

 lib/source/KernelGraph/Transformations/AddComputeIndex.cpp
 lib/source/KernelGraph/Transformations/AddConvert.cpp
 lib/source/KernelGraph/Transformations/AddDeallocate.cpp
 lib/source/KernelGraph/Transformations/AddLDS.cpp
 lib/source/KernelGraph/Transformations/AddStreamK.cpp
 lib/source/KernelGraph/Transformations/CleanArguments.cpp
 lib/source/KernelGraph/Transformations/CleanLoops.cpp
 lib/source/KernelGraph/Transformations/ConnectWorkgroups.cpp
 lib/source/KernelGraph/Transformations/FuseExpressions.cpp
 lib/source/KernelGraph/Transformations/FuseLoops.cpp
 lib/source/KernelGraph/Transformations/InlineIncrements.cpp
 lib/source/KernelGraph/Transformations/KernelGraph_loop.cpp
 lib/source/KernelGraph/Transformations/LoopOverTileNumbers.cpp
 lib/source/KernelGraph/Transformations/LowerLinear.cpp
 lib/source/KernelGraph/Transformations/LowerTensorContraction.cpp
 lib/source/KernelGraph/Transformations/LowerTile.cpp
 lib/source/KernelGraph/Transformations/OrderEpilogueBlocks.cpp
 lib/source/KernelGraph/Transformations/OrderMemory.cpp
 lib/source/KernelGraph/Transformations/Simplify.cpp
 lib/source/KernelGraph/Transformations/UnrollLoops.cpp
 lib/source/KernelGraph/Transformations/UpdateParameters.cpp
 lib/source/KernelGraph/Transformations/ConstantPropagation.cpp

 lib/source/Observers/ACCVGPRReadWrite.cpp
 lib/source/Observers/ACCVGPRWriteWrite.cpp
 lib/source/Observers/CMPXWriteExec.cpp
 lib/source/Observers/DGEMM16x16x4Write.cpp
 lib/source/Observers/DGEMM4x4x4Write.cpp
 lib/source/Observers/DLWrite.cpp
 lib/source/Observers/ObserverCreation.cpp
 lib/source/Observers/OPSEL94x.cpp
 lib/source/Observers/RegisterLivenessObserver.cpp
 lib/source/Observers/SupportedInstructionObserver.cpp
 lib/source/Observers/VALUTransWrite94x.cpp
 lib/source/Observers/VALUWrite.cpp
 lib/source/Observers/VALUWriteReadlane94x.cpp
 lib/source/Observers/VALUWriteSGPRVCC94x.cpp
 lib/source/Observers/VALUWriteSGPRVMEM.cpp
 lib/source/Observers/VALUWriteVCCVDIVFMAS.cpp
 lib/source/Observers/VCMPXWrite94x.cpp
 lib/source/Observers/WaitcntObserver.cpp
 lib/source/Observers/XDLReadSrcC908.cpp
 lib/source/Observers/XDLReadSrcC94x.cpp
 lib/source/Observers/XDLReadSrcC90a.cpp
 lib/source/Observers/XDLWrite908.cpp
 lib/source/Observers/XDLWrite94x.cpp
 lib/source/Observers/XDLWrite90a.cpp
 lib/source/Observers/MFMAObserver.cpp

 lib/source/Operations/Operation.cpp
 lib/source/Operations/TensorScalar.cpp
 lib/source/Operations/BlockScale.cpp

 lib/source/Schedulers/CooperativeScheduler.cpp
 lib/source/Schedulers/PriorityScheduler.cpp
 lib/source/Schedulers/RandomScheduler.cpp
 lib/source/Schedulers/RoundRobinScheduler.cpp
 lib/source/Schedulers/SequentialScheduler.cpp

 ${ROCROLLER_DEFAULT_ARCHITECTURE_CPP_FILE}
)

if(ROCROLLER_USE_HIP)
  set(rocroller_src
   ${rocroller_src}
   lib/source/HIPTimer.cpp
  )
endif()

add_library(rocroller
 ${rocroller_src}
)

add_dependencies(rocroller spdlog)

# Users of rocroller should link with rocroller_interface for GPUArchitecture_def dependency
add_library(rocroller_interface INTERFACE)
target_link_libraries(rocroller_interface INTERFACE
  -Wl,--whole-archive
  rocroller
  -Wl,--no-whole-archive
)
add_dependencies(rocroller_interface GPUArchitecture_def)

target_compile_features(rocroller PUBLIC cxx_std_20)
target_compile_options(rocroller PUBLIC ${COROUTINES_COMPILE_OPTION} ${EXTRA_COMPILE_OPTIONS})
target_link_options(rocroller PUBLIC ${EXTRA_LINK_OPTIONS})
#target_compile_options(rocroller PUBLIC -fconcepts-diagnostics-depth=2)
target_include_directories(rocroller PUBLIC
                           ${CMAKE_CURRENT_SOURCE_DIR}/lib/include
                           ${CMAKE_CURRENT_SOURCE_DIR}/lib/include/rocRoller
                           )

# Using SYSTEM include paths for third party libraries speeds up cppcheck a lot
target_include_directories(rocroller SYSTEM PUBLIC
                           ${libdivide_SOURCE_DIR}/
                           ${CMAKE_CURRENT_BINARY_DIR}/spdlog/include/)
# target_link_libraries(rocroller
#   PRIVATE spdlog::spdlog
#   PUBLIC fmt::fmt
# )

if(YAML_BACKEND STREQUAL "YAML_CPP")
  target_link_libraries(rocroller PUBLIC yaml-cpp)
endif()

target_link_libraries(rocroller PRIVATE ${llvm_libs} lldELF PUBLIC msgpackc-cxx)
target_include_directories(rocroller SYSTEM PUBLIC
                            ${LLVM_INCLUDE_DIRS} ${LLD_INCLUDE_DIRS})

if(ROCROLLER_USE_HIP)
  target_link_libraries(rocroller PUBLIC hip::host)
  # target_link_libraries(rocroller PUBLIC amdhip64)
  target_compile_definitions(rocroller PUBLIC ROCROLLER_USE_HIP __HIP_PLATFORM_AMD__)
endif()

# Component stand-alone test
add_executable(ComponentTest test/standalone/ComponentTest.cpp)
target_compile_options(ComponentTest PUBLIC ${EXTRA_COMPILE_OPTIONS})
target_link_options(ComponentTest PUBLIC ${EXTRA_LINK_OPTIONS} -pthread)
target_link_libraries(ComponentTest
  rocroller_interface
  msgpackc-cxx
)
target_include_directories(ComponentTest PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}/lib/include
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

add_library(common_test_utilities
  test/common/Utilities.cpp
)

target_compile_features(common_test_utilities PUBLIC cxx_std_20)
target_compile_options(common_test_utilities PUBLIC ${COROUTINES_COMPILE_OPTION} ${EXTRA_COMPILE_OPTIONS})
target_link_options(common_test_utilities PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(common_test_utilities
  rocroller_interface
  openblas
)
#target_compile_options(rocroller PUBLIC -fconcepts-diagnostics-depth=2)
target_include_directories(common_test_utilities PUBLIC
                           ${CMAKE_CURRENT_SOURCE_DIR}/test/common/
                           )


set(
  rocRollerTests_SOURCES
  test/unit/ContextFixture.cpp
  test/unit/GenericContextFixture.cpp
  test/unit/GPUContextFixture.cpp

  test/unit/AddStreamKTest.cpp
  test/unit/AllocatingObserverTest.cpp
  test/unit/ArgumentLoaderTest.cpp
  test/unit/ArgumentValueTest.cpp
  test/unit/ArithmeticTest.cpp
  test/unit/AssemblerTest.cpp
  test/unit/AssertTest.cpp
  test/unit/BranchGeneratorTest.cpp
  test/unit/BufferDescriptorTest.cpp
  test/unit/CommandTest.cpp
  test/unit/CommonSubexpressionElimTest.cpp
  test/unit/ComponentTest.cpp
  test/unit/ControlGraphTest.cpp
  test/unit/CoordinateGraphTest.cpp
  test/unit/CopyGeneratorTest.cpp
  test/unit/DataTypesTest.cpp
  test/unit/DependencyTest.cpp
  test/unit/ErrorTest.cpp
  test/unit/ExpressionTransformationTest.cpp
  test/unit/FastDivisionTest.cpp
  test/unit/LowerExponentialTest.cpp
  test/unit/FastMultiplicationTest.cpp
  test/unit/FileWritingObserverTest.cpp
  test/unit/GPUArchitectureTest.cpp
  test/unit/GPUInstructionInfoTest.cpp
  test/unit/GEMMTest.cpp
  test/unit/GEMMFusion.cpp
  test/unit/GeneratorTest.cpp
  test/unit/HalfPrecisionTest.cpp
  test/unit/ConversionTest.cpp
  test/unit/F8Test.cpp
  test/unit/HazardObserverTest.cpp
  test/unit/HypergraphTest.cpp
  test/unit/InstructionTest.cpp
  test/unit/KernelGraphTest.cpp
  test/unit/KernelGraphTest/ConstantPropagation.cpp
  test/unit/KernelGraphTest/KernelGraphUtils.cpp
  test/unit/KernelGraphTest/Simplify.cpp
  test/unit/KernelGraphTest/UpdateParameters.cpp
  test/unit/KernelIndexTest.cpp
  test/unit/KernelOptionsTest.cpp
  test/unit/KernelTest.cpp
  test/unit/LabelTest.cpp
  test/unit/LDSTest.cpp
  test/unit/LockTest.cpp
  test/unit/MemoryInstructionsTest.cpp
  test/unit/MatrixMultiplyTest.cpp
  test/unit/MetaObserverTest.cpp
  test/unit/MFMAHazardObserverTest.cpp
  test/unit/MFMAUnitObserverTest.cpp
  test/unit/MixedArithmeticTest.cpp
  test/unit/RandomTest.cpp
  test/unit/RegisterAllocatorTest.cpp
  test/unit/RegisterTest.cpp
  test/unit/RegisterLivenessObserverTest.cpp
  test/unit/CostTest.cpp
  test/unit/SchedulerTest.cpp
  test/unit/ScopeTest.cpp
  test/unit/SettingsTest.cpp
  test/unit/SourceMatcherTest.cpp
  test/unit/SupportedInstructionObserverTest.cpp
  test/unit/TileTransposeAdd.cpp
  test/unit/TimerTest.cpp
  test/unit/UtilsTest.cpp
  test/unit/VariantTest.cpp
  test/unit/VectorAddBenchmark.cpp
  test/unit/WaitCountObserverTest.cpp
  test/unit/WaitCountTest.cpp
  test/unit/OperationTagTest.cpp
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
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/include
)
target_link_options(rocRollerTests PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(
  rocRollerTests
  rocroller_interface
  common_test_utilities
  msgpackc-cxx
  gtest_main
  gmock
  openblas
  OpenMP::OpenMP_CXX
)

if(ROCROLLER_USE_HIP)
  target_link_libraries(rocRollerTests hip::host)
  target_compile_definitions(rocRollerTests PUBLIC ROCROLLER_USE_HIP __HIP_PLATFORM_AMD__)
endif()

if(YAML_BACKEND STREQUAL "YAML_CPP")
  target_compile_definitions(rocRollerTests PUBLIC ROCROLLER_TESTS_USE_YAML_CPP)
endif()


set(
  rocRollerTests_catch_SOURCES

  test/catch/CustomAssertions.cpp
  test/catch/TestKernels.cpp

  test/catch/BinaryExpressionTest.cpp
  test/catch/CustomAssertionTest.cpp
  test/catch/CustomMatcherTest.cpp
  test/catch/ExpressionTest.cpp
  test/catch/KernelArgumentsTest.cpp
  test/catch/ScalarExpressionTest.cpp
  test/catch/TernaryExpressionTest.cpp
  test/catch/TestContextTest.cpp
  test/catch/UtilsTest.cpp
)

add_executable(rocRollerTests_catch ${rocRollerTests_catch_SOURCES})
target_compile_options(rocRollerTests_catch PUBLIC ${EXTRA_COMPILE_OPTIONS})

target_include_directories(rocRollerTests_catch PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}/lib/include
                           )
target_link_options(rocRollerTests_catch PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(
  rocRollerTests_catch
  rocroller_interface
  common_test_utilities
  msgpackc-cxx
  Catch2::Catch2WithMain
  openblas
  OpenMP::OpenMP_CXX
)

if(ROCROLLER_USE_HIP)
  target_link_libraries(rocRollerTests_catch hip::host)
  target_compile_definitions(rocRollerTests_catch PUBLIC ROCROLLER_USE_HIP __HIP_PLATFORM_AMD__)
endif()

if(YAML_BACKEND STREQUAL "YAML_CPP")
  target_compile_definitions(rocRollerTests_catch PUBLIC ROCROLLER_TESTS_USE_YAML_CPP)
endif()

# ArchGen unit tests
add_executable(
  ArchGenTests
  test/unit/GPUArchitectureGeneratorTest.cpp
)
add_dependencies(ArchGenTests spdlog)
target_compile_features(ArchGenTests PUBLIC cxx_std_20)
target_compile_options(ArchGenTests PUBLIC  ${COROUTINES_COMPILE_OPTION} ${EXTRA_COMPILE_OPTIONS})
target_link_options(ArchGenTests PUBLIC ${EXTRA_LINK_OPTIONS})
target_include_directories(ArchGenTests PUBLIC
                           GPUArchitectureGenerator/include
                           ${CMAKE_CURRENT_SOURCE_DIR}/lib/include
                           ${CMAKE_CURRENT_SOURCE_DIR}/lib/include/rocRoller
                           )

target_include_directories(ArchGenTests SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/spdlog/include/
                           ${LLVM_INCLUDE_DIRS}
                           ${LLD_INCLUDE_DIRS})

if(ROCROLLER_USE_HIP)
  target_link_libraries(ArchGenTests PUBLIC hip::host)
  target_compile_definitions(ArchGenTests PUBLIC ROCROLLER_USE_HIP __HIP_PLATFORM_AMD__)
endif()

target_link_libraries(
  ArchGenTests PUBLIC
  ${llvm_libs}
  common_test_utilities
  rocroller_interface
  gtest_main
  gmock
  msgpackc-cxx
)

string(LENGTH "${CMAKE_CURRENT_SOURCE_DIR}" PATH_PREFIX_LENGTH)
target_compile_definitions(rocroller PUBLIC ROCROLLER_PATH_PREFIX_LENGTH=${PATH_PREFIX_LENGTH})
target_compile_definitions(ArchGenTests PUBLIC ROCROLLER_PATH_PREFIX_LENGTH=${PATH_PREFIX_LENGTH})
target_compile_definitions(GPUArchitectureGenerator PUBLIC ROCROLLER_PATH_PREFIX_LENGTH=${PATH_PREFIX_LENGTH})

# Add docs directory
add_subdirectory(docs)

# Add client directory
add_subdirectory(client)

if (NOT CODE_ANALYSIS)
  set(TEST_REPORT_DIR "${CMAKE_BINARY_DIR}/test_report")

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

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
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
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  set_tests_properties(PythonLint
                       PROPERTIES ENVIRONMENT "ROCROLLER_BUILD_DIR=${CMAKE_BINARY_DIR}"
  )

  if (ROCROLLER_TESTS_SKIP_SLOW)
    set(PYTHON_TEST_SKIP "-m not slow")
  endif()

  # Add rrperf unit tests
  add_test(NAME PythonTest
           COMMAND pytest -s --cov-report html:python_cov_html --cov=rrperf scripts ${PYTHON_TEST_SKIP} --junit-xml=${TEST_REPORT_DIR}/python_tests.xml
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  set_tests_properties(PythonTest
                       PROPERTIES ENVIRONMENT "ROCROLLER_BUILD_DIR=${CMAKE_BINARY_DIR};PYTHONPATH=${CMAKE_SOURCE_DIR}/scripts/lib"
  )


  add_test(NAME GraupelInstall
           COMMAND npm install
           WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/utils/graupel")

  set_tests_properties(GraupelInstall PROPERTIES FIXTURES_SETUP Graupel)

  add_test(NAME GraupelTests
           COMMAND npm test
           WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/utils/graupel")

  set_tests_properties(GraupelTests PROPERTIES
                       ENVIRONMENT "JEST_JUNIT_OUTPUT_FILE=${TEST_REPORT_DIR}/graupel_tests.xml"
                       FIXTURES_REQUIRED Graupel)
endif()
