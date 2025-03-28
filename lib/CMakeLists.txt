
################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################


# rocRoller library

# Print out compiler flags for viewing/debug
if(BUILD_VERBOSE)
    set(VERBOSE_VARIABLES
        ROCROLLER_VERSION
        CMAKE_BUILD_TYPE
        CMAKE_CXX_COMPILER
        BUILD_SHARED_LIBS
        CMAKE_INSTALL_PREFIX
        CMAKE_MODULE_PATH
        CMAKE_PREFIX_PATH
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}
    )

    message(STATUS "Build variable information: ")
    foreach(VAR IN LISTS VERBOSE_VARIABLES)
        message(STATUS "\t==>${VAR}: ${${VAR}}")
    endforeach()
endif()

set(rocroller_src
    source/AssemblyKernel_serialization.cpp
    source/AssemblyKernel.cpp
    source/AssemblyKernelArgument.cpp
    source/AssertOpKinds.cpp
    source/Command_serialization.cpp
    source/CommandSolution.cpp
    source/CommonSubexpressionElim.cpp
    source/Context.cpp
    source/DataTypes_BF16_Utils.cpp
    source/DataTypes_F6_Utils.cpp
    source/DataTypes_F8_Utils.cpp
    source/DataTypes_Utils.cpp
    source/DataTypes.cpp
    source/Error.cpp
    source/ExecutableKernel.cpp
    source/Expression_evaluate.cpp
    source/Expression_generate.cpp
    source/Expression_resultType.cpp
    source/Expression_serialization.cpp
    source/Expression_toString.cpp
    source/Expression.cpp
    source/GPUArchitecture.cpp
    source/GPUArchitectureLibrary.cpp
    source/GPUInstructionInfo.cpp
    source/KernelArguments.cpp
    source/KernelOptions.cpp
    source/LDSAllocator.cpp
    source/Logging.cpp
    source/Random.cpp
    source/Register.cpp
    source/ScheduledInstructions.cpp
    source/Scheduler.cpp
    source/Timer.cpp

    source/Assemblers/Assembler.cpp
    source/Assemblers/InProcessAssembler.cpp
    source/Assemblers/InProcessAssembler_assemble.cpp
    source/Assemblers/SubprocessAssembler.cpp

    source/ExpressionTransformations/CombineShifts.cpp
    source/ExpressionTransformations/DataFlowTagPropagation.cpp
    source/ExpressionTransformations/FastArithmetic.cpp
    source/ExpressionTransformations/FastDivision.cpp
    source/ExpressionTransformations/FastMultiplication.cpp
    source/ExpressionTransformations/FuseAssociative.cpp
    source/ExpressionTransformations/FuseTernary.cpp
    source/ExpressionTransformations/Identity.cpp
    source/ExpressionTransformations/LaunchTimeSubExpressions.cpp
    source/ExpressionTransformations/LowerBitfieldValues.cpp
    source/ExpressionTransformations/LowerExponential.cpp
    source/ExpressionTransformations/LowerPRNG.cpp
    source/ExpressionTransformations/RestoreCommandArguments.cpp
    source/ExpressionTransformations/Simplify.cpp

    source/CodeGen/AddInstruction.cpp
    source/CodeGen/ArgumentLoader.cpp
    source/CodeGen/Arithmetic/Add.cpp
    source/CodeGen/Arithmetic/AddShiftL.cpp
    source/CodeGen/Arithmetic/ArithmeticGenerator.cpp
    source/CodeGen/Arithmetic/ArithmeticShiftR.cpp
    source/CodeGen/Arithmetic/BitFieldExtract.cpp
    source/CodeGen/Arithmetic/BitwiseAnd.cpp
    source/CodeGen/Arithmetic/BitwiseNegate.cpp
    source/CodeGen/Arithmetic/BitwiseOr.cpp
    source/CodeGen/Arithmetic/BitwiseXor.cpp
    source/CodeGen/Arithmetic/Conditional.cpp
    source/CodeGen/Arithmetic/Convert.cpp
    source/CodeGen/Arithmetic/Divide.cpp
    source/CodeGen/Arithmetic/Equal.cpp
    source/CodeGen/Arithmetic/Exponential2.cpp
    source/CodeGen/Arithmetic/GreaterThan.cpp
    source/CodeGen/Arithmetic/GreaterThanEqual.cpp
    source/CodeGen/Arithmetic/LessThan.cpp
    source/CodeGen/Arithmetic/LessThanEqual.cpp
    source/CodeGen/Arithmetic/LogicalAnd.cpp
    source/CodeGen/Arithmetic/LogicalNot.cpp
    source/CodeGen/Arithmetic/LogicalOr.cpp
    source/CodeGen/Arithmetic/LogicalShiftR.cpp
    source/CodeGen/Arithmetic/MatrixMultiply.cpp
    source/CodeGen/Arithmetic/Modulo.cpp
    source/CodeGen/Arithmetic/Multiply.cpp
    source/CodeGen/Arithmetic/MultiplyAdd.cpp
    source/CodeGen/Arithmetic/MultiplyHigh.cpp
    source/CodeGen/Arithmetic/Negate.cpp
    source/CodeGen/Arithmetic/NotEqual.cpp
    source/CodeGen/Arithmetic/RandomNumber.cpp
    source/CodeGen/Arithmetic/ScaledMatrixMultiply.cpp
    source/CodeGen/Arithmetic/ShiftL.cpp
    source/CodeGen/Arithmetic/ShiftLAdd.cpp
    source/CodeGen/Arithmetic/Subtract.cpp
    source/CodeGen/Arithmetic/Utility.cpp
    source/CodeGen/Buffer.cpp
    source/CodeGen/CrashKernelGenerator.cpp
    source/CodeGen/Instruction.cpp
    source/CodeGen/LoadStoreTileGenerator.cpp
    source/CodeGen/LowerFromKernelGraph.cpp
    source/CodeGen/MemoryInstructions.cpp
    source/CodeGen/SubInstruction.cpp
    source/CodeGen/WaitCount.cpp

    source/Costs/Cost.cpp
    source/Costs/LinearWeightedCost.cpp
    source/Costs/MinNopsCost.cpp
    source/Costs/NoneCost.cpp
    source/Costs/UniformCost.cpp
    source/Costs/WaitCntNopCost.cpp

    source/KernelGraph/ControlFlowRWTracer.cpp
    source/KernelGraph/ControlToCoordinateMapper.cpp
    source/KernelGraph/GlobalConstraints.cpp
    source/KernelGraph/KernelGraph.cpp
    source/KernelGraph/KernelGraph_serialization.cpp
    source/KernelGraph/LastRWTracer.cpp
    source/KernelGraph/LowerFromCommand.cpp
    source/KernelGraph/RegisterTagManager.cpp
    source/KernelGraph/Reindexer.cpp
    source/KernelGraph/ScopeManager.cpp
    source/KernelGraph/Utils.cpp

    source/KernelGraph/ControlGraph/ControlGraph.cpp
    source/KernelGraph/ControlGraph/Operation.cpp

    source/KernelGraph/CoordinateGraph/CoordinateGraph.cpp
    source/KernelGraph/CoordinateGraph/Dimension.cpp
    source/KernelGraph/CoordinateGraph/Transformer.cpp

    source/KernelGraph/Transformations/AddComputeIndex.cpp
    source/KernelGraph/Transformations/AddConvert.cpp
    source/KernelGraph/Transformations/AddDeallocate.cpp
    source/KernelGraph/Transformations/AddDirect2LDS.cpp
    source/KernelGraph/Transformations/AddF6LDSPadding.cpp
    source/KernelGraph/Transformations/AddLDS.cpp
    source/KernelGraph/Transformations/AddPRNG.cpp
    source/KernelGraph/Transformations/AddStreamK.cpp
    source/KernelGraph/Transformations/CleanArguments.cpp
    source/KernelGraph/Transformations/CleanLoops.cpp
    source/KernelGraph/Transformations/ConnectWorkgroups.cpp
    source/KernelGraph/Transformations/ConstantPropagation.cpp
    source/KernelGraph/Transformations/FuseExpressions.cpp
    source/KernelGraph/Transformations/FuseLoops.cpp
    source/KernelGraph/Transformations/IdentifyParallelDimensions.cpp
    source/KernelGraph/Transformations/InlineIncrements.cpp
    source/KernelGraph/Transformations/KernelGraph_loop.cpp
    source/KernelGraph/Transformations/LoadPacked.cpp
    source/KernelGraph/Transformations/LoopOverTileNumbers.cpp
    source/KernelGraph/Transformations/LowerLinear.cpp
    source/KernelGraph/Transformations/LowerTensorContraction.cpp
    source/KernelGraph/Transformations/LowerTile.cpp
    source/KernelGraph/Transformations/OrderEpilogueBlocks.cpp
    source/KernelGraph/Transformations/OrderMemory.cpp
    source/KernelGraph/Transformations/RemoveDuplicates.cpp
    source/KernelGraph/Transformations/Simplify.cpp
    source/KernelGraph/Transformations/SwizzleScale.cpp
    source/KernelGraph/Transformations/UnrollLoops.cpp
    source/KernelGraph/Transformations/UpdateParameters.cpp

    source/Observers/ACCVGPRReadWrite.cpp
    source/Observers/ACCVGPRWriteWrite.cpp
    source/Observers/BufferStoreDwordXXRead.cpp
    source/Observers/CMPXWriteExec.cpp
    source/Observers/DGEMM16x16x4Write.cpp
    source/Observers/DGEMM4x4x4Write.cpp
    source/Observers/DLWrite.cpp
    source/Observers/MFMAObserver.cpp
    source/Observers/OPSEL94x.cpp
    source/Observers/ObserverCreation.cpp
    source/Observers/RegisterLivenessObserver.cpp
    source/Observers/SupportedInstructionObserver.cpp
    source/Observers/VALUTransWrite94x.cpp
    source/Observers/VALUWrite.cpp
    source/Observers/VALUWriteReadlane94x.cpp
    source/Observers/VALUWriteSGPRVCC94x.cpp
    source/Observers/VALUWriteSGPRVMEM.cpp
    source/Observers/VALUWriteVCCVDIVFMAS.cpp
    source/Observers/VCMPXWrite94x.cpp
    source/Observers/WaitcntObserver.cpp
    source/Observers/WMMAObserver.cpp
    source/Observers/WMMAReadSrcD.cpp
    source/Observers/WMMAWriteSrcD.cpp
    source/Observers/WMMAWrite.cpp
    source/Observers/XDLReadSrcC908.cpp
    source/Observers/XDLReadSrcC90a.cpp
    source/Observers/XDLReadSrcC94x.cpp
    source/Observers/XDLWrite908.cpp
    source/Observers/XDLWrite90a.cpp
    source/Observers/XDLWrite94x.cpp

    source/Operations/BlockScale.cpp
    source/Operations/Command.cpp
    source/Operations/Operation.cpp
    source/Operations/RandomNumberGenerator.cpp
    source/Operations/TensorScalar.cpp
    source/Operations/T_LoadStore.cpp

    source/Schedulers/CooperativeScheduler.cpp
    source/Schedulers/PriorityScheduler.cpp
    source/Schedulers/RandomScheduler.cpp
    source/Schedulers/RoundRobinScheduler.cpp
    source/Schedulers/SequentialScheduler.cpp

    source/Utilities/ReadELF.cpp
    source/Utilities/Settings.cpp
    source/Utilities/Utils.cpp

    ${ROCROLLER_GIT_VERSION_CPP_FILE}
)

if(ROCROLLER_USE_HIP)
    list(APPEND rocroller_src source/HIPTimer.cpp)
endif()

add_library(rocroller ${rocroller_src})
add_library(roc::rocroller ALIAS rocroller)
set_target_properties(
    rocroller PROPERTIES
        VERSION ${ROCROLLER_VERSION}
        SOVERSION ${ROCROLLER_SOVERSION}
)

target_link_libraries(rocroller
    PRIVATE
    spdlog::spdlog_header_only
    PUBLIC
    fmt::fmt-header-only
)
if(ROCROLLER_EMBED_ARCH_DEF)
    target_link_libraries(rocroller PRIVATE $<BUILD_INTERFACE:rocRoller::GPUArchitecture_def>)
else()
    target_compile_definitions(rocroller PRIVATE ROCROLLER_NO_EMBED_ARCH_DEF)
endif()

# Users of rocroller should link with rocroller_interface for GPUArchitecture_def dependency
add_library(rocroller_interface INTERFACE)
add_library(roc::rocroller_interface ALIAS rocroller_interface)
target_link_libraries(
    rocroller_interface
    INTERFACE
        -Wl,--whole-archive # Important for RocRoller Component functionality
        rocroller
        -Wl,--no-whole-archive
)
add_dependencies(rocroller_interface GPUArchitecture_def)

target_compile_options(
    rocroller PUBLIC ${COROUTINES_COMPILE_OPTION} ${EXTRA_COMPILE_OPTIONS}
)
target_link_options(rocroller PUBLIC ${EXTRA_LINK_OPTIONS})

#target_compile_options(rocroller PUBLIC -fconcepts-diagnostics-depth=2)
target_include_directories(
    rocroller PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                     $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
)

# Header only, not exposed in any public headers, system package not required
rocroller_add_dependency(libdivide VERSION 5.1)
target_link_libraries(rocroller PRIVATE $<BUILD_INTERFACE:libdivide>)

rocroller_add_dependency(boost)
target_link_libraries(rocroller PUBLIC $<BUILD_INTERFACE:Boost::multi_index>)

# LLVM has RTTI disabled by default and ROCm's LLVM also has RTTI disabled.
# RTTI is enabled for rocRoller, except for the compilation units below.
# If RTTI is also desired for this CUs, then the options are:
#    - Have ROCm ship with LLVM with RTTI enabled;
#    - Use a pre-build LLVM library with RTTI enabled;
#    - Build LLVM (or the required components) from source as pre-build step;
set_source_files_properties(
    source/Utilities/ReadELF.cpp
    source/Assemblers/InProcessAssembler.cpp
    source/ExpressionTransformations/FastDivision.cpp
    PROPERTIES COMPILE_FLAGS -fno-rtti
)

if(YAML_BACKEND STREQUAL "YAML_CPP")
    target_link_libraries(rocroller PUBLIC yaml-cpp)
    set(yaml_export_dependency PACKAGE yaml-cpp VERSION 0.7.0)
endif()

target_link_libraries(
    rocroller
    PRIVATE ${llvm_libs} ${llvm_yaml_libs} lldELF
    PUBLIC msgpackc-cxx
)
target_include_directories(rocroller SYSTEM PUBLIC ${LLD_INCLUDE_DIRS})

if(ROCROLLER_USE_HIP)
    target_link_libraries(rocroller PUBLIC hip::host)
    # target_link_libraries(rocroller PUBLIC amdhip64)
    target_compile_definitions(
        rocroller PUBLIC ROCROLLER_USE_HIP __HIP_PLATFORM_AMD__
    )
endif()

rocm_install(
    TARGETS rocroller rocroller_interface
    EXPORT rocroller_targets
    INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include
)

rocm_export_targets(
    TARGETS roc::rocroller roc::rocroller_interface
    NAMESPACE roc::
    EXPORT rocroller_targets
    DEPENDS
        PACKAGE msgpack VERSION 3.3.0
        ${yaml_export_dependency}
)
