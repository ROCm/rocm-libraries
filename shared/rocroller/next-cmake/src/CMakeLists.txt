################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the Software), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################


set(_CMAKE_CURRENT_SOURCE_DIR "${ROCROLLER_LIB_DIR}/source")
target_sources(rocroller
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/AssemblyKernelArgument.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AssemblyKernel.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AssemblyKernel_serialization.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AssertOpKinds.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Command_serialization.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/CommandSolution.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/CommonSubexpressionElim.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Context.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/DataTypes_BF16_Utils.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/DataTypes.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/DataTypes_F6_Utils.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/DataTypes_F8_Utils.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/DataTypes_Utils.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Error.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ExecutableKernel.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Expression.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Expression_evaluate.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Expression_generate.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Expression_resultType.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Expression_serialization.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Expression_toString.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/GPUArchitecture.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/GPUArchitectureLibrary.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/GPUInstructionInfo.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/HIPTimer.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/KernelArguments.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/KernelOptions.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LDSAllocator.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Logging.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Random.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Register.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ReplaceKernelArgs.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ScheduledInstructions.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Scheduler.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Timer.cpp"
)

add_subdirectory(Assemblers)
add_subdirectory(Schedulers)
add_subdirectory(CodeGen)
add_subdirectory(Costs)
add_subdirectory(ExpressionTransformations)
add_subdirectory(Observers)
add_subdirectory(Utilities)
add_subdirectory(Operations)
add_subdirectory(KernelGraph)

add_library(rocroller-no-rtti
    OBJECT
        "${_CMAKE_CURRENT_SOURCE_DIR}/Utilities/ReadELF.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Assemblers/InProcessAssembler.cpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ExpressionTransformations/FastDivision.cpp"
)
if(ROCROLLER_ENABLE_LLD)
    target_compile_definitions(rocroller-no-rtti PRIVATE ROCROLLER_ENABLE_LLD)
    target_link_libraries(rocroller-no-rtti PRIVATE lldELF)
endif()
target_compile_options(rocroller-no-rtti PRIVATE -fno-rtti)
target_compile_features(rocroller-no-rtti PRIVATE cxx_std_20)
target_compile_definitions(rocroller-no-rtti
    PUBLIC
        ROCROLLER_USE_HIP
        __HIP_PLATFORM_AMD__
)
target_include_directories(rocroller-no-rtti
    PRIVATE
        "${LLVM_INCLUDE_DIRS}"
        "${ROCROLLER_LIB_DIR}/include"
)
target_link_libraries(rocroller-no-rtti
    PRIVATE
        hip::host
        $<BUILD_INTERFACE:spdlog::spdlog_header_only>
        fmt::fmt-header-only
        Boost::headers
)

target_link_libraries(rocroller PRIVATE $<BUILD_INTERFACE:rocroller-no-rtti>)

set_target_properties(
    rocroller-no-rtti PROPERTIES POSITION_INDEPENDENT_CODE ON
)
