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


set(_CMAKE_CURRENT_SOURCE_DIR "${ROCROLLER_LIB_DIR}/include/rocRoller")
target_sources(rocroller
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/AssemblyKernelArgument_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AssemblyKernelArgument.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AssemblyKernel_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AssemblyKernel.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AssemblyKernel_impl.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AssertOpKinds_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/AssertOpKinds.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/CommandSolution_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/CommandSolution.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/CommonSubexpressionElim.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Context_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Context.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Context_impl.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ExecutableKernel_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ExecutableKernel.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ExecutableKernel_impl.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Expression_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Expression.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Expression_impl.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ExpressionTransformations.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/KernelArguments_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/KernelArguments.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/KernelArguments_impl.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/KernelOptions.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ReplaceKernelArgs.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ScheduledInstructions_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ScheduledInstructions.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/TensorDescriptor.hpp"
)

add_subdirectory(Assemblers)
add_subdirectory(InstructionValues)
add_subdirectory(GPUArchitecture)
add_subdirectory(CodeGen)
add_subdirectory(Graph)
add_subdirectory(Scheduling)
add_subdirectory(DataTypes)
add_subdirectory(Utilities)
add_subdirectory(Serialization)
add_subdirectory(Operations)
add_subdirectory(KernelGraph)
