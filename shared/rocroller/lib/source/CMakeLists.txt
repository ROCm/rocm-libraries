# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(rocroller
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/AssemblyKernel.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AssemblyKernelArgument.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AssemblyKernel_serialization.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AssertOpKinds.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CommandSolution.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Command_serialization.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CommonSubexpressionElim.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Context.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DataTypes.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DataTypes_BF16_Utils.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DataTypes_F6_Utils.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DataTypes_F8_Utils.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DataTypes_Utils.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Error.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ExecutableKernel.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Expression.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Expression_evaluate.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Expression_generate.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Expression_referencedKernelArguments.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Expression_resultType.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Expression_serialization.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Expression_toString.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/GPUArchitecture.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/GPUArchitectureLibrary.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/GPUInstructionInfo.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/HIPTimer.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/KernelArguments.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/KernelOptions.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LDSAllocator.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Logging.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Random.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Register.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ReplaceKernelArgs.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ScheduledInstructions.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Scheduler.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TensorDescriptor.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Timer.cpp"
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
        "${CMAKE_CURRENT_SOURCE_DIR}/Utilities/ReadELF.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Assemblers/InProcessAssembler.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ExpressionTransformations/FastDivision.cpp"
)
if(ROCROLLER_ENABLE_LLD)
    target_compile_definitions(rocroller-no-rtti PRIVATE ROCROLLER_USE_LLD)
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
        "${PROJECT_SOURCE_DIR}/lib/include" # is this necessary?
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
    rocroller-no-rtti PROPERTIES POSITION_INDEPENDENT_CODE ON # necessary?
)
