# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(rocroller
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/CombineShifts.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ConvertPropagation.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DataFlowTagPropagation.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FastArithmetic.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FastDivision.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FastMultiplication.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FuseAssociative.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/FuseTernary.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Identity.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LaunchTimeSubExpressions.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LowerUnsignedArithmeticShiftR.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LowerBitfieldValues.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LowerExponential.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LowerPRNG.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/PositionalArgumentPropagation.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RestoreCommandArguments.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Simplify.cpp"
)
