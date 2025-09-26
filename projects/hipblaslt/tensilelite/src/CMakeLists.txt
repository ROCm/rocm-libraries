# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(HIPBLASLT_ENABLE_LLVM)
    add_subdirectory(llvm)
endif()

if(HIPBLASLT_ENABLE_MSGPACK)
    add_subdirectory(msgpack)
endif()

add_subdirectory(hip)

target_sources(tensilelite-host
    PRIVATE
        #"${CMAKE_CURRENT_SOURCE_DIR}/ArithmeticUnitTypes.cpp" # this file isn't built in existing src tree
        "${CMAKE_CURRENT_SOURCE_DIR}/Activation.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/AMDGPU.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ContractionProblem.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ContractionSolution.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DataTypes.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Debug.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/EmbeddedData.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/EmbeddedLibrary.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/KernelArguments.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/KernelLanguageTypes.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MLFeatures.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MLPNet.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/PerformanceMetricTypes.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ScalarValueTypes.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TensorDescriptor.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TensorOps.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Tensile.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp"
)
