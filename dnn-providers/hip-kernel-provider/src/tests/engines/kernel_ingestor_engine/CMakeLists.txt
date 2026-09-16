# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hip_kernel_provider_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/TestHandleDeviceResolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestKernelIngestorEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestIngestorPacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwiseAddDispatchHandler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwiseAddMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwisePacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestConvFwdPack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestBatchnormInferencePacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestBatchnormInferenceMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestHiprtcFileKernelSource.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestConvBiasPacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestConvBiasMatchers.cpp
)

# TestConvBiasMatchers.cpp drives buildConvBiasGraph() through a ConvBiasGraphSpec whose
# call sites name the one or two knobs each case varies. Clang accepts that spelling under
# -std=gnu++17 as an extension; -Wpedantic reports it and -Werror promotes it. The
# suppression is scoped to this one file so the rest of the suite keeps the diagnostic.
# TARGET_DIRECTORY is required because hip_kernel_provider_tests is declared in the parent
# directory, and source-file properties resolve against the target's directory scope.
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/packs/TestConvBiasMatchers.cpp
    TARGET_DIRECTORY hip_kernel_provider_tests
    PROPERTIES COMPILE_OPTIONS "-Wno-c++20-designator"
)
