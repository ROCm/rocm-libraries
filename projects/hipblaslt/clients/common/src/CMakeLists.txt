# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hipblaslt-clients-common
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/singletons.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_bench_options.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/utility.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/efficiency_monitor.cpp"
        "${ROCM_LIBRARIES_ROOT}/shared/host-validation/adapters/hipblaslt/HipblasltReferenceGemm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/argument_model.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_parse_data.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_arguments.cpp"
        "${ROCM_LIBRARIES_ROOT}/shared/host-validation/adapters/hipblaslt/hipblaslt_random.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_init_device.cpp"
)

if(amd_smi_FOUND)
    target_link_libraries(hipblaslt-clients-common PRIVATE amd_smi)
endif()
