# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

# Enable PIC/PIE to ensure compatibility with the plugin loader system (dlopen). This prevents
# potential Thread Local Storage (TLS) model mismatches between the executable and dynamically
# loaded backend plugins.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(hipdnn_samples VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(hip REQUIRED)

# Technically this is all found as part of hipdnn_test_sdk, but including these as an example for
# consumers. hipdnn_test_sdk shouldn't be needed by normal users of hipDNN, and is only incuded here
# for validation purposes.
find_package(hipdnn_frontend CONFIG REQUIRED)
find_package(hipdnn_data_sdk CONFIG REQUIRED)

# !! NOTE !! - hipdnn_test_sdk is needed for sample validation, regular users of hipDNN do not need
# this package.
find_package(hipdnn_test_sdk CONFIG REQUIRED)

# Helper function to add a hipDNN sample executable NAME: The name of the executable target SOURCE:
# The source file for the sample
function(add_hipdnn_sample NAME SOURCE)
    add_executable(${NAME} ${SOURCE})
    # !! NOTE !! - hipdnn_test_sdk is needed for sample validation, regular users of hipDNN do not
    # need this package. Linking against hipdnn_frontend would be sufficient for normal users
    # (brings in hipdnn_data_sdk as well).
    target_link_libraries(${NAME} PRIVATE hip::host hipdnn_frontend hipdnn_test_sdk)
endfunction()

# Temporarily disabled bn_inference due to https://github.com/ROCm/rocm-libraries/issues/2459
# add_hipdnn_sample(bn_inference batchnorm/BnInference.cpp)
add_hipdnn_sample(bn_training batchnorm/BnTraining.cpp)
add_hipdnn_sample(fused_bn_training_activ batchnorm/FusedBnTrainingActiv.cpp)
add_hipdnn_sample(bn_backward batchnorm/BnBackward.cpp)
add_hipdnn_sample(fused_bn_inference_drelu_bn_backward batchnorm/FusedBnInfDReluBnBwd.cpp)

add_hipdnn_sample(conv_dgrad convolution/ConvDgrad.cpp)
add_hipdnn_sample(conv_fprop convolution/ConvFprop.cpp)
add_hipdnn_sample(fused_conv_fprop_activ convolution/FusedConvFpropActiv.cpp)
add_hipdnn_sample(fused_conv_fprop_bias_activ convolution/FusedConvFpropBiasActiv.cpp)
add_hipdnn_sample(conv_wgrad convolution/ConvWgrad.cpp)

if(WIN32)
    # On Windows, spdlog::spdlog_header_only has some INTERFACE_COMPILE_OPTIONS that cause issues
    # when consuming the hipdnn_data_sdk.  We need to clear these options Refer to issue:
    # https://github.com/ROCm/TheRock/issues/2003

    if(TARGET spdlog::spdlog_header_only)
        # Can't do this to an alias
        get_property(aliased_target_name TARGET spdlog::spdlog_header_only PROPERTY ALIASED_TARGET)
        message(STATUS "spdlog::spdlog_header_only aliased target name: '${aliased_target_name}'")

        if(NOT aliased_target_name)
            set_target_properties(
                spdlog::spdlog_header_only PROPERTIES INTERFACE_COMPILE_OPTIONS ""
            )
        endif()
    endif()
endif()
