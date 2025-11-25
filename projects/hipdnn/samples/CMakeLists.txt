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
find_package(hipdnn_frontend CONFIG REQUIRED)

# Helper function to add a hipDNN sample executable NAME: The name of the executable target SOURCE:
# The source file for the sample
function(add_hipdnn_sample NAME SOURCE)
    add_executable(${NAME} ${SOURCE})
    target_link_libraries(${NAME} PRIVATE hip::host hipdnn_frontend)
endfunction()

# Temporarily disabled bn_inference due to https://github.com/ROCm/rocm-libraries/issues/2459
# add_hipdnn_sample(bn_inference batchnorm/BnInference.cpp)
add_hipdnn_sample(bn_training batchnorm/BnTraining.cpp)
add_hipdnn_sample(fused_bn_training_activ batchnorm/FusedBnTrainingActiv.cpp)
add_hipdnn_sample(bn_backward batchnorm/BnBackward.cpp)
add_hipdnn_sample(fused_bn_inference_drelu_bn_backward batchnorm/FusedBnInfDReluBnBwd.cpp)

add_hipdnn_sample(conv_dgrad convolution/ConvDgrad.cpp)
add_hipdnn_sample(conv_fprop convolution/ConvFprop.cpp)
add_hipdnn_sample(conv_wgrad convolution/ConvWgrad.cpp)
