# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)
project(hipdnn_samples VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(hip REQUIRED)
find_package(hipdnn_frontend CONFIG REQUIRED)

function(add_hipdnn_sample NAME SOURCE)
    add_executable(${NAME} ${SOURCE})
    target_link_libraries(${NAME} PRIVATE 
        hip::host
        hipdnn_frontend
    )
endfunction()

add_hipdnn_sample(bn_inference batchnorm/BnInference.cpp)
add_hipdnn_sample(bn_training batchnorm/BnTraining.cpp)
add_hipdnn_sample(bn_backward batchnorm/BnBackward.cpp)

add_hipdnn_sample(conv_forward convolution/ConvForward.cpp)
