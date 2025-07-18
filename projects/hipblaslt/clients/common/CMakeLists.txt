# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_subdirectory(src)

target_sources(hipblaslt-clients-common
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include/allclose.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/argument_model.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/cblas_interface.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/datatype_interface.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/device_vector.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/d_vector.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/efficiency_monitor.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/flops.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblaslt_arguments.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblaslt_data.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblaslt_datatype2string.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblaslt_init.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblaslt_math.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblaslt_parse_data.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblaslt_random.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblaslt_test.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/host_vector.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/hipBuffer.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/mxDataGen.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/near.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/norm.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/singletons.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/TensorDataManipulation.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/test_cleanup.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/testing_auxiliary.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/testing_matmul.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/unit.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/utility.hpp"
)
