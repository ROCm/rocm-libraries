# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_subdirectory(src)

set(HIPBLASLT_TEST_DATA "${CMAKE_CURRENT_BINARY_DIR}/../hipblaslt_gtest.data")

add_custom_command(
    OUTPUT "${HIPBLASLT_TEST_DATA}"
    COMMAND ${HIPBLASLT_PYTHON_COMMAND} "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gentest.py"
                                        -I "${CMAKE_CURRENT_SOURCE_DIR}/../common/include"
                                        "${CMAKE_CURRENT_SOURCE_DIR}/data/hipblaslt_gtest.yaml"
                                        -o ${HIPBLASLT_TEST_DATA}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gentest.py
                                        ${CMAKE_CURRENT_SOURCE_DIR}/data/hipblaslt_common.yaml
                                        ${CMAKE_CURRENT_SOURCE_DIR}/data/matmul_common.yaml
                                        ${CMAKE_CURRENT_SOURCE_DIR}/data/matmul_gtest.yaml
                                        ${CMAKE_CURRENT_SOURCE_DIR}/data/auxiliary_gtest.yaml
                                        ${CMAKE_CURRENT_SOURCE_DIR}/data/smoke_gtest.yaml
                                        ${CMAKE_CURRENT_SOURCE_DIR}/data/rocroller_gtest.yaml

)

add_custom_target(hipblaslt-test-data DEPENDS ${HIPBLASLT_TEST_DATA})
add_dependencies(hipblaslt-test hipblaslt-test-data)
if (HIPBLASLT_ENABLE_COVERAGE)
    target_compile_options(hipblaslt-test PRIVATE -H)
    target_link_libraries(hipblaslt-test PRIVATE hipblaslt)
	target_sources(hipblaslt-test
	    PRIVATE
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/include/auxiliary.hpp"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/include/exceptions.hpp"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/include/hipblaslt_ostream.hpp"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/hipblaslt_internal.hpp"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/hipblaslt-ext-op-internal.hpp"

		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/include/Debug.hpp"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/include/rocblaslt-auxiliary.h"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/include/rocblaslt-functions.h"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/include/rocblaslt-types.h"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/include/rocblaslt.h"

		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include/debug.h"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include/definitions.h"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include/handle.h"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include/logging.h"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include/rocblaslt_mat_utils.hpp"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include/status.h"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include/tuple_helper.hpp"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include/tensile_host.hpp"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include/UserDrivenTuningParser.hpp"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include/utility.hpp"

		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/rocroller/include/analytical_utils.hpp"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/rocroller/include/gemm.hpp"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/rocroller/include/kernel_type.hpp"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/rocroller/include/parameter_selection.hpp"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/rocroller/include/rocroller_host.hpp"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/rocroller/include/runtime_args_selection.hpp"
		"${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/rocroller/include/solution_selection.hpp"
	)

    target_include_directories(hipblaslt-test
        PRIVATE
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/clients/common/include>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/common/include>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tensilelite/include>
	    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tensilelite>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/amd_detail/include>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/amd_detail>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/include>
            "${PROJECT_SOURCE_DIR}/tensilelite"
    )
endif()
