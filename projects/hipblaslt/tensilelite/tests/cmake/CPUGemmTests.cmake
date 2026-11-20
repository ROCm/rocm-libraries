# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Register the same command in the build tree and the relocatable test package.
# Keep the original AMAX-D name for limit 8; only limits 1 and 3 add variants.
function(add_cpu_gemm_test TEST_NAME)
    cmake_parse_arguments(ARG "WILL_FAIL" "" "ARGS" ${ARGN})

    set(thread_limits 8)
    if("--outputAmaxD" IN_LIST ARG_ARGS)
        list(APPEND thread_limits 1 3)
    endif()

    foreach(thread_limit IN LISTS thread_limits)
        set(test_name "CPUGemm.${TEST_NAME}")
        if(NOT thread_limit EQUAL 8)
            string(APPEND test_name "_Threads${thread_limit}")
        endif()
        set(test_environment
            "OMP_THREAD_LIMIT=${thread_limit};OMP_NUM_THREADS=${thread_limit};OMP_DYNAMIC=FALSE;KMP_BLOCKTIME=0")

        add_test(NAME "${test_name}" COMMAND cpu-gemm-driver ${ARG_ARGS})
        set_tests_properties("${test_name}" PROPERTIES
            ENVIRONMENT "${test_environment}"
            PROCESSORS "${thread_limit}"
            TIMEOUT 120
            WILL_FAIL "${ARG_WILL_FAIL}"
        )
        list(APPEND CPU_GEMM_TEST_NAMES "${test_name}")

        if(TENSILELITE_TESTS_CTEST_INSTALL_FILE)
            set(install_command "\"./cpu-gemm-driver\"")
            foreach(arg IN LISTS ARG_ARGS)
                string(REPLACE "\\" "\\\\" quoted_arg "${arg}")
                string(REPLACE "\"" "\\\"" quoted_arg "${quoted_arg}")
                string(APPEND install_command " \"${quoted_arg}\"")
            endforeach()
            file(APPEND "${TENSILELITE_TESTS_CTEST_INSTALL_FILE}"
                "add_test(\"${test_name}\" ${install_command})\n"
                "set_tests_properties(\"${test_name}\" PROPERTIES ENVIRONMENT \"${test_environment}\" PROCESSORS ${thread_limit} TIMEOUT 120 WILL_FAIL ${ARG_WILL_FAIL})\n"
            )
        endif()
    endforeach()

    set(CPU_GEMM_TEST_NAMES "${CPU_GEMM_TEST_NAMES}" PARENT_SCOPE)
endfunction()
