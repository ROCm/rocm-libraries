# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

function(hipblaslt_add_tensilelite_identity_tests)
    set(_checker "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tensilelite/tests/check_client_version.cmake")
    set(_labels "standard;comprehensive;full;pr;precheckin;nightly;extended;weekly")
    foreach(_query IN ITEMS version source-revision build-rocm)
        add_test(NAME TensileLiteClient.${_query}
            COMMAND "${CMAKE_COMMAND}"
                "-DCLIENT=$<TARGET_FILE:tensilelite-client>"
                "-DQUERY=--${_query}" "-DMETADATA=${TENSILELITE_METADATA_FILE}"
                -P "${_checker}")
        set_tests_properties(TensileLiteClient.${_query} PROPERTIES LABELS "${_labels}" TIMEOUT 10)
    endforeach()
    if(TENSILELITE_TESTS_CTEST_INSTALL_FILE)
        # Tests must work on another machine: resolve cmake on the runner's PATH
        # and keep artifact paths relative to the installed test directory.
        file(RELATIVE_PATH _client_relative
            "/${CMAKE_INSTALL_BINDIR}/${TENSILELITE_TESTS_INSTALL_SUBDIR}"
            "/${CMAKE_INSTALL_LIBEXECDIR}/hipblaslt/tensilelite/tensilelite-client${CMAKE_EXECUTABLE_SUFFIX}")
        foreach(_query IN ITEMS version source-revision build-rocm)
            file(APPEND "${TENSILELITE_TESTS_CTEST_INSTALL_FILE}"
                "add_test(TensileLiteClient.${_query} \"cmake\" \"-DCLIENT=${_client_relative}\" \"-DQUERY=--${_query}\" \"-DMETADATA=metadata.json\" -P \"check_client_version.cmake\")\n"
                "set_tests_properties(TensileLiteClient.${_query} PROPERTIES LABELS \"${_labels}\" TIMEOUT 10)\n")
        endforeach()
        install(FILES "${_checker}" "${TENSILELITE_METADATA_FILE}"
            DESTINATION "${CMAKE_INSTALL_BINDIR}/${TENSILELITE_TESTS_INSTALL_SUBDIR}" COMPONENT tests)
    endif()
endfunction()
