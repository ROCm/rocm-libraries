# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(tensilelite-tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/JitCompiler_test.cpp"
)

target_link_libraries(tensilelite-tests PRIVATE roc::tensilelite-jit)

if(TENSILELITE_ENABLE_JIT_PYTHON)
    # Bytecode and cache writes are disabled so the run leaves the source tree clean.
    add_test(NAME tensilelite-jit-python
        COMMAND "${Python_EXECUTABLE}" -m pytest -q -p no:cacheprovider
            "${CMAKE_CURRENT_SOURCE_DIR}/bindings"
    )
    set_tests_properties(tensilelite-jit-python
        PROPERTIES
            ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:tensilelite_jit>;PYTHONDONTWRITEBYTECODE=1"
    )
endif()
