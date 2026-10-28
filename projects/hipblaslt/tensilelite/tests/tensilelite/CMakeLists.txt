# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(TENSILELITE_ENABLE_JIT)
    target_sources(tensilelite-tests
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/API/Compiler_test.cpp"
    )

    target_link_libraries(tensilelite-tests PRIVATE roc::tensilelite-jit)
endif()

# The codegen tests assemble and launch kernels, so they need the JIT library.
if(TENSILELITE_ENABLE_CODEGEN AND TENSILELITE_ENABLE_JIT)
    add_subdirectory(Codegen)
endif()

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
