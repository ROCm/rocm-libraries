# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(TENSILELITE_ENABLE_JIT)
    target_sources(tensilelite-tests
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/JitCompiler_test.cpp"
    )

    target_link_libraries(tensilelite-tests PRIVATE roc::tensilelite-jit)
endif()

if(TENSILELITE_ENABLE_CODEGEN)
    add_subdirectory(codegen)
endif()

if(TENSILELITE_ENABLE_JIT_PYTHON)
    # Bytecode and cache writes are disabled so the run leaves the source tree clean.
    add_test(NAME tensilelite-jit-python
        COMMAND "${Python_EXECUTABLE}" -m pytest -q -p no:cacheprovider
            "${CMAKE_CURRENT_SOURCE_DIR}/bindings/test_compiler.py"
    )
    set_tests_properties(tensilelite-jit-python
        PROPERTIES
            ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:tensilelite_jit>;PYTHONDONTWRITEBYTECODE=1"
    )

    # The module and the tests that launch kernels need the HIP runtime.
    if(TENSILELITE_ENABLE_CODEGEN)
        add_test(NAME tensilelite-codegen-python
            COMMAND "${Python_EXECUTABLE}" -m pytest -q -p no:cacheprovider
                "${CMAKE_CURRENT_SOURCE_DIR}/bindings/test_skinny_gemm.py"
        )
        set_tests_properties(tensilelite-codegen-python
            PROPERTIES
                ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:tensilelite_codegen>:$<TARGET_FILE_DIR:tensilelite_jit>;PYTHONDONTWRITEBYTECODE=1"
        )
        if(TARGET hip::amdhip64)
            set_tests_properties(tensilelite-codegen-python
                PROPERTIES
                    ENVIRONMENT_MODIFICATION
                        "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:hip::amdhip64>"
            )
        endif()
    endif()
endif()
