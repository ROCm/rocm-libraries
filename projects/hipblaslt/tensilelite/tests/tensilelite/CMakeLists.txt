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
    target_sources(tensilelite-tests
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/API/SkinnyGEMM_test.cpp"
    )
    # API tests include the shared support header as Codegen/TestSupport.hpp.
    target_include_directories(tensilelite-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_subdirectory(Codegen)
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
