function(add_thrust_test_inclusive TEST)
    set(TEST_SOURCE "${TEST}.cu")
    set(TEST_TARGET "test_thrust_inclusive_${TEST}")
    # Unless this property isn't set, CMake silently discards .cu files when
    # CUDA language has not been enabled. If enabled, it will do the regular compiler
    # detection and search for a fully functioning CUDA compiler, which hipcc isn't (yet).
    # Note that removing this property builds will continue to pass, running the tests
    # will also succeed, however the tests will be empty and be near empty main functions.
    # This code path used to add "-x c++" to the compiler flags too, but that was
    # superfluous as "-x hip" was already present on the command-line due to linking to
    # hip::device transitively from roc::rocprim_hip.
    set_source_files_properties(${TEST_SOURCE}
        PROPERTIES
            LANGUAGE CXX
    )
    if(NOT CMAKE_VERSION VERSION_LESS 3.13)
        add_executable(${TEST_TARGET} ${TEST_SOURCE} $<TARGET_OBJECTS:testframework>)
        target_link_libraries(${TEST_TARGET}
            PRIVATE
                testing_common
        )
    else() # Workaround
        add_executable(${TEST_TARGET} ${TEST_SOURCE} ../../unittest/testframework.cu)
        target_link_libraries(${TEST_TARGET}
            PRIVATE
                testing_common
                testframework
        )
    endif()
    set_target_properties(${TEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/testing/"
    )
    add_test(${TEST} ${TEST_TARGET})
    set_tests_properties(${TEST}
        PROPERTIES
            LABELS upstream
    )
endfunction()


# add_thrust_test_inclusive("counting_iterator")
# add_thrust_test_inclusive("discard_output")
# add_thrust_test_inclusive("large_indices")
# add_thrust_test_inclusive("large_types")
add_thrust_test_inclusive("mixed_types")
# add_thrust_test_inclusive("simple")
add_thrust_test_inclusive("stateful_operator")
# add_thrust_test_inclusive("using_vs_adl")
