# Try to find an installed GTest first. QUIET suppresses the "not found" message
# so we can fall back to FetchContent without noisy warnings.
find_package(GTest 1.12.1 QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
        SYSTEM
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(race-emulator-tests
    e2e_hip_general.cpp
    e2e_hip_kittens.cpp
    e2e_tensilelite.cpp
    race_tests.cpp
    vector_instruction_tests.cpp
    scalar_instruction_tests.cpp
    memory_instruction_tests.cpp
    control_flow_instruction_tests.cpp
    parsing_tests.cpp
    util_tests.cpp
)

target_compile_definitions(race-emulator-tests PRIVATE
    TEST_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/asm"
)

target_link_libraries(race-emulator-tests PRIVATE race-emulator GTest::gtest_main)

# race-emulator uses self-registering static globals to populate the instruction
# registry. With a static library the linker strips object files that have no
# externally referenced symbols, so those constructors never run. Force all
# object files to be linked in.
if(NOT BUILD_SHARED_LIBS AND NOT RACE_EMULATOR_BUILD_SHARED_LIBS)
    if(MSVC)
        target_link_options(race-emulator-tests PRIVATE "/WHOLEARCHIVE:race-emulator")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_link_options(race-emulator-tests PRIVATE
            "LINKER:--whole-archive" $<TARGET_FILE:race-emulator> "LINKER:--no-whole-archive")
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(race-emulator-tests PRIVATE -Wall -Wextra -pedantic)
endif()

include(GoogleTest)
gtest_discover_tests(race-emulator-tests)
