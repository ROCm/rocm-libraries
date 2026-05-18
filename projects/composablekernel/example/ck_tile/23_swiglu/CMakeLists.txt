find_package(HIP CONFIG REQUIRED)

add_executable(example_swiglu "main.cpp")
target_sources(
    example_swiglu
    PUBLIC
    FILE_SET HEADERS
    FILES
    config.hpp
    invoker.hpp
    run_swiglu_example.hpp
    PRIVATE
    main.cpp
)
