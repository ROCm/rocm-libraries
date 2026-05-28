# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx90a|gfx125")
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
endif()
