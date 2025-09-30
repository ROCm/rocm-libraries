# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Enable Address Sanitizer and set linker flags for security
message(STATUS "Building with Sanitizers: ${BUILD_ADDRESS_SANITIZER}")

if (BUILD_ADDRESS_SANITIZER)

    # Address Sanitizer requires specific GPU targets which
    # support XNACK.
    set(GPU_TARGETS 
        gfx908:xnack+    # MI100 (Arcturus)
        gfx90a:xnack+    # MI200 series (MI210, MI250, MI250X)
        gfx942:xnack+    # MI300X (GPU)
    )

    link_directories(${ROCM_LLVM_LIB_DIR}/clang/20/lib/linux)

    # Define sanitizer flags as variables for reuse
    set(SANITIZER_COMPILE_FLAGS 
        -fsanitize=address
        -fno-omit-frame-pointer
    )
    
    set(SANITIZER_LINK_FLAGS 
        -fsanitize=address
        -fno-omit-frame-pointer
        -shared-libasan
    )

    # Apply sanitizer flags globally (can be overridden per target)
    add_compile_options(${SANITIZER_COMPILE_FLAGS})
    add_link_options(${SANITIZER_LINK_FLAGS})
    
    # Add compile definition for conditional compilation
    add_compile_definitions(ADDRESS_SANITIZER)

    # Set environment variables for Address Sanitizer
    # XNACK is required for Address Sanitizer on AMD GPUs
    # ASAN_SYMBOLIZER_PATH is set to the LLVM symbolizer to make the output
    # from leak detection more readable.
    set(TEST_ENVIRONMENT 
        "ASAN_SYMBOLIZER_PATH=${CMAKE_SYMBOLIZER}" 
        "HSA_XNACK=1"
        #"ASAN_OPTIONS=halt_on_error=1:abort_on_error=1"
    )
endif()
