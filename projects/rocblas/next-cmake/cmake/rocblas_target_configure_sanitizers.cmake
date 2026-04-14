# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(rocblas_target_configure_sanitizers rocblas_target visibility)
    # Add asan flags to target
    target_compile_options(${rocblas_target}
        ${visibility}
            -fsanitize=address
            -shared-libasan
    )
    target_link_options(${rocblas_target}
        ${visibility}
            -fsanitize=address
            -shared-libasan
            -fuse-ld=lld
    )
endfunction()
