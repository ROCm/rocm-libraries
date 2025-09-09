# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

function(rocroller_target_configure_sanitizers rocroller_target linkage)
    # Add asan flags to hipblas_target
    target_compile_options(${rocroller_target}
        ${linkage}
            -fsanitize=address
            -shared-libasan
    )
    target_link_options(${rocroller_target}
        ${linkage}
            -fsanitize=address
            -shared-libasan
            -fuse-ld=lld
    )
endfunction()
