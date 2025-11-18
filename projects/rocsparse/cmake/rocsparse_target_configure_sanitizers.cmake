# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(rocsparse_target_configure_sanitizers rocsparse_target visibility)
    # Add asan flags to target
    target_compile_definitions(${rocsparse_target} ${visibility} ASAN_BUILD)
    target_compile_options(${rocsparse_target}
        ${visibility}
            -fsanitize=address
            -shared-libasan
    )
    target_link_options(${rocsparse_target}
        ${visibility}
            -fsanitize=address
            -shared-libasan
            -fuse-ld=lld
    )
endfunction()
