# TheRock / Packaging Notes

## Multi-arch Approach: Precompiled .co (Code Objects)

Following discussion with Brian Harrison (2026-06-17), we use the **precompiled
hsaco approach** (Option 2) rather than hipRTC or static compilation.

**Rationale:**
- rocWMMA is NOT hipRTC-friendly (C++ template library, not single-header)
- hipRTC compile overhead would be ~10s+ on first use
- Precompiled `.co` files are arch-specific, no rocWMMA dependency at runtime

### Precompiled Kernels (in this PR)

```
kernels/
├── hip_flash2_fwd_gfx942.co   # MI300X / MI325X (CDNA3, 145KB)
└── hip_flash2_fwd_gfx950.co   # MI355X / MI350X (CDNA4, 131KB)
```

Generated on Alola (2026-06-17) using:
```bash
clang++ --offload-arch=gfx942 -O3 -std=c++17 --cuda-device-only \
    -I/opt/rocm-7.2.0/include -x hip HipFlash2FwdPlan.hip \
    -o kernels/hip_flash2_fwd_gfx942.co

clang++ --offload-arch=gfx950 -O3 -std=c++17 --cuda-device-only \
    -I/opt/rocm-7.2.0/include -x hip HipFlash2FwdPlan.hip \
    -o kernels/hip_flash2_fwd_gfx950.co
```

### TheRock Change (simplified - NO rocWMMA needed)

Since the kernel is pre-compiled, **no build-time dependency on rocWMMA**.

The only TheRock change needed is to install the `.co` files with the package:

```cmake
# In artifact-hipkernelprovider.toml or CMakeLists.txt install rules:
install(FILES
    src/engines/hip_flash2_engine/kernels/hip_flash2_fwd_gfx942.co
    src/engines/hip_flash2_engine/kernels/hip_flash2_fwd_gfx950.co
    DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/hip_kernel_provider/hip_flash2_kernels
)
```

And set the compile definition:
```cmake
-DHIP_FLASH2_KERNEL_DIR="${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/hip_kernel_provider/hip_flash2_kernels"
```

### Runtime Loading

The `HipFlas
