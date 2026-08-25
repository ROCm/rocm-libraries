---
id: project-stinkytofu
title: "stinkytofu"
type: project
repo: ROCm/rocm-libraries
tree: shared/stinkytofu
tags: [stinkytofu]
operator_families: [gemm]
architecture_families: [gfx12]
architectures: [gfx1250]
related: [hw-gfx1250, technique-gfx1250-wmma-k32, project-tensilelite]
kernel_types: [gemm]
---

# stinkytofu

LLVM-style pass pipeline on AMDGPU **assembly** (logical IR + asm IR): DAG
schedule, waitcnt insertion, DCE, peephole. Used by hipBLASLt/TensileLite for
gfx1250. TableGen `.def` files add arches without C++ rewrites.

Rocke does not replace this for Tensile-generated kernels. When optimizing
gfx1250 **rocke** WMMA builders, still inspect waitcnt with rocke probes; when
the kernel came from TensileLite, stinkytofu is the scheduler to reason about.

`shared/stinkytofu/README.md` on `develop`.
