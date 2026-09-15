# Mining an algorithm

[RUNBOOK.md](RUNBOOK.md) owns execution order; this page owns where to look and what
each source is good for. Search order is in-tree first, then external, then
documentation — and the search is for an **algorithm**, not for code to paste. What
you take, you cite in the report; what you rejected and why is worth one line each.

Two different questions, answered by different sources:

| Question | Ask |
|---|---|
| What does this operation *mean*, exactly? | the hipDNN frontend attributes and schema ([graph-analysis.md](graph-analysis.md)), then a framework's reference implementation |
| How is it *implemented* well on AMD hardware? | in-tree ROCm libraries, then ISA and optimization documentation |

A framework's semantics are that framework's, not hipDNN's. Use PyTorch to
understand what an operator computes; use the hipDNN schema to decide what *this*
graph asks for.

## In-tree: the monorepo you already have

All paths are `projects/<name>` in this repository. These are the highest-value
sources because they are local, versioned with the graph schema, and written for the
same hardware.

| Op family | Where | Notes |
|---|---|---|
| GEMM, batched/grouped GEMM, GEMM+epilogue fusions | `composablekernel/example/{01_gemm,15_grouped_gemm,24_batched_gemm,65_gemm_multiply_multiply,68_gemm_add,69_gemm_add_relu}`, `composablekernel/include/ck_tile/ops/gemm` | numbered examples are the readable entry point; `ck_tile` is the current tile-programming layer |
| Attention / FMHA | `composablekernel/include/ck_tile/ops/fmha`, `.../sparse_attn`, `.../sageattention`, `composablekernel/example/32_batched_gemm_scale_softmax_gemm` | flash-style tiling, online softmax, masking conventions |
| Convolution | `composablekernel/example/{09_convnd_fwd,11_convnd_fwd_bias,17_convnd_bwd_data,20_grouped_conv_bwd_weight}`, `ck_tile/ops/grouped_convolution`, `miopen/src/kernels/MIOpenConv*.cpp` | MIOpen's direct kernels are the naive-but-correct end; CK is the tiled end |
| Normalization (layernorm, RMS norm, groupnorm, batchnorm) | `ck_tile/ops/{layernorm2d,rmsnorm2d,norm_reduce}` and `composablekernel/example/{27_layernorm2d_fwd,42_groupnorm_fwd}` for layer/group norm; `composablekernel/example/34_batchnorm` — `ck_tile` has no batchnorm — plus `miopen/src/kernels/MIOpenBatchNorm*.cpp` and hipDNN's own `dnn-providers/hip-kernel-provider/src/engines/hip_mlops_engine/kernels/{layernorm,rmsnorm,batchnorm}` | the hip_mlops kernels are already hipRTC-compiled here — closest precedent of all |
| Softmax, reduction, scan, top-k | `ck_tile/ops/{softmax,reduce,topk,topk_softmax}`, `composablekernel/example/{12_reduce,23_softmax,33_multiple_reduce}`, `rocprim/rocprim/include/rocprim/{block,device,intrinsics}` | rocPRIM block/warp primitives are the right shape for a hand-written reduction |
| Pointwise / elementwise / activation | `ck_tile/ops/elementwise`, `composablekernel/example/{19_binary_elementwise,44_elementwise_permute}`, `dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/kernels/Pointwise*.cpp` | the Pointwise kernels are the minimal hipRTC style exemplar |
| Transpose, permute, layout change, im2col | `ck_tile/ops/{batched_transpose,permute,image_to_column}`, `composablekernel/example/{39_permute,52_im2col_col2im}` | |
| Pooling / resample | `ck_tile/ops/pooling`, `composablekernel/example/{13_pool2d_fwd,48_pool3d_fwd,49_maxpool2d_bwd}`, `hip_mlops_engine/kernels/resample` | |
| MoE / grouped matmul | `ck_tile/ops/{fused_moe,moe_flatmm.hpp}`, `composablekernel/example/{15_grouped_gemm,59_grouped_gemm_multi_ABD}` | |
| Quantization / block scaling | `ck_tile/ops/{gemm_quant,smoothquant,add_rmsnorm2d_rdquant}`, `composablekernel/example/{14_gemm_quantization,67_gemm_microscaling}` | |
| Matrix-core (MFMA) fragment handling | `rocwmma/library/include/rocwmma`, CK's tile layer | see the MFMA caveat in [device-envelope.md](device-envelope.md) |
| hipBLASLt / Tensile | `hipblaslt`, `shared/tensile` | assembly-generated GEMM; read for tiling and scheduling ideas, not for source to adapt |

`miopen/src/kernels/*.cpp` deserves a note: those are HIP kernels MIOpen compiles at
runtime, so they are written under the same no-STL, macro-parameterized constraints a
hipRTC kernel must respect. They are the closest stylistic match to what this skill
produces. MIOpen's `.s` files are hand-written assembly and are not a model for a
hipRTC kernel.

hipDNN's own hipRTC kernels — `kernel_ingestor_engine/kernels/{PointwiseAdd,PointwiseMul,PointwiseSub,ConvFwd}.cpp`
and the richer `hip_mlops_engine/kernels/**` tree — are the style to imitate first:
they already compile through the exact path your kernel will.

## External: semantics and published implementations

Neither PyTorch nor Triton is cloned on this machine; read them over the network.

| Source | URL | Good for |
|---|---|---|
| PyTorch ATen CUDA/HIP kernels | https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/cuda | the authoritative *semantics* of an operator, including edge-case and dtype-promotion behaviour |
| Triton | https://github.com/triton-lang/triton | tutorials (`python/tutorials`) are the clearest published derivations of tiled softmax, matmul and flash attention; `third_party/amd` for the AMD backend |
| AOTriton | https://github.com/ROCm/aotriton | Triton-authored attention kernels shipped for ROCm; the closest published AMD attention numerics |
| AITER | https://github.com/ROCm/aiter | AMD's kernel collection for inference operators |
| Composable Kernel (upstream) | https://github.com/ROCm/composable_kernel | when the in-tree copy is older than the discussion you found |

Use these for the shape of the algorithm and the definition of the operator. Do not
transplant a CUDA kernel's warp size, shared-memory budget or instruction selection:
wavefront 64 and the CDNA LDS/register budgets are different, and a `__syncwarp()`
assumption that is free on NVIDIA is a correctness bug here.

## Documentation

Verified reachable:

| URL | Answers |
|---|---|
| https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html | what the kernel language provides: qualifiers, built-in variables, intrinsics, cooperative groups, and the explicit "no warp matrix types, no dynamic parallelism" limits |
| https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html | hipRTC's own contract: what it compiles, which headers it bundles, `__HIPCC_RTC__`, and its documented restrictions |
| https://rocm.docs.amd.com/projects/HIP/en/latest/reference/math_api.html | device math functions and their precision |
| https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html | memory coalescing, occupancy, and launch-configuration guidance |
| https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html | CUDA-to-HIP differences, including the ones that silently change behaviour |
| https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html | per-architecture specifications — the table to use when the target device is not in front of you |
| https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch.html | CDNA/RDNA architecture overviews |
| https://llvm.org/docs/AMDGPUUsage.html | the AMDGPU backend's own reference: target features, address spaces, and the `__builtin_amdgcn_*` surface |
| https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html | MFMA instruction shapes, fragment layouts and worked examples |
| https://rocm.blogs.amd.com/ , https://gpuopen.com/learn/amd-lab-notes/ | AMD lab notes: memory-bound optimization, LDS bank conflicts, kernel case studies |

Architecture ISA PDFs (CDNA3/CDNA4 instruction set architecture) are published on
amd.com but were not reachable from this environment at authoring time; search
`rocm.docs.amd.com` or `amd.com` for the current link rather than citing a stale one.

A documentation fact used for sizing — LDS per workgroup, register budget, maximum
threads per block — should be cross-checked against `rocminfo` on the actual target
before a kernel is tuned to it ([device-envelope.md](device-envelope.md)).

## Recording what you mined

In the report, per source used: what it gave you (algorithm, tiling scheme, masking
convention, numeric edge case), and what you changed. One line each. A kernel whose
provenance is "written from the operation specification" is a legitimate answer —
say that rather than implying a source you did not read.
