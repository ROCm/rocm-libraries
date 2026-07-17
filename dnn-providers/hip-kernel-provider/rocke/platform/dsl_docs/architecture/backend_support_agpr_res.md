What needs to be done in CK DSL
Add a function-attribute knob in the kernel emitter. Wire CK DSL’s kernel attribute map so it can emit, on the LLVM define amdgpu_kernel function:

The min/max are unsigned ints. 0,0 forbids AGPR allocation, which forces all MFMAs in the function to lower to the VGPR form (_vgprcd_e64). Higher values let RA use up to that many AGPRs.
Expose it on the spec / IRBuilder. Add a kernel-level option, e.g. KernelAttrs(agpr_alloc=(min, max) or "none"), used by:
attention 2D / 3D / unified
GEMM, MoE GEMM
any kernel that does element-wise math on MFMA accumulators between MFMAs (online softmax, per-tile scaling, fused activations on accumulators)
Default policy:
Kernels that touch the C accumulator with VALU between MFMAs (attention, MFMA + scale + activation): set agpr_alloc="0,0" (force VGPR form).
Pure GEMM-style kernels where C is read once at the epilogue: leave at default and let the backend pick.
Kernels using warp-specialized explicit asm: leave at default; do not constrain.
Optionally expose the cl::opt mirror. Allow the runtime/comgr layer to pass -mllvm -amdgpu-mfma-vgpr-form (already default-true on this LLVM, but explicit is better). Not strictly required if (1)+(2) is in place.
Document the trade-off in the spec docs.
VGPR form removes AGPR↔VGPR copies, cheaper for accumulator-touching pipelines like online-softmax attention.
VGPR form increases VGPR pressure → may reduce occupancy on register-bound kernels (attention 2D NW=4/8, large MFMA widths). Note that some workloads will want AGPRs back; provide a knob, not a hard rule.
Validate via ISA dumps and benchmarks.
Diff the disassembly of attention 2D before/after (v_mfma_f32_32x32x16 v[..]:..., v[..]..., v[..]..., v[..].. vs a[..]..).
Confirm the rescale loops drop the v_accvgpr_read_b32 / v_accvgpr_write_b32 traffic.
Re-run perf on attention 2D / 3D and any MoE/GEMM kernels we are constraining to confirm no regression at the chosen agpr_alloc setting.
Out-of-scope for now (but document as follow-ups).
Warp-specialized inline-asm MFMA ops with explicit a[..] / v[..] register numbers. Useful for finer scheduling control, not needed for residency selection.
Algorithmic reformulation of online softmax to avoid acc *= alpha. Independent of the residency knob, can stack on top.
