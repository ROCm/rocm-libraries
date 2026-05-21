# Glossary

## AMDGPU

AMD GPU architecture target. The DSL emits AMDGPU LLVM IR and uses AMDGPU intrinsics directly. Default target ISA in this repo: `amdgcn-amd-amdhsa--gfx950`.

## BF8

Bfloat8 e5m2 storage dtype. Loads via `cvt_bf8_to_f32`. 5 exponent bits, 2 mantissa bits, max representable abs ~57344.

## BF16

Bfloat16 storage dtype. Used by selected attention paths. The 16x16x16 bf16 MFMA on gfx950 lowers via the `_1k` variant with `<4 x i16>` operands.

## Buffer Resource

An AMDGPU 128-bit buffer descriptor used for bounds-checked raw buffer loads / stores. Created by `b.buffer_rsrc(ptr, num_bytes)` -> `llvm.amdgcn.make.buffer.rsrc.p1`. DW3 flags = `0x00027000` (TYPE=2, DATA_FORMAT=4, NUM_FORMAT=4). OOB byte offsets silently return zero.

## CDNA

AMD compute-die architecture family. CDNA3 = gfx940 / gfx942 / gfx950. CDNA4 = gfx950 successor.

## CHIPLET / XCD

Accelerated Compute Die. MI300X / MI325X / MI350X have 8 XCDs. The default linear blockIdx round-robins WGs across XCDs (worst-case for L2 reuse). `chiplet_transform_chunked` remaps so every `chunk_size` consecutive WGs land on the same XCD.

## COMGR

AMD compiler-support library (`libamd_comgr.so`). Drives LLVM IR -> bitcode -> relocatable -> HSACO in-process.

## CShuffle

An epilogue strategy that stages accumulator results through LDS so final stores can be wide and coalesced. Implemented in `helpers/epilogues.py::CShuffleEpilogue`.

## CTA

Cooperative thread array; equivalent to one GPU workgroup / block.

## Descriptor

A data object that maps logical tensor coordinates to linear offsets and, when needed, validity predicates. The "rich" form in `transforms.py` produces SSA arithmetic via `TensorDescriptor.offset(b, **coords)`.

## Direct Epilogue

An epilogue strategy that stores each lane's accumulator values directly to output memory (no LDS staging). Implemented in `helpers/epilogues.py::DirectEpilogue`.

## ds_bpermute

AMDGPU wave64 cross-lane permute via LDS. Lane `l` reads `data[(addr[l] >> 2) & 63]`. The substrate for `warp_shuffle_xor` and the attention butterfly reductions.

## FP8

8-bit floating-point storage. `FP8E4M3`: 4 exp / 3 mantissa, max abs 448. `BF8E5M2`: 5 exp / 2 mantissa, max abs ~57344.

## GEMM

General matrix multiply. In the DSL the canonical builder is `build_universal_gemm` with `RCR` layout.

## gfx940/942/950

AMDGPU MCPU strings for CDNA3 family devices. The default in this repo is `gfx950`.

## HSACO

HSA code object — the AMDGPU device binary. Output of COMGR; input to `hipModuleLoadData`.

## IRBuilder

The Python object used to emit typed SSA operations into a `KernelDef`. Defined in `core/ir.py`.

## KernelDef

The first-class kernel object: name, params, body region, and attributes. The boundary between authoring (helpers / instances) and lowering (`core/lower_*`).

## K-packed MFMA

A matrix-multiply atom that increases the K dimension per atom (e.g. 16x16x32, 32x32x16). On wave64, the per-lane A operand covers a contiguous K slice `[c4 * width_k : c4 * width_k + width_k]`. The wrong "flat-concat" packing compiles and runs but is not bit-correct.

## LDS

Local Data Share — AMD's shared memory. Modeled by `SmemType`, `tile.smem_alloc`, and `addrspace(3)` LLVM lowering. Two-bank-conflict-avoidance tools: `LdsLayout` (K-pad / packed-async layout) and `TransposeLdsReader` (XOR swizzle in consumer reads).

## libamdhip64

The ROCm HIP runtime library. Used by `runtime/hip_module.py` via ctypes for module load, function lookup, launch, allocation, memcpy, and event timing.

## MFMA

Matrix Fused Multiply-Add. AMD matrix instruction (`v_mfma_f32_*`). Used by GEMM, convolution (implicit and direct), and attention.

## MfmaAtom

DSL helper that packages one MFMA shape (`m, n, k`), per-lane operand widths (`a_per_lane`, `b_per_lane`, `c_per_lane`), accumulator dtype, intrinsic dispatch (`atom.emit(b, A, B, C)`), and lane-to-output mapping (`atom.lane_to_output(b, lane, i)`).

## MLIR-Style IR

The printed textual form from `print_ir()`. Inspection only; not parsed back into the compiler.

## OOB Sentinel

A deliberately invalid byte offset (`INT32_MAX = (1 << 31) - 1 = 2147483647 = 0x7FFFFFFF`) selected for false-mask lanes so the AMDGPU buffer-resource bounds check makes the access safe. The default in `AsyncTileLoader.issue` and the loader helpers.

## Online Softmax

Numerically stable streaming softmax update for attention. Tracks `m` (current max), `l` (denominator), and `acc` (weighted accumulator) without materializing the full attention matrix.

## PipelineLauncher

Runtime helper that launches multiple `KernelLauncher`s on the same stream as one logical pipeline. Same-stream FIFO ordering provides correctness; only the last stage honors `LaunchConfig.fence`.

## Raw Buffer Load LDS

AMDGPU intrinsic `llvm.amdgcn.raw.ptr.buffer.load.lds` that copies from a buffer resource directly into LDS without a VGPR payload. The substrate for `AsyncTileLoader`. Constraints: `dwords in {1, 3, 4}`, lane-contiguous LDS writes, VMEM completion (consumer needs `s_waitcnt(vmcnt=0)`).

## readfirstlane / pin_sgpr

`readfirstlane` broadcasts lane 0's value across the wave (lifts to SGPR). `pin_sgpr` is an `asm` constraint that forces the value to stay in an SGPR across uses. `to_sgpr_u32(v)` composes them — the canonical "lift wave-uniform i32 to scalar registers" idiom.

## Region

A list of IR operations. Kernel body and structured control-flow bodies are regions.

## SmemType

`SmemType(elem, shape)` — the type of an LDS allocation token. Lowers to `ptr addrspace(3)`.

## SSA Value

A typed single-assignment result in the IR. `Value.__bool__` raises by design; SSA values cannot drive Python branches.

## scf.for / scf.for_iter / scf.if

Structured control flow. `scf.for` is a runtime loop with a single induction variable; `scf.for_iter` carries loop-carried values (used for accumulators); `scf.if` is a runtime conditional. Lowering emits explicit basic blocks with phi nodes.

## Tile

A block-owned or warp-owned logical rectangular chunk of work (e.g. `tile_m x tile_n`).

## TileDistributionEncoding

The CK Tile distributed-tensor mapping: `(Rs, Hs, Ps2RHs, Ys2RHs)` describing how a tile is partitioned across replicates / spatial dims / threads / per-lane Y space. Used by the distribution-driven path (`load_tile`, `store_tile`).

## TileWindow

A helper object representing a tile-sized window over a tensor view with an origin and movement operations (`move_to`, `shift_by`).

## Transform DAG

Coordinate-transform graph in `transforms.py` that maps user / logical coordinates to lower-level coordinates, offsets, and validity predicates. Constructors: `pass_through`, `pad`, `pad_dynamic`, `embed`, `merge`, `unmerge`, `indirect`.

## VGPR / SGPR

Vector general-purpose register / scalar general-purpose register. High VGPR usage can reduce occupancy. Inspect via `analyze_hsaco(...).resources`.

## waitcnt

The AMDGPU wait-counter instruction. Used to drain pending VMEM / LGKM / EXP operations before consumption. `b.s_waitcnt(vmcnt=N, lgkmcnt=M)` emits the encoded immarg directly; `b.sync()` and `b.sync_lds_only()` wrap typical drain-plus-barrier patterns.

## Wave64

AMD wavefront of 64 lanes. Current MFMA lane mappings and helpers assume wave64. No wave32 path exists today.

## WorkspacePool

Runtime helper that keeps temporary torch workspace tensors alive and reusable across asynchronous pipeline launches. Solves the workspace-lifetime race where torch's caching allocator can recycle storage while a raw HIP kernel is still reading it.
