# Core IR Model

The core IR lives in `python/ck_dsl/core/ir.py`. It is a small typed SSA IR designed around CK Tile-style kernels and AMDGPU intrinsics. Every operation it represents has a direct LLVM lowering in `core/lower_llvm.py`.

## Core Objects

### `Type`

Named scalar types are singletons exported from `core/ir.py`:

```text
I1, I8, I32, I64
F16, BF16, F32
FP8E4M3      # e4m3 fp8 (cvt_fp8_to_f32 / cvt_f32_to_fp8)
BF8E5M2      # e5m2 fp8 (cvt_bf8_to_f32 / cvt_f32_to_bf8)
```

Composite types:

- `VectorType(elem, count)` — name format `vec<f16x8>`.
- `PtrType(pointee, space)` — name format `ptr<f16,global>`. Address spaces lower to:
  - `global` -> LLVM `ptr addrspace(1)`
  - `lds`    -> LLVM `ptr addrspace(3)`
  - other     -> LLVM `ptr`
- `SmemType(elem, shape)` — opaque token for an LDS allocation. Lowers to `ptr addrspace(3)`.

Cache-coherence AUX hints (constants in `core/ir.py`) for `async_buffer_load_lds_addr` / `buffer_load_vN_*`:

```text
CACHE_ALL    = 0   # default, cache at all levels
CACHE_GLOBAL = 1   # GLC set, skip L2
CACHE_STREAM = 2   # SLC set, streaming hint (don't evict useful lines)
NON_TEMPORAL = 3   # GLC + SLC, bypass cache hierarchy
```

### `Value`

```text
name : "%v42" or "%A"
type : Type
op   : Optional[Op]   # producer
```

`Value.__bool__` raises `TypeError`. This is the central safety rail:

> Runtime SSA values cannot drive Python control flow.

Use `IRBuilder.static_if(host_bool, ...)` for Python-time decisions and `IRBuilder.scf_if(ssa_predicate)` for device-time branches.

### `Op`

```text
name      : str
operands  : List[Value]
results   : List[Value]
attrs     : Dict[str, Any]
regions   : List[Region]
loc       : Optional[str]
```

`Op.is_pure` is consulted by `core/passes.py`. Memory ops, barriers, async copies, atomics, and MFMAs are not removed by dead-code elimination.

### `Region`, `Param`, `KernelDef`

```text
Region(label, ops)
Param(name, type, attrs)
KernelDef(name, params, body, attrs)
```

Kernel attributes consumed by lowering:

- `max_workgroup_size` (int, default 256) — emitted as `"amdgpu-flat-work-group-size"="64,N"`.
- `waves_per_eu` (int or `(lo, hi)` tuple) — emitted as `"amdgpu-waves-per-eu"`.

## IRBuilder API

`IRBuilder("name")` owns the SSA counter, region stack, parameter list, and `KernelDef`. The full method surface below is verified against `core/ir.py`; entries link to op name and a one-line semantic. Numeric methods accept matching dtypes unless noted.

### Constants

```text
const_i32(value)
const_i64(value)
const_f32(value)
fp16_zero()
zero_vec_f32(n)            # <n x f32>, all zeros
zero_vec_f32_4()           # <4 x f32>, all zeros
zero_vec(elem, n)          # f32 / f16 / bf16
```

### Arithmetic / compare / cast

```text
add, sub, mul, div, mod                    # integer arithmetic
fadd, fsub, fmul, fdiv, fneg               # f32 arithmetic
fmax, fmin                                 # llvm.maxnum / llvm.minnum (f32/f16/bf16)
fcmp(pred, a, b)                           # pred in {olt,ole,ogt,oge,oeq,one,ord,uno}
cmp_lt, cmp_le, cmp_gt, cmp_ge, cmp_eq, cmp_ne   # integer compare (-> i1)
land, lor, lnot                            # bitwise + i1 logic
zext(v, target_int_type)
sext(v, target_int_type)
select(cond, a, b)                         # i1 -> select
masked_select                              # alias for select
sitofp_f32(v)                              # i32 -> f32 (LLVM sitofp)
cast_to_f32(v)                             # f16/bf16 -> f32
cast_f32_to(v, target)                     # f32 -> f16/bf16
trunc_f32_to_f16(v)                        # f32 -> f16 (fptrunc with attrs)
bitcast(v, target)                         # scalar bitcast of equal size
vec_bitcast(v, target_vec_type)            # vector bitcast (e.g. <2xi32> <-> <4xf16>)
clamp_f32(v, lo, hi)                       # fmin(hi, fmax(lo, v)) -> v_med3_f32

# fp/int quantization conversions (one element at a time):
cvt_fp8_to_f32(v)        # llvm.amdgcn.cvt.f32.fp8
cvt_bf8_to_f32(v)        # llvm.amdgcn.cvt.f32.bf8
cvt_f32_to_fp8(v)        # llvm.amdgcn.cvt.pk.fp8.f32 (low byte)
cvt_f32_to_bf8(v)        # llvm.amdgcn.cvt.pk.bf8.f32 (low byte)
cvt_f32_to_i8_sat(v)     # round + saturate to i8
```

### Math intrinsics

LLVM target intrinsics; lowering is f32-centric.

```text
exp2(v)        # llvm.exp2.f32
sqrt(v)        # llvm.sqrt.f32
rsqrt(v)       # llvm.amdgcn.rsq.f32 (single hardware reciprocal-sqrt)
tanh(v)        # llvm.tanh.f32
rcp(v)         # 1/v, lowered to hardware reciprocal
```

### Vector ops

```text
vector_binary(op_name, a, b)
vector_add(a, b)
vector_mul(a, b)
vector_max(a, b)
vector_sum(v)                      # elementwise sum -> scalar
vector_reduce_max(v)               # elementwise max -> scalar
vector_splat(scalar, n)
vector_select(mask, a, b)
```

### GPU thread / block IDs

```text
thread_id_x()
block_id_x(), block_id_y(), block_id_z()
lane_id()                          # llvm.amdgcn.mbcnt.hi(-1, mbcnt.lo(-1, 0))
```

### Compile-time structure

```text
static_for(start, stop, step, body=lambda i: ...)   # emits no scf.for; Python-time unroll
unroll(start[, stop], step=1) -> range              # documents intentional unrolling
static_if(host_bool, body)                          # rejects SSA Value
```

### Memory: global / typed

```text
global_load(ptr, idx, dtype, align=1)
global_load_f16(ptr, idx, align=2)
global_load_bf16(ptr, idx, align=2)
global_load_f32(ptr, idx, align=4)
global_load_i32(ptr, idx, align=4)
global_load_i64(ptr, idx, align=8)
global_load_fp8e4m3(ptr, idx, align=1)
masked_global_load(ptr, idx, mask, other, dtype, align=1)   # clamps false-lane idx to 0
global_store(ptr, idx, value, align=1)
global_load_vN_f16(ptr, idx, n)        # n in {2,4,8}; aligned by default
global_load_vN(ptr, idx, dtype, n)     # f16 or bf16; n in {2,4,8}
global_store_vN(...)                   # vector stores
global_atomic_add_f32(ptr, idx, value) # used by split-K paths
```

### Memory: LDS (`tile.smem_*`)

```text
smem_alloc(elem, shape, name_hint)            # module-level addrspace(3) global
smem_addr_of(smem)                            # i64 LDS base address
smem_ptr_add(lds_addr, byte_off)              # i64 LDS address arithmetic
smem_load_f16(smem, indices)
smem_load_vN_f16(smem, *indices, n in {1,2,4,8})
smem_load_v4_f16(smem, row, col)
smem_load_vN(smem, *indices, dtype, n)        # f16/bf16
smem_store_f16(smem, indices, value)
smem_store_vN_f16(smem, indices, value, n)    # ds_write_b{16,32,64,128}
smem_store_vN(smem, indices, value, n)        # n in {1,2,4,8}
```

### Memory: AMDGPU buffer resources

```text
buffer_rsrc(ptr, num_bytes)                   # llvm.amdgcn.make.buffer.rsrc.p1
                                              #   flags=0x00027000  (TYPE=2, DATA_FMT=4, NUM_FMT=4)
buffer_load_f16(rsrc, voffset, soffset)
buffer_load_vN_f16(rsrc, voffset, soffset, dwords)
buffer_store_f16(...), buffer_store_vN_f16(...)
async_buffer_load_lds(rsrc, smem, voffset, soffset, dwords[, coherency])
async_buffer_load_lds_addr(rsrc, lds_addr_i64, voffset, soffset, dwords[, coherency])
```

`async_buffer_load_lds_addr` is the lane-contiguous DMA primitive consumed by `AsyncTileLoader`. The intrinsic accepts `dwords in {1, 3, 4}` on this target (no 2-dword form). Completion uses the VMEM counter; consumers must issue `s_waitcnt(vmcnt=0)` before reading the destination LDS.

### MFMA atoms (matrix instructions)

```text
mfma_f32_16x16x16_f16(a, b, c)     # <4 x half>, <4 x half>, <4 x float>
mfma_f32_16x16x32_f16(a, b, c)     # <8 x half>, <8 x half>, <4 x float>   (gfx950 only)
mfma_f32_16x16x16_bf16(a, b, c)    # gfx950 lowers via *_1k variant with <4 x i16> operands
mfma_f32_16x16x32_bf16(a, b, c)    # <8 x bfloat>, <8 x bfloat>, <4 x float>
mfma_f32_32x32x8_f16(a, b, c)      # <4 x half>, <4 x half>, <16 x float>
mfma_f32_32x32x16_f16(a, b, c)     # <8 x half>, <8 x half>, <16 x float>  (gfx950 only)
mfma_f32_4x4x4_f16(a, b, c)        # 16 independent 4x4 matmuls per wave
```

The shipped `MFMA_F16_ATOMS` catalog in `helpers/atoms.py` exposes the five f16 variants with full lane mapping and dispatch. See `reference/mfma_atom_catalog.md`.

### Wave / cross-lane

```text
readfirstlane(v)         # llvm.amdgcn.readfirstlane.{i32,i64}; broadcast lane 0
pin_sgpr(v)              # asm volatile("" : "+s"(x)); keep v in SGPR across uses
to_sgpr_u32(v)           # pin_sgpr(readfirstlane(v))
wave_all(pred)           # i32 = 1 iff all lanes' pred != 0
wave_any(pred)           # i32 = 1 iff any lane's pred != 0
wave_ballot(pred)        # i64 mask of lanes satisfying pred
ds_bpermute(addr, data)  # llvm.amdgcn.ds.bpermute  (per-lane source-lane index in bits [7:2])
warp_shuffle_xor(v, lane_xor)   # bpermute with (lane ^ lane_xor) << 2
ds_read_tr16_b64(smem, *indices, dtype=F16)
                         # 16x16 fp16 transpose-read returning <4 x dtype> per lane
                         # exactly the v_mfma_f32_16x16x16_f16 B-operand layout
```

### Synchronization

```text
s_waitcnt(vmcnt=None, lgkmcnt=None, expcnt=None)   # gfx9/10 encoded immarg
sync()                                              # waitcnt vmcnt=0 lgkmcnt=0 + s.barrier
sync_lds_only()                                     # waitcnt lgkmcnt=0 + s.barrier
sync_half_block(...)                                # conditional barrier; dangerous if asymmetric
sched_group_barrier(mask, count, sync_id)
sched_barrier(mask)
s_setprio(priority)
```

Important encoding fact verified by the test suite: gfx950 `s_waitcnt` uses the gfx9/gfx10 layout where `vmcnt` is six bits split across `[3:0]` and `[15:14]`. Asking for `vmcnt=16` must not wrap to `vmcnt(0)`; the lowering clamps `lgkmcnt=16` to 15 (gfx950 LGKM is 4 bits).

### Structured control flow

```text
scf_for(lower, upper, step, body=lambda i: ...)
scf_for_iter(lower, upper, step, init_vars, body=lambda i, *iters: ...)   # loop-carried values
scf_if(cond)                                        # context manager
scf_yield(*values)
```

`scf_for_iter` lowers to header/body/latch/exit basic blocks with phi nodes for the induction variable and every loop-carried value. `scf_yield` records yielded operands; the latch block emits the IV increment and back-edge.

### Parameters and naming

```text
param(name, type, **attrs) -> Value
get_param(name) -> Value
```

Pointer attributes consumed by LLVM lowering:

```text
noalias        -> noalias
readonly       -> readonly
writeonly      -> writeonly + nocapture
align=N        -> align N
dereferenceable=N -> dereferenceable(N)
```

Verified by `test_param_metadata_lowers_to_llvm_arg_attrs` in `python/test/test_ck_dsl.py`.

## IR Printer

`core/ir_print.py::print_ir(kernel)` emits MLIR-style textual IR for inspection and manifest artifacts. It is not parsed back into the compiler. The production input to `core/lower_llvm.py::lower_kernel_to_llvm` is the `KernelDef` object, not this text.

## Conservative Passes

`core/passes.py::optimize_kernel(kernel, max_iter=3)` runs:

- constant folding (integer add/sub/mul/div/mod, bit ops, compare, zext/sext, select);
- common subexpression elimination on pure ops with one result;
- dead-pure-op elimination (recursive into nested regions).

It never removes or reorders side-effecting operations. `PassStats` records `constants_folded`, `common_subexpressions`, `dead_ops_removed`.

## IR Design Rules

- Keep operation names close to the hardware primitive when the primitive matters for performance.
- Put shape-specific algorithm logic in instance builders, not in helpers or core.
- Put repeated CK Tile-style tile/memory/MFMA patterns in `helpers/`.
- Keep runtime predicates as SSA values; refuse Python-bool-on-SSA.
- Prefer explicit validity predicates from `transforms.TensorDescriptor.offset` over implicit assumptions about tails.
- Preserve enough high-level operation identity that generated LLVM/ISA can be inspected against intent (see `analysis/ir.py`, `analysis/isa.py`).

## Common Pitfalls

- Calling `b.and_/or_/not_`: those names do **not** exist. The methods are `b.land`, `b.lor`, `b.lnot`.
- Asking for `async_buffer_load_lds` with `dwords=2`: unsupported on this LLVM target. Use 1, 3, or 4.
- `b.cast_to_f32` only accepts `f16` / `bf16` / `f32` inputs.
- `b.cast_f32_to(target)` only accepts `f32` input and `f16` / `bf16` / `f32` target.
- `b.sitofp_f32(v)` requires `i32` input; signed-int-to-f32 for other integer widths must be widened first.
- `b.ds_bpermute(addr, data)` requires both operands to be `i32`. For half/bf16 payloads, pack to `i32` first.
- `b.s_waitcnt(vmcnt=16)` must not wrap. The encoding spans `[3:0]` and `[15:14]`; LGKM clamps at 15 on gfx950.
- Pinning a wave-uniform value with `b.pin_sgpr` is meaningful only after `readfirstlane`. Use `b.to_sgpr_u32(v)` to combine both.
