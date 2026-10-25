# IR Op Vocabulary

Complete reference of operations recognized by `core/ir.py` and lowered to AMDGPU LLVM IR. The native Python lowerer (`core/lower_llvm.py`) and the C++ engine (`cpp/`, reached through the `rocke_engine` extension) are two interchangeable engines that emit byte-identical IR for these ops; `core/backend.py` selects between them. Verified against the IR source. Each entry lists the operation name (as it appears in `print_ir`), the IRBuilder method, and the LLVM emission summary.

## Arithmetic

| Op name                | Builder                           | LLVM emission                          |
|------------------------|-----------------------------------|----------------------------------------|
| `arith.constant`       | `const_i32`, `const_i64`, `const_f32`, `const_f64`, `fp16_zero` | constant value |
| `arith.constant_vec`   | `zero_vec_f32`, `zero_vec`        | `zeroinitializer` vector               |
| `arith.add`            | `add`, `add_nuw`                  | `add nsw` (`add`), `add nuw` (`add_nuw`, attr `wrap = "nuw"`) |
| `arith.sub`            | `sub`                             | `sub`                                  |
| `arith.mul`            | `mul`                             | `mul`                                  |
| `arith.div`            | `div`                             | `sdiv`                                 |
| `arith.mod`            | `mod`                             | `srem`                                 |
| `arith.shl`            | (internal, used by `warp_shuffle_xor`) | `shl`                              |
| `arith.fadd`           | `fadd`                            | `fadd`                                 |
| `arith.fsub`           | `fsub`                            | `fsub`                                 |
| `arith.fmul`           | `fmul`                            | `fmul`                                 |
| `arith.fdiv`           | `fdiv`                            | `fdiv`                                 |
| `arith.fneg`           | `fneg`                            | `fneg`                                 |
| `arith.fmax`           | `fmax`                            | `llvm.maxnum.{f64,f32,f16,bf16}`       |
| `arith.fmin`           | `fmin`                            | `llvm.minnum.{f64,f32,f16,bf16}`       |
| `arith.fdot2`          | `fdot2`                           | `llvm.amdgcn.fdot2` (f16) / `llvm.amdgcn.fdot2.f32.bf16` (bf16), clamp off; f32 accumulator |
| `arith.cmp`            | `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `cmp_eq`, `cmp_ne`, `cmp_ult` | `icmp {slt,sle,sgt,sge,eq,ne,ult}` |
| `arith.assume_range`   | (private `_assume_range`, for checked callers only) | nothing: uses of the result print the marked value; the attrs record a range and congruence for the tiling layer's range walk |
| `arith.fcmp`           | `fcmp(pred, ...)`                 | `fcmp {olt,ole,ogt,oge,oeq,one,ord,uno}` |
| `arith.and`            | `land`                            | `and`                                  |
| `arith.or`             | `lor`                             | `or`                                   |
| `arith.xor`            | (internal)                        | `xor`                                  |
| `arith.not`            | `lnot`                            | `xor` against all-ones                 |
| `arith.zext`           | `zext`                            | `zext`                                 |
| `arith.sext`           | `sext`                            | `sext`                                 |
| `arith.select`         | `select`, `masked_select`         | `select`                               |
| `arith.bitcast`        | `bitcast`                         | `bitcast`                              |
| `arith.cast_to_f32`    | `cast_to_f32`                     | `fpext` (f16/bf16 -> f32)              |
| `arith.cast_f32_to`    | `cast_f32_to`                     | `fptrunc` to f16/bf16                  |
| `arith.trunc_f32_to_f16` | `trunc_f32_to_f16`              | `fptrunc` (vec-packed in some emissions) |
| `arith.sitofp_f32`     | `sitofp_f32`                      | `sitofp i32 to float`                  |
| `arith.cvt_f32_to_f64` | `cvt_f32_to_f64`                  | `fpext float to double` (exact)        |
| `arith.cvt_f64_to_f32` | `cvt_f64_to_f32`                  | `fptrunc double to float` (round to nearest even) |
| `arith.cvt_fp8_to_f32` | `cvt_fp8_to_f32`                  | `llvm.amdgcn.cvt.f32.fp8`              |
| `arith.cvt_bf8_to_f32` | `cvt_bf8_to_f32`                  | `llvm.amdgcn.cvt.f32.bf8`              |
| `arith.cvt_f32_to_fp8` | `cvt_f32_to_fp8`                  | `llvm.amdgcn.cvt.pk.fp8.f32` (low byte) |
| `arith.cvt_f32_to_bf8` | `cvt_f32_to_bf8`                  | `llvm.amdgcn.cvt.pk.bf8.f32` (low byte) |
| `arith.cvt_f32_to_i8_sat` | `cvt_f32_to_i8_sat`            | round-to-nearest + saturating cast     |

## Math

| Op name        | Builder      | LLVM intrinsic                  |
|----------------|--------------|---------------------------------|
| `math.exp2`    | `exp2`       | `llvm.exp2.f32`; f64: expansion |
| `math.exp`     | `exp`        | `llvm.exp.f32`; f64: expansion  |
| `math.log2`    | `log2`       | `llvm.log2.f32`; f64: expansion |
| `math.log`     | `log`        | `llvm.log.f32`; f64: expansion  |
| `math.sqrt`    | `sqrt`       | `llvm.sqrt.f32` / `llvm.sqrt.f64` |
| `math.rsqrt`   | `rsqrt`      | `llvm.amdgcn.rsq.f32`; f64: `1 / llvm.sqrt.f64` |
| `math.tanh`    | `tanh`       | piecewise f32 expansion: OCML polynomial for `abs(x) < 0.625`, otherwise `exp2` + reciprocal; bitwise sign restoration. f64: expansion |
| `math.rcp`     | `rcp`        | `1.0 / v` (hardware reciprocal) |

**f64.** The AMDGPU backend has no f64 `exp` / `log` / `tanh`, so the f64 forms of
`exp`, `exp2`, `log`, `log2`, `tanh`, `sqrt` and `rsqrt` expand to fixed straight-line
instruction programs defined once in `rocke/core/f64_math.py` (the C++ engine walks the
same programs from `cpp/core/lower_llvm/f64_math_programs.inc`, generated by
`tools/gen_f64_math.py`): `exp`/`exp2` reduce with `fma` (Cody-Waite) and scale with
`llvm.ldexp`; `log`/`log2` split with `llvm.frexp` (subnormals pre-scaled) and use the
`atanh` series; `tanh` is `u / (u + 2)` with `u = expm1(2|x|)`. Every special value
is handled with `select`, so the programs are branch-free. Accuracy, measured on the
emitted instructions: `exp`, `exp2`, `log`, `log2`, `tanh` and `rsqrt` within 1 ulp,
`sqrt` correctly rounded. `fma` and `vector_fma` lower to
`llvm.fma.f64` / `llvm.fma.v<N>f64` (always fused, unlike the `fmuladd` of the narrow
types), `fmax3` / `fmin3` to `llvm.maxnum.f64` / `llvm.minnum.f64` chains. HIP uses the
double-precision device library (`exp`, `log2`, `tanh`, `fma`, ...), so an f64 math result
can differ in the last bits between a kernel lowered to LLVM IR and the same kernel
lowered to HIP. `rcp_fast` and
`exp2_fast` (single-instruction approximations) reject f64; `exp2_fast` has no HIP
lowering for any dtype. `exp`, `log` and `tanh` accept f32 and f64 only.

`clamp_f32(v, lo, hi)` is `fmin(hi, fmax(lo, v))` — folds to `v_med3_f32`.

## Vector

| Op name              | Builder                  | LLVM emission                  |
|----------------------|--------------------------|--------------------------------|
| `vector.add`         | `vector_add`             | elementwise vector `fadd`/`add` |
| `vector.mul`         | `vector_mul`             | elementwise vector multiply     |
| `vector.max`         | `vector_max`             | elementwise vector max          |
| `vector.sum`         | `vector_sum`             | `llvm.vector.reduce.fadd.*`     |
| `vector.reduce_max`  | `vector_reduce_max`      | `llvm.vector.reduce.fmax.*`     |
| `vector.splat`       | `vector_splat`           | `insertelement` + `shufflevector` |
| `vector.select`      | `vector_select`          | vector `select`                 |
| `vector.bitcast`     | `vec_bitcast`            | `bitcast` between equal-size vectors |

## GPU / Lane

| Op name             | Builder            | LLVM emission                        |
|---------------------|--------------------|--------------------------------------|
| `gpu.thread_id`     | `thread_id_x`      | `llvm.amdgcn.workitem.id.x()`        |
| `gpu.block_id`      | `block_id_x/y/z`   | `llvm.amdgcn.workgroup.id.{x,y,z}()` |
| `tile.lane_id`      | `lane_id`          | `mbcnt.hi(-1, mbcnt.lo(-1, 0))`      |
| `tile.readfirstlane`| `readfirstlane`    | `llvm.amdgcn.readfirstlane.{i32,i64}` |
| `tile.pin_sgpr`     | `pin_sgpr`         | `asm volatile("" : "+s"(x))`         |
| `tile.wave_all`     | `wave_all`         | ballot + compare                     |
| `tile.wave_any`     | `wave_any`         | ballot + compare                     |
| `tile.wave_ballot`  | `wave_ballot`      | `llvm.amdgcn.ballot.i64`             |
| `tile.ds_bpermute`  | `ds_bpermute`      | `llvm.amdgcn.ds.bpermute`            |

`warp_shuffle_xor(v, lane_xor)` is a helper that combines `xor` + `ds_bpermute` and bitcasts as needed.

## Memory

### Global

| Op name                       | Builder                       | LLVM emission |
|-------------------------------|-------------------------------|---------------|
| `memref.global_load`          | `global_load_f16`             | typed `load addrspace(1)` |
| `memref.global_load_typed`    | `global_load`, `global_load_i8`, `global_load_i16`, `global_load_i32`, `global_load_i64`, `global_load_fp8e4m3`, `global_load_bf8e5m2`, `global_load_bf16`, `global_load_f32`, `global_load_f64` | typed `load addrspace(1)` |
| `memref.global_load_vN`       | `global_load_vN`, `global_load_vN_f16` | vector `load addrspace(1)` (n in {2,4,8}) |
| `memref.global_store_typed`   | `global_store`                | typed `store addrspace(1)` |
| `memref.global_store_vN`      | `global_store_vN`             | vector `store addrspace(1)` |
| `memref.global_atomic_add_f32`| `global_atomic_add_f32`       | `atomicrmw fadd addrspace(1)` |
| `memref.fence`               | `fence`                       | `fence syncscope(...) <ordering>` |
| `memref.global_flag_store`   | `global_flag_store`           | `store atomic i32 ... syncscope(...) <ordering>` |
| `memref.global_flag_wait_eq` | `global_flag_wait_eq`         | spin loop around `load atomic i32 ... acquire` |

`masked_global_load(ptr, idx, mask, other, dtype)` clamps the false-lane index to 0 before loading, then selects `other` afterward.

### LDS

| Op name                | Builder                  | LLVM emission                                  |
|------------------------|--------------------------|------------------------------------------------|
| `tile.smem_alloc`      | `smem_alloc`             | module-level `addrspace(3) global`             |
| `tile.smem_addr_of`    | `smem_addr_of`           | `ptrtoint` of LDS base                         |
| `tile.smem_ptr_add`    | `smem_ptr_add`           | `add i64` over LDS addresses                   |
| `tile.smem_load`       | `smem_load_f16`          | scalar `load addrspace(3)`                     |
| `tile.smem_load_v4`    | `smem_load_v4_f16`       | `<4 x half>` load                              |
| `tile.smem_load_vN`    | `smem_load_vN_f16`, `smem_load_vN` | `<N x dtype>` load (n in {1,2,4,8})  |
| `tile.smem_store`      | `smem_store_f16`         | scalar `store addrspace(3)`                    |
| `tile.smem_store_vN`   | `smem_store_vN_f16`, `smem_store_vN` | `<N x dtype>` store                |
| `tile.ds_read_tr16_b64`| `ds_read_tr16_b64`       | `llvm.amdgcn.ds.read.tr16.b64`                 |

### Buffer (addrspace(8))

| Op name                          | Builder                          | LLVM emission                                                |
|----------------------------------|----------------------------------|--------------------------------------------------------------|
| `tile.buffer_rsrc`               | `buffer_rsrc`                    | `llvm.amdgcn.make.buffer.rsrc.p1(ptr, 0, num_bytes, 159744)` |
| `tile.buffer_load_f16`           | `buffer_load_f16`                | `raw.ptr.buffer.load.i32` + truncate / bitcast               |
| `tile.buffer_load_vN_f16`        | `buffer_load_vN_f16`             | `raw.ptr.buffer.load.{i32,v2i32,v4i32}` + `vec_bitcast`      |
| `tile.buffer_store_f16`          | `buffer_store_f16`               | `raw.ptr.buffer.store.i16`                                   |
| `tile.buffer_store_vN_f16`       | `buffer_store_vN_f16`            | `raw.ptr.buffer.store.{i32,v2i32,v4i32}`                     |
| `tile.async_buffer_load_lds`     | `async_buffer_load_lds`          | `llvm.amdgcn.raw.ptr.buffer.load.lds`                        |
| `tile.async_buffer_load_lds_addr`| `async_buffer_load_lds_addr`     | same as above; explicit LDS i64 address                       |

## MFMA

| Op name                          | Operands (per-lane)              | Output (per-lane) |
|----------------------------------|----------------------------------|--------------------|
| `tile.mfma_f32_16x16x16_f16`     | `<4xhalf>`, `<4xhalf>`, `<4xfloat>` | `<4xfloat>`     |
| `tile.mfma_f32_16x16x32_f16`     | `<8xhalf>`, `<8xhalf>`, `<4xfloat>` | `<4xfloat>`     |
| `tile.mfma_f32_16x16x16_bf16`    | `<4xi16>` (bf16 bitcast), same, `<4xfloat>` | `<4xfloat>` |
| `tile.mfma_f32_16x16x32_bf16`    | `<8xbfloat>`, same, `<4xfloat>`  | `<4xfloat>`        |
| `tile.mfma_f32_32x32x8_f16`      | `<4xhalf>`, `<4xhalf>`, `<16xfloat>` | `<16xfloat>`   |
| `tile.mfma_f32_32x32x16_f16`     | `<8xhalf>`, `<8xhalf>`, `<16xfloat>` | `<16xfloat>`   |
| `tile.mfma_f32_4x4x4_f16`        | `<4xhalf>`, `<4xhalf>`, `<4xfloat>` | `<4xfloat>`     |
| `tile.mfma_f32_4x4x4_bf16`       | `<4xi16>` (bf16 bitcast), same, `<4xfloat>` | `<4xfloat>` |

All MFMAs take three literal immarg constants (`cbsz=0, abid=0, blgp=0`) at the LLVM level.

## Synchronization / Scheduling

| Op name                  | Builder                | LLVM emission                                                |
|--------------------------|------------------------|--------------------------------------------------------------|
| `tile.s_waitcnt`         | `s_waitcnt`            | `llvm.amdgcn.s.waitcnt(i32 encoded_imm)`                     |
| `tile.sync`              | `sync`                 | `s_waitcnt(vmcnt=0, lgkmcnt=0)` + `s.barrier`                |
| `tile.sync_lds_only`     | `sync_lds_only`        | `s_waitcnt(lgkmcnt=0)` + `s.barrier`                         |
| `tile.sync_half_block`   | `sync_half_block`      | conditional barrier (correctness depends on caller control flow) |
| `tile.sched_group_barrier` | `sched_group_barrier` | `llvm.amdgcn.sched.group.barrier(mask, count, sync_id)`      |
| `tile.sched_barrier`     | `sched_barrier`        | `llvm.amdgcn.sched.barrier(mask)`                            |
| `tile.s_setprio`         | `s_setprio`            | `llvm.amdgcn.s.setprio(i16 prio)`                            |

`s_waitcnt` immarg encoding on gfx950 spans both `[3:0]` and `[15:14]` for vmcnt; LGKM is 4 bits and clamps at 15.

## Control Flow

| Op name      | Builder                | Lowering                                              |
|--------------|------------------------|-------------------------------------------------------|
| `scf.for`    | `scf_for`              | header / body / latch / exit blocks + phi (induction) |
| `scf.for_iter` | `scf_for_iter`       | header / body / latch / exit + phi for each iter_var  |
| `scf.if`     | `scf_if` (ctx manager) | conditional branch + then block + join block          |
| `scf.yield`  | `scf_yield(*values)`   | records yielded values; consumed by enclosing for/if  |
| `cf.return`  | (auto-emitted)         | `ret void`                                            |

`static_for` and `unroll` are Python-time helpers that emit no `scf.for`; they execute their body in Python and emit straight-line IR.

## Compile-Time Helpers

These are not IR ops; they affect IR construction:

```text
static_for(start, stop, step, body=lambda i: ...)
unroll(stop) -> range
unroll(start, stop, step) -> range
static_if(host_bool, body)             # raises TypeError on SSA Value
```

## See Also

- `ir_lowering/ir_model.md` for the IRBuilder API by category.
- `ir_lowering/lowering_pipeline.md` for the op-to-LLVM mapping.
- `ir_lowering/backend_details.md` for purity classification and waitcnt encoding.

`IRBuilder.cvt_f32_to_tf32` / `rocke_b_cvt_f32_to_tf32` is a scalar F32-to-TF32
RNE recipe built from existing integer arithmetic, select, and bitcast ops; it
introduces no serialized opcode. It quiets NaNs and retains their upper payload
bits. `bitcast` to logical TF32 only reinterprets the payload. Logical TF32 uses
I32 storage; general arithmetic requires explicitly reinterpreting it as F32.
See [TF32 numerics](../../python/rocke/examples/gfx942/tf32_numerics/README.md).
