# IR Op Vocabulary

Complete reference of operations recognized by `core/ir.py` and lowered by `core/lower_llvm.py`. Verified against the IR source. Each entry lists the operation name (as it appears in `print_ir`), the IRBuilder method, and the LLVM emission summary.

## Arithmetic

| Op name                | Builder                           | LLVM emission                          |
|------------------------|-----------------------------------|----------------------------------------|
| `arith.constant`       | `const_i32`, `const_i64`, `const_f32`, `fp16_zero` | constant value |
| `arith.constant_vec`   | `zero_vec_f32`, `zero_vec`        | `zeroinitializer` vector               |
| `arith.add`            | `add`                             | `add` (integer)                        |
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
| `arith.fmax`           | `fmax`                            | `llvm.maxnum.{f32,f16,bf16}`           |
| `arith.fmin`           | `fmin`                            | `llvm.minnum.{f32,f16,bf16}`           |
| `arith.cmp`            | `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `cmp_eq`, `cmp_ne` | `icmp {slt,sle,sgt,sge,eq,ne}` |
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
| `arith.cvt_fp8_to_f32` | `cvt_fp8_to_f32`                  | `llvm.amdgcn.cvt.f32.fp8`              |
| `arith.cvt_bf8_to_f32` | `cvt_bf8_to_f32`                  | `llvm.amdgcn.cvt.f32.bf8`              |
| `arith.cvt_f32_to_fp8` | `cvt_f32_to_fp8`                  | `llvm.amdgcn.cvt.pk.fp8.f32` (low byte) |
| `arith.cvt_f32_to_bf8` | `cvt_f32_to_bf8`                  | `llvm.amdgcn.cvt.pk.bf8.f32` (low byte) |
| `arith.cvt_f32_to_i8_sat` | `cvt_f32_to_i8_sat`            | round-to-nearest + saturating cast     |

## Math

| Op name        | Builder      | LLVM intrinsic                  |
|----------------|--------------|---------------------------------|
| `math.exp2`    | `exp2`       | `llvm.exp2.f32`                 |
| `math.sqrt`    | `sqrt`       | `llvm.sqrt.f32`                 |
| `math.rsqrt`   | `rsqrt`      | `llvm.amdgcn.rsq.f32`           |
| `math.tanh`    | `tanh`       | `llvm.tanh.f32`                 |
| `math.rcp`     | `rcp`        | `1.0 / v` (hardware reciprocal) |

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
| `memref.global_load_typed`    | `global_load`, `global_load_f32`, `global_load_i32`, `global_load_i64`, `global_load_bf16`, `global_load_fp8e4m3` | typed `load addrspace(1)` |
| `memref.global_load_vN`       | `global_load_vN`, `global_load_vN_f16` | vector `load addrspace(1)` (n in {2,4,8}) |
| `memref.global_store_typed`   | `global_store`                | typed `store addrspace(1)` |
| `memref.global_store_vN`      | `global_store_vN`             | vector `store addrspace(1)` |
| `memref.global_atomic_add_f32`| `global_atomic_add_f32`       | `atomicrmw fadd addrspace(1)` |

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
