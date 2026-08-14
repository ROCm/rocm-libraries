// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * src/lower_llvm_data.c -- module-level CONST DATA tables for the C99 port of
 * rocke.core.lower_llvm. Defines the externs declared in
 * rocke/lower_llvm_internal.h:
 *
 *   - ROCKE_LL_DATALAYOUT / ROCKE_LL_TRIPLE          (Python _DATALAYOUT / _TRIPLE)
 *   - ROCKE_LL_INTRINSIC_DECLS[]   (+ _COUNT)      (Python _INTRINSIC_DECLS)
 *   - ROCKE_LL_INTRINSIC_DECLS_LLVM22_OVERRIDES[]  (+ _COUNT)
 *                                                (Python ..._LLVM22_OVERRIDES)
 *
 * The decl table is INSERTION-ORDERED exactly like the Python dict; that order
 * drives finalize()'s emit order. Transcribed verbatim from
 * rocke/core/lower_llvm.py.
 */

#include "rocke/arena.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm_internal.h"
#include "rocke/strbuf.h"
#include "rocke/vec.h"

namespace ckc
{

/* ---------------------------------------------------------------------- */
/* datalayout / triple (Python _DATALAYOUT_LLVM20 / _DATALAYOUT_LLVM22 /   */
/* _DATALAYOUT_LLVM23 / _TRIPLE). The AMDGPU datalayout is FLAVOR-KEYED:    */
/* two fields drift across LLVM flavors:                                    */
/*   * buffer-resource address space (p8, the 128-bit buffer descriptor --   */
/*     NOT the p7 fat pointer) gained an index-width field:                  */
/*       LLVM 20 (ROCm 7.0/7.1):  e-...-p8:128:128-...                       */
/*       LLVM 22 (ROCm 7.2):      e-...-p8:128:128:128:48-...                */
/*   * ELF symbol-mangling spec (m:e): absent in LLVM 20 and LLVM 22,        */
/*     present in LLVM 23 (ROCm 7.13+, AMD clang 23.0.0git):                 */
/*       LLVM 20 / 22:  e-p:64:64-...                                        */
/*       LLVM 23:       e-m:e-p:64:64-...                                    */
/* Pick via rocke_ll_datalayout_for_flavor, mirroring _datalayout_for_flavor.*/
/* ---------------------------------------------------------------------- */

const char* const ROCKE_LL_DATALAYOUT_LLVM20
    = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
      "-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32"
      "-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048"
      "-n32:64-S32-A5-G1-ni:7:8:9";

const char* const ROCKE_LL_DATALAYOUT_LLVM22
    = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
      "-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32"
      "-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048"
      "-n32:64-S32-A5-G1-ni:7:8:9";

/* LLVM 23 (ROCm 7.13+): re-derived on an LLVM 23 host and found to drift from
 * LLVM 22 by one field -- it emits the ELF symbol-mangling spec m:e that LLVM 22
 * omits. The p8-indexed layout is otherwise identical (Python
 * _DATALAYOUT_LLVM23). */
const char* const ROCKE_LL_DATALAYOUT_LLVM23
    = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
      "-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32"
      "-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048"
      "-n32:64-S32-A5-G1-ni:7:8:9";

/* Back-compat alias: callers that have not yet been flavor-threaded see the
 * LLVM20 form (the historical hardcoded value). New code keys on the flavor
 * via rocke_ll_datalayout_for_flavor / rocke_isa_datalayout. */
const char* const ROCKE_LL_DATALAYOUT = ROCKE_LL_DATALAYOUT_LLVM20;

const char* const ROCKE_LL_TRIPLE = "amdgcn-amd-amdhsa";

/* Python _datalayout_for_flavor: LLVM20 returns the legacy p8 layout, LLVM23
 * returns its own m:e form, and anything else (incl. an unexpected value)
 * degrades to the modern LLVM22 form. */
const char* rocke_ll_datalayout_for_flavor(rocke_llvm_flavor_t flavor)
{
    if(flavor == ROCKE_LLVM_FLAVOR_LLVM20)
        return ROCKE_LL_DATALAYOUT_LLVM20;
    if(flavor == ROCKE_LLVM_FLAVOR_LLVM23)
        return ROCKE_LL_DATALAYOUT_LLVM23;
    return ROCKE_LL_DATALAYOUT_LLVM22;
}

/* ---------------------------------------------------------------------- */
/* The base (LLVM20) intrinsic-declaration table (Python _INTRINSIC_DECLS) */
/* Insertion order preserved.                                              */
/* ---------------------------------------------------------------------- */

const rocke_ll_decl_t ROCKE_LL_INTRINSIC_DECLS[] = {
    {"workitem.x", "declare i32 @llvm.amdgcn.workitem.id.x()"},
    {"workitem.y", "declare i32 @llvm.amdgcn.workitem.id.y()"},
    {"workitem.z", "declare i32 @llvm.amdgcn.workitem.id.z()"},
    {"workgroup.x", "declare i32 @llvm.amdgcn.workgroup.id.x()"},
    {"workgroup.y", "declare i32 @llvm.amdgcn.workgroup.id.y()"},
    {"workgroup.z", "declare i32 @llvm.amdgcn.workgroup.id.z()"},
    {"s.barrier", "declare void @llvm.amdgcn.s.barrier()"},
    {"s.wait.dscnt", "declare void @llvm.amdgcn.s.wait.dscnt(i16)"},
    {"s.wait.loadcnt", "declare void @llvm.amdgcn.s.wait.loadcnt(i16)"},
    {"s.wait.asynccnt", "declare void @llvm.amdgcn.s.wait.asynccnt(i16 immarg)"},
    {"global.load.async.to.lds.b32",
     "declare void @llvm.amdgcn.global.load.async.to.lds.b32(ptr addrspace(1) nocapture, ptr "
     "addrspace(3) nocapture, i32 immarg, i32 immarg)"},
    {"global.load.async.to.lds.b64",
     "declare void @llvm.amdgcn.global.load.async.to.lds.b64(ptr addrspace(1) nocapture, ptr "
     "addrspace(3) nocapture, i32 immarg, i32 immarg)"},
    {"global.load.async.to.lds.b128",
     "declare void @llvm.amdgcn.global.load.async.to.lds.b128(ptr addrspace(1) nocapture, ptr "
     "addrspace(3) nocapture, i32 immarg, i32 immarg)"},
    {"exp2.f32", "declare float @llvm.exp2.f32(float)"},
    {"amdgcn.exp2.f32", "declare float @llvm.amdgcn.exp2.f32(float)"},
    {"log2.f32", "declare float @llvm.log2.f32(float)"},
    {"sqrt.f32", "declare float @llvm.sqrt.f32(float)"},
    {"rsqrt.f32", "declare float @llvm.amdgcn.rsq.f32(float)"},
    {"rcp.f32", "declare float @llvm.amdgcn.rcp.f32(float)"},
    {"tanh.f32", "declare float @llvm.tanh.f32(float)"},
    {"maxnum.f32", "declare float @llvm.maxnum.f32(float, float)"},
    {"maxnum.f16", "declare half @llvm.maxnum.f16(half, half)"},
    {"maxnum.bf16", "declare bfloat @llvm.maxnum.bf16(bfloat, bfloat)"},
    {"minnum.f32", "declare float @llvm.minnum.f32(float, float)"},
    {"minnum.f16", "declare half @llvm.minnum.f16(half, half)"},
    {"minnum.bf16", "declare bfloat @llvm.minnum.bf16(bfloat, bfloat)"},
    {"fabs.f32", "declare float @llvm.fabs.f32(float)"},
    {"fabs.f16", "declare half @llvm.fabs.f16(half)"},
    {"fabs.bf16", "declare bfloat @llvm.fabs.bf16(bfloat)"},
    {"fmuladd.f32", "declare float @llvm.fmuladd.f32(float, float, float)"},
    {"fmuladd.f16", "declare half @llvm.fmuladd.f16(half, half, half)"},
    {"fmuladd.bf16", "declare bfloat @llvm.fmuladd.bf16(bfloat, bfloat, bfloat)"},
    {"fmuladd.v2f32",
     "declare <2 x float> @llvm.fmuladd.v2f32(<2 x float>, <2 x float>, <2 x float>)"},
    {"fmuladd.v4f32",
     "declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>)"},
    {"fmuladd.v8f32",
     "declare <8 x float> @llvm.fmuladd.v8f32(<8 x float>, <8 x float>, <8 x float>)"},
    {"fmuladd.v16f32",
     "declare <16 x float> @llvm.fmuladd.v16f32(<16 x float>, <16 x float>, <16 x float>)"},
    {"fmuladd.v2f16", "declare <2 x half> @llvm.fmuladd.v2f16(<2 x half>, <2 x half>, <2 x half>)"},
    {"fmuladd.v4f16", "declare <4 x half> @llvm.fmuladd.v4f16(<4 x half>, <4 x half>, <4 x half>)"},
    {"fmuladd.v8f16", "declare <8 x half> @llvm.fmuladd.v8f16(<8 x half>, <8 x half>, <8 x half>)"},
    {"fmuladd.v2bf16",
     "declare <2 x bfloat> @llvm.fmuladd.v2bf16(<2 x bfloat>, <2 x bfloat>, <2 x bfloat>)"},
    {"fmuladd.v4bf16",
     "declare <4 x bfloat> @llvm.fmuladd.v4bf16(<4 x bfloat>, <4 x bfloat>, <4 x bfloat>)"},
    {"fmuladd.v8bf16",
     "declare <8 x bfloat> @llvm.fmuladd.v8bf16(<8 x bfloat>, <8 x bfloat>, <8 x bfloat>)"},
    {"wmma.f32.16x16x16.f16",
     "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16(<16 x half>, <16 x "
     "half>, <8 x float>)"},
    {"wmma.f32.16x16x16.bf16",
     "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16(<16 x i16>, <16 x i16>, "
     "<8 x float>)"},
    {"wmma.i32.16x16x16.iu8",
     "declare <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32(i1, <4 x i32>, i1, <4 x "
     "i32>, <8 x i32>, i1)"},
    {"wmma.i32.16x16x16.iu4",
     "declare <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.v2i32(i1, <2 x i32>, i1, <2 x "
     "i32>, <8 x i32>, i1)"},
    {"wmma.gfx12.f32.16x16x16.f16",
     "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half>, <8 x half>, "
     "<8 x float>)"},
    {"wmma.gfx12.f32.16x16x16.bf16",
     "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8i16(<8 x i16>, <8 x i16>, <8 "
     "x float>)"},
    {"wmma.gfx1250.f32.16x16x32.f16",
     "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.f16.v8f32.v16f16(i1 immarg, <16 x half>, "
     "i1 immarg, <16 x half>, i16 immarg, <8 x float>, i1 immarg, i1 immarg)"},
    {"wmma.gfx1250.f32.16x16x32.bf16",
     "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(i1 immarg, <16 x "
     "bfloat>, i1 immarg, <16 x bfloat>, i16 immarg, <8 x float>, i1 immarg, i1 immarg)"},
    {"wmma.gfx1250.f32.16x16x64.fp8.fp8",
     "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8.v8f32.v8i32(<8 x i32>, <8 x i32>, "
     "i16 immarg, <8 x float>, i1 immarg, i1 immarg)"},
    {"wmma.gfx1250.f32.16x16x64.fp8.bf8",
     "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.fp8.bf8.v8f32.v8i32(<8 x i32>, <8 x i32>, "
     "i16 immarg, <8 x float>, i1 immarg, i1 immarg)"},
    {"wmma.gfx1250.f32.16x16x64.bf8.fp8",
     "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.bf8.fp8.v8f32.v8i32(<8 x i32>, <8 x i32>, "
     "i16 immarg, <8 x float>, i1 immarg, i1 immarg)"},
    {"wmma.gfx1250.f32.16x16x64.bf8.bf8",
     "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.bf8.bf8.v8f32.v8i32(<8 x i32>, <8 x i32>, "
     "i16 immarg, <8 x float>, i1 immarg, i1 immarg)"},
    {"mfma.f32.16x16x16f16",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half>, <4 x half>, <4 x float>, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.16x16x32.f16",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half>, <8 x half>, <4 x float>, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.16x16x16bf16.1k",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16>, <4 x i16>, <4 x float>, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.16x16x32.bf16",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat>, <8 x bfloat>, <4 x "
     "float>, i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.32x32x8f16",
     "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half>, <4 x half>, <16 x float>, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.32x32x8bf16.1k",
     "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x8bf16.1k(<4 x i16>, <4 x i16>, <16 x "
     "float>, i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.32x32x16.f16",
     "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16(<8 x half>, <8 x half>, <16 x "
     "float>, i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.16x16x4f32",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float, float, <4 x float>, i32 immarg, "
     "i32 immarg, i32 immarg)"},
    {"mfma.f32.32x32x2f32",
     "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x2f32(float, float, <16 x float>, i32 "
     "immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.4x4x4f16",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x4f16(<4 x half>, <4 x half>, <4 x float>, i32 "
     "immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.16x16x32.fp8.fp8",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(<2 x i32>, <2 x i32>, <4 x "
     "float>, i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.16x16x32.bf8.bf8",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8(<2 x i32>, <2 x i32>, <4 x "
     "float>, i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.32x32x16.fp8.fp8",
     "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.fp8.fp8(<2 x i32>, <2 x i32>, <16 x "
     "float>, i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.32x32x16.bf8.bf8",
     "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.bf8(<2 x i32>, <2 x i32>, <16 x "
     "float>, i32 immarg, i32 immarg, i32 immarg)"},
    {"readfirstlane.i32", "declare i32 @llvm.amdgcn.readfirstlane.i32(i32)"},
    {"readfirstlane.i64", "declare i64 @llvm.amdgcn.readfirstlane.i64(i64)"},
    {"ballot.i64", "declare i64 @llvm.amdgcn.ballot.i64(i1)"},
    {"ds.bpermute", "declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)"},
    {"update.dpp.i32",
     "declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32 immarg, i32 immarg, i32 immarg, i1 "
     "immarg)"},
    {"global.atomic.fadd.v2bf16",
     "declare <2 x bfloat> @llvm.amdgcn.global.atomic.fadd.v2bf16.p1("
     "ptr addrspace(1), <2 x bfloat>)"},
    {"global.atomic.fadd.v2f16",
     "declare <2 x half> @llvm.amdgcn.global.atomic.fadd.v2f16.p1("
     "ptr addrspace(1), <2 x half>)"},
    {"mbcnt.lo", "declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32)"},
    {"mbcnt.hi", "declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32)"},
    {"ds.read.tr16.b64", "declare <4 x i16> @llvm.amdgcn.ds.read.tr16.b64(ptr addrspace(3))"},
    {"ds.read.tr16.b128", "declare <8 x i16> @llvm.amdgcn.ds.read.tr16.b128(ptr addrspace(3))"},
    {"ds.load.tr16.b128.v8bf16",
     "declare <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3))"},
    {"ds.load.tr16.b128.v8f16",
     "declare <8 x half> @llvm.amdgcn.ds.load.tr16.b128.v8f16(ptr addrspace(3))"},
    {"iglp.opt", "declare void @llvm.amdgcn.iglp.opt(i32 immarg)"},
    {"sched.barrier", "declare void @llvm.amdgcn.sched.barrier(i32 immarg)"},
    {"sched.group.barrier",
     "declare void @llvm.amdgcn.sched.group.barrier(i32 immarg, i32 immarg, i32 immarg)"},
    {"s.setprio", "declare void @llvm.amdgcn.s.setprio(i16 immarg)"},
    {"s.waitcnt", "declare void @llvm.amdgcn.s.waitcnt(i32 immarg)"},
    {"make.buffer.rsrc.p1",
     "declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p1(ptr addrspace(1) nocapture "
     "readnone, i16, i32, i32)"},
    {"raw.ptr.buffer.load.lds",
     "declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) nocapture readonly, ptr "
     "addrspace(3) nocapture, i32, i32, i32, i32 immarg, i32 immarg)"},
    {"global.load.lds",
     "declare void @llvm.amdgcn.global.load.lds(ptr addrspace(1) nocapture readonly, ptr "
     "addrspace(3) nocapture, i32 immarg, i32 immarg, i32 immarg)"},
    {"raw.ptr.buffer.load.v2i32",
     "declare <2 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v2i32(ptr addrspace(8) nocapture "
     "readonly, i32, i32, i32 immarg)"},
    {"raw.ptr.buffer.load.v4i32",
     "declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) nocapture "
     "readonly, i32, i32, i32 immarg)"},
    {"raw.ptr.buffer.load.i32",
     "declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) nocapture readonly, i32, "
     "i32, i32 immarg)"},
    {"raw.ptr.buffer.load.i16",
     "declare i16 @llvm.amdgcn.raw.ptr.buffer.load.i16(ptr addrspace(8) nocapture readonly, i32, "
     "i32, i32 immarg)"},
    {"raw.ptr.buffer.store.i32",
     "declare void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32, ptr addrspace(8) nocapture "
     "writeonly, i32, i32, i32 immarg)"},
    {"raw.ptr.buffer.store.v2i32",
     "declare void @llvm.amdgcn.raw.ptr.buffer.store.v2i32(<2 x i32>, ptr addrspace(8) nocapture "
     "writeonly, i32, i32, i32 immarg)"},
    {"raw.ptr.buffer.store.v4i32",
     "declare void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32>, ptr addrspace(8) nocapture "
     "writeonly, i32, i32, i32 immarg)"},
    {"raw.ptr.buffer.store.i16",
     "declare void @llvm.amdgcn.raw.ptr.buffer.store.i16(i16, ptr addrspace(8) nocapture "
     "writeonly, i32, i32, i32 immarg)"},
    {"amdgcn.cvt.f32.fp8", "declare float @llvm.amdgcn.cvt.f32.fp8(i32, i32 immarg)"},
    {"amdgcn.cvt.f32.bf8", "declare float @llvm.amdgcn.cvt.f32.bf8(i32, i32 immarg)"},
    {"amdgcn.cvt.pk.fp8.f32", "declare i32 @llvm.amdgcn.cvt.pk.fp8.f32(float, float, i32, i1)"},
    {"amdgcn.cvt.pk.bf8.f32", "declare i32 @llvm.amdgcn.cvt.pk.bf8.f32(float, float, i32, i1)"},
    {"amdgcn.cvt.pk.f32.fp8", "declare <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32, i1)"},
    {"amdgcn.cvt.pk.f32.bf8", "declare <2 x float> @llvm.amdgcn.cvt.pk.f32.bf8(i32, i1)"},
    {"rint.f32", "declare float @llvm.rint.f32(float)"},
    {"smax.i32", "declare i32 @llvm.smax.i32(i32, i32)"},
    {"smin.i32", "declare i32 @llvm.smin.i32(i32, i32)"},
    {"amdgcn.perm", "declare i32 @llvm.amdgcn.perm(i32, i32, i32)"},
    {"amdgcn.cvt.scalef32.pk.f32.fp8",
     "declare <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.fp8(i32, float, i1)"},
    {"amdgcn.cvt.scalef32.pk.f32.bf8",
     "declare <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.bf8(i32, float, i1)"},
    {"amdgcn.cvt.scalef32.pk.fp8.f32",
     "declare i32 @llvm.amdgcn.cvt.scalef32.pk.fp8.f32(i32, <2 x float>, float, i1)"},
    {"amdgcn.cvt.scalef32.pk.bf8.f32",
     "declare i32 @llvm.amdgcn.cvt.scalef32.pk.bf8.f32(i32, <2 x float>, float, i1)"},
    {"amdgcn.ds.swizzle", "declare i32 @llvm.amdgcn.ds.swizzle(i32, i32 immarg)"},
    /* Not overloaded, so no name suffix, but the flags are immarg like every
     * other permlane* flag pair. */
    {"amdgcn.permlane32.swap",
     "declare { i32, i32 } @llvm.amdgcn.permlane32.swap(i32, i32, i1 immarg, i1 immarg)"},
    /* Overloaded on the data type; see the amdgcn.permlane16 note below. */
    {"amdgcn.permlanex16",
     "declare i32 @llvm.amdgcn.permlanex16.i32(i32, i32, i32, i32, i1 immarg, i1 immarg)"},
    {"mfma.f32.32x32x16.bf16",
     "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16(<8 x bfloat>, <8 x bfloat>, <16 x "
     "float>, i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.scale.f32.16x16x128.f8f6f4",
     "declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4(<8 x i32>, <8 x i32>, <4 x "
     "float>, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32, i32 immarg)"},
    {"mfma.f32.16x16x128.fp4",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x128.fp4(i64, i64, <4 x float>, i32 immarg, "
     "i32 immarg, i32 immarg)"},
    {"mfma.f32.16x16x96.fp6",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x96.fp6(<3 x i32>, <3 x i32>, <4 x float>, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.16x16x128.fp8.hero",
     "declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4(<8 x i32>, <8 x i32>, <4 x "
     "float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32)"},
    {"asyncmark", "declare void @llvm.amdgcn.asyncmark()"},
    {"wait.asyncmark", "declare void @llvm.amdgcn.wait.asyncmark(i16 immarg)"},
    {"raw.ptr.buffer.load.async.lds",
     "declare void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) nocapture readonly, "
     "ptr addrspace(3) nocapture, i32, i32, i32, i32 immarg, i32 immarg)"},
    {"global.load.async.to.lds.b8",
     "declare void @llvm.amdgcn.global.load.async.to.lds.b8(ptr addrspace(1) nocapture, ptr "
     "addrspace(3) nocapture, i32 immarg, i32 immarg)"},
    {"mov.dpp8.i32", "declare i32 @llvm.amdgcn.mov.dpp8.i32(i32, i32 immarg)"},
    {"mov.dpp8.f32", "declare float @llvm.amdgcn.mov.dpp8.f32(float, i32 immarg)"},
    {"wave.reduce.fmax.f32", "declare float @llvm.amdgcn.wave.reduce.fmax.f32(float, i32 immarg)"},
    {"wave.reduce.fadd.f32", "declare float @llvm.amdgcn.wave.reduce.fadd.f32(float, i32 immarg)"},
    {"wave.reduce.add.i32", "declare i32 @llvm.amdgcn.wave.reduce.add.i32(i32, i32 immarg)"},
    {"wave.reduce.max.i32", "declare i32 @llvm.amdgcn.wave.reduce.max.i32(i32, i32 immarg)"},
    {"wave.reduce.min.i32", "declare i32 @llvm.amdgcn.wave.reduce.min.i32(i32, i32 immarg)"},
    {"readlane.i32", "declare i32 @llvm.amdgcn.readlane.i32(i32, i32)"},
    {"readlane.f32", "declare float @llvm.amdgcn.readlane.f32(float, i32)"},
    {"writelane.i32", "declare i32 @llvm.amdgcn.writelane.i32(i32, i32, i32)"},
    {"writelane.f32", "declare float @llvm.amdgcn.writelane.f32(float, i32, float)"},
    /* permlane16/64 and s.wqm are overloaded on their value type, so LLVM
     * mangles a suffix per overloaded position: one for permlane* (the data
     * type), two for s.wqm (result and operand are separately overloaded).
     * The unmangled spellings parse -- LLVM auto-upgrades them -- but they do
     * not survive a round trip, so emitting them makes the canonical form the
     * odd one out and any test pinning it fail. Mirrors Python. */
    {"amdgcn.permlane16",
     "declare i32 @llvm.amdgcn.permlane16.i32(i32, i32, i32, i32, i1 immarg, i1 immarg)"},
    {"amdgcn.permlane64", "declare i32 @llvm.amdgcn.permlane64.i32(i32)"},
    {"amdgcn.alignbyte", "declare i32 @llvm.amdgcn.alignbyte(i32, i32, i32)"},
    {"amdgcn.s.wqm.i64", "declare i64 @llvm.amdgcn.s.wqm.i64.i64(i64)"},
    {"amdgcn.s.wqm.i32", "declare i32 @llvm.amdgcn.s.wqm.i32.i32(i32)"},
    /* av.load/store.b128 are llvm_anyptr_ty too; see the s.prefetch.inst note
     * below and ROCKE_LL_AV_B128_PTR_TYPES. */
    {"av.load.b128.p0", "declare <4 x i32> @llvm.amdgcn.av.load.b128.p0(ptr, metadata)"},
    {"av.load.b128.p1",
     "declare <4 x i32> @llvm.amdgcn.av.load.b128.p1(ptr addrspace(1), metadata)"},
    {"av.store.b128.p0", "declare void @llvm.amdgcn.av.store.b128.p0(ptr, <4 x i32>, metadata)"},
    {"av.store.b128.p1",
     "declare void @llvm.amdgcn.av.store.b128.p1(ptr addrspace(1), <4 x i32>, metadata)"},
    {"s.alloc.vgpr", "declare i1 @llvm.amdgcn.s.alloc.vgpr(i32)"},
    {"s.wait.event", "declare void @llvm.amdgcn.s.wait.event(i16 immarg)"},
    /* s.prefetch.inst takes an llvm_anyptr_ty operand, so the overload is
     * mangled by address space and the declare has to name the SAME address
     * space the call site passes. One key per space rather than a single
     * unmangled declare: a kernel prefetching two pointers in different spaces
     * would otherwise redefine one name with two signatures. Mirrors Python
     * _S_PREFETCH_INST_PTR_TYPES. */
    {"s.prefetch.inst.p0", "declare void @llvm.amdgcn.s.prefetch.inst.p0(ptr, i32)"},
    {"s.prefetch.inst.p1", "declare void @llvm.amdgcn.s.prefetch.inst.p1(ptr addrspace(1), i32)"},
    {"s.prefetch.inst.p4", "declare void @llvm.amdgcn.s.prefetch.inst.p4(ptr addrspace(4), i32)"},
};

const int ROCKE_LL_INTRINSIC_DECLS_COUNT
    = (int)(sizeof(ROCKE_LL_INTRINSIC_DECLS) / sizeof(ROCKE_LL_INTRINSIC_DECLS[0]));

/* ---------------------------------------------------------------------- */
/* anyptr overload tables (Python _S_PREFETCH_INST_PTR_TYPES /            */
/* _AV_B128_PTR_TYPES)                                                    */
/* ---------------------------------------------------------------------- */

/* s.prefetch.inst reaches instruction memory through a flat, global, or
 * constant pointer (LLVM's own test uses addrspace(4)). */
const rocke_ll_anyptr_space_t ROCKE_LL_S_PREFETCH_INST_PTR_TYPES[] = {
    {0, "ptr"},
    {1, "ptr addrspace(1)"},
    {4, "ptr addrspace(4)"},
};
const int ROCKE_LL_S_PREFETCH_INST_PTR_TYPES_COUNT
    = (int)(sizeof(ROCKE_LL_S_PREFETCH_INST_PTR_TYPES)
            / sizeof(ROCKE_LL_S_PREFETCH_INST_PTR_TYPES[0]));

/* av.load/store.b128 are documented as flat or global only: a global pointer
 * selects global_load/store, a flat pointer flat_load/store. */
const rocke_ll_anyptr_space_t ROCKE_LL_AV_B128_PTR_TYPES[] = {
    {0, "ptr"},
    {1, "ptr addrspace(1)"},
};
const int ROCKE_LL_AV_B128_PTR_TYPES_COUNT
    = (int)(sizeof(ROCKE_LL_AV_B128_PTR_TYPES) / sizeof(ROCKE_LL_AV_B128_PTR_TYPES[0]));

/* ---------------------------------------------------------------------- */
/* LLVM22 overrides (Python _INTRINSIC_DECLS_LLVM22_OVERRIDES)            */
/* Same keys, different decl text.                                         */
/* ---------------------------------------------------------------------- */

const rocke_ll_decl_t ROCKE_LL_INTRINSIC_DECLS_LLVM22_OVERRIDES[] = {
    {"mfma.f32.16x16x32.fp8.fp8",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8("
     "i64, i64, <4 x float>, i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.16x16x32.bf8.bf8",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8("
     "i64, i64, <4 x float>, i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.32x32x16.fp8.fp8",
     "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.fp8.fp8("
     "i64, i64, <16 x float>, i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.32x32x16.bf8.bf8",
     "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.bf8("
     "i64, i64, <16 x float>, i32 immarg, i32 immarg, i32 immarg)"},
    {"make.buffer.rsrc.p1",
     "declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1("
     "ptr addrspace(1) nocapture readnone, i16, i64, i32)"},
};

const int ROCKE_LL_INTRINSIC_DECLS_LLVM22_OVERRIDES_COUNT
    = (int)(sizeof(ROCKE_LL_INTRINSIC_DECLS_LLVM22_OVERRIDES)
            / sizeof(ROCKE_LL_INTRINSIC_DECLS_LLVM22_OVERRIDES[0]));

/* ---------------------------------------------------------------------- */
/* LLVM23 overrides (Python _INTRINSIC_DECLS_LLVM23_OVERRIDES)            */
/* Identical to the LLVM22 set for the declares rocke emits today; split  */
/* entries here if an LLVM 23 host proves drift.                          */
/* ---------------------------------------------------------------------- */

const rocke_ll_decl_t ROCKE_LL_INTRINSIC_DECLS_LLVM23_OVERRIDES[] = {
    {"mfma.f32.16x16x32.fp8.fp8",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8("
     "i64, i64, <4 x float>, i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.16x16x32.bf8.bf8",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8("
     "i64, i64, <4 x float>, i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.32x32x16.fp8.fp8",
     "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.fp8.fp8("
     "i64, i64, <16 x float>, i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.32x32x16.bf8.bf8",
     "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.bf8("
     "i64, i64, <16 x float>, i32 immarg, i32 immarg, i32 immarg)"},
    {"make.buffer.rsrc.p1",
     "declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1("
     "ptr addrspace(1) nocapture readnone, i16, i64, i32)"},
};

const int ROCKE_LL_INTRINSIC_DECLS_LLVM23_OVERRIDES_COUNT
    = (int)(sizeof(ROCKE_LL_INTRINSIC_DECLS_LLVM23_OVERRIDES)
            / sizeof(ROCKE_LL_INTRINSIC_DECLS_LLVM23_OVERRIDES[0]));

/* Resolve the flavor-specific override table (NULL/0 for non-modern flavors),
 * mirroring the Python _Lowerer choosing which OVERRIDES dict to .update(). */
const rocke_ll_decl_t* rocke_ll_flavor_overrides(rocke_llvm_flavor_t flavor, int* out_count)
{
    if(flavor == ROCKE_LLVM_FLAVOR_LLVM22)
    {
        if(out_count)
            *out_count = ROCKE_LL_INTRINSIC_DECLS_LLVM22_OVERRIDES_COUNT;
        return ROCKE_LL_INTRINSIC_DECLS_LLVM22_OVERRIDES;
    }
    if(flavor == ROCKE_LLVM_FLAVOR_LLVM23)
    {
        if(out_count)
            *out_count = ROCKE_LL_INTRINSIC_DECLS_LLVM23_OVERRIDES_COUNT;
        return ROCKE_LL_INTRINSIC_DECLS_LLVM23_OVERRIDES;
    }
    if(out_count)
        *out_count = 0;
    return NULL;
}

} /* namespace ckc */
