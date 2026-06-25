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
/* _TRIPLE). The AMDGPU datalayout is FLAVOR-KEYED: only the buffer-fat-    */
/* pointer address space (p8) drifts between LLVM flavors --                */
/*   LLVM 20 (ROCm 7.0/7.1):  ...-p8:128:128-...                            */
/*   LLVM 22 (ROCm >= 7.2):   ...-p8:128:128:128:48-...                     */
/* (Python _DATALAYOUT_LLVM20 / _DATALAYOUT_LLVM22; pick via               */
/* rocke_ll_datalayout_for_flavor, mirroring _datalayout_for_flavor.)        */
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

/* Back-compat alias: callers that have not yet been flavor-threaded see the
 * LLVM20 form (the historical hardcoded value). New code keys on the flavor
 * via rocke_ll_datalayout_for_flavor / rocke_isa_datalayout. */
const char* const ROCKE_LL_DATALAYOUT = ROCKE_LL_DATALAYOUT_LLVM20;

const char* const ROCKE_LL_TRIPLE = "amdgcn-amd-amdhsa";

/* Python _datalayout_for_flavor: only LLVM20 returns the legacy p8 layout;
 * anything else (incl. an unexpected value) degrades to the modern LLVM22
 * form, exactly like the Python helper. */
const char* rocke_ll_datalayout_for_flavor(rocke_llvm_flavor_t flavor)
{
    return (flavor == ROCKE_LLVM_FLAVOR_LLVM20) ? ROCKE_LL_DATALAYOUT_LLVM20
                                                : ROCKE_LL_DATALAYOUT_LLVM22;
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
    {"exp2.f32", "declare float @llvm.exp2.f32(float)"},
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
     "declare <2 x float> @llvm.fmuladd.v2f32(<2 x float>, <2 x float>, "
     "<2 x float>)"},
    {"fmuladd.v4f32",
     "declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, "
     "<4 x float>)"},
    {"fmuladd.v8f32",
     "declare <8 x float> @llvm.fmuladd.v8f32(<8 x float>, <8 x float>, "
     "<8 x float>)"},
    {"fmuladd.v16f32",
     "declare <16 x float> @llvm.fmuladd.v16f32("
     "<16 x float>, <16 x float>, <16 x float>)"},
    {"fmuladd.v2f16",
     "declare <2 x half> @llvm.fmuladd.v2f16(<2 x half>, <2 x half>, "
     "<2 x half>)"},
    {"fmuladd.v4f16",
     "declare <4 x half> @llvm.fmuladd.v4f16(<4 x half>, <4 x half>, "
     "<4 x half>)"},
    {"fmuladd.v8f16",
     "declare <8 x half> @llvm.fmuladd.v8f16(<8 x half>, <8 x half>, "
     "<8 x half>)"},
    {"fmuladd.v2bf16",
     "declare <2 x bfloat> @llvm.fmuladd.v2bf16("
     "<2 x bfloat>, <2 x bfloat>, <2 x bfloat>)"},
    {"fmuladd.v4bf16",
     "declare <4 x bfloat> @llvm.fmuladd.v4bf16("
     "<4 x bfloat>, <4 x bfloat>, <4 x bfloat>)"},
    {"fmuladd.v8bf16",
     "declare <8 x bfloat> @llvm.fmuladd.v8bf16("
     "<8 x bfloat>, <8 x bfloat>, <8 x bfloat>)"},
    {"wmma.f32.16x16x16.f16",
     "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16("
     "<16 x half>, <16 x half>, <8 x float>)"},
    {"wmma.f32.16x16x16.bf16",
     "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16("
     "<16 x i16>, <16 x i16>, <8 x float>)"},
    {"wmma.i32.16x16x16.iu8",
     "declare <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v4i32("
     "i1, <4 x i32>, i1, <4 x i32>, <8 x i32>, i1)"},
    {"wmma.i32.16x16x16.iu4",
     "declare <8 x i32> @llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.v2i32("
     "i1, <2 x i32>, i1, <2 x i32>, <8 x i32>, i1)"},
    {"wmma.gfx12.f32.16x16x16.f16",
     "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16("
     "<8 x half>, <8 x half>, <8 x float>)"},
    {"wmma.gfx12.f32.16x16x16.bf16",
     "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v8i16("
     "<8 x i16>, <8 x i16>, <8 x float>)"},
    {"mfma.f32.16x16x16f16",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16("
     "<4 x half>, <4 x half>, <4 x float>, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.16x16x32.f16",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16("
     "<8 x half>, <8 x half>, <4 x float>, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.16x16x16bf16.1k",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k("
     "<4 x i16>, <4 x i16>, <4 x float>, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.16x16x32.bf16",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16("
     "<8 x bfloat>, <8 x bfloat>, <4 x float>, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.32x32x8f16",
     "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16("
     "<4 x half>, <4 x half>, <16 x float>, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.32x32x16.f16",
     "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.f16("
     "<8 x half>, <8 x half>, <16 x float>, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.32x32x8bf16.1k",
     "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x8bf16.1k("
     "<4 x i16>, <4 x i16>, <16 x float>, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.16x16x4f32",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32("
     "float, float, <4 x float>, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.32x32x2f32",
     "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x2f32("
     "float, float, <16 x float>, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.4x4x4f16",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x4f16("
     "<4 x half>, <4 x half>, <4 x float>, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.16x16x32.fp8.fp8",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8("
     "<2 x i32>, <2 x i32>, <4 x float>, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.16x16x32.bf8.bf8",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8("
     "<2 x i32>, <2 x i32>, <4 x float>, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.32x32x16.fp8.fp8",
     "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.fp8.fp8("
     "<2 x i32>, <2 x i32>, <16 x float>, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.32x32x16.bf8.bf8",
     "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.bf8("
     "<2 x i32>, <2 x i32>, <16 x float>, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"readfirstlane.i32", "declare i32 @llvm.amdgcn.readfirstlane.i32(i32)"},
    {"readfirstlane.i64", "declare i64 @llvm.amdgcn.readfirstlane.i64(i64)"},
    {"ballot.i64", "declare i64 @llvm.amdgcn.ballot.i64(i1)"},
    {"ds.bpermute", "declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)"},
    {"update.dpp.i32",
     "declare i32 @llvm.amdgcn.update.dpp.i32("
     "i32, i32, i32 immarg, i32 immarg, i32 immarg, i1 immarg)"},
    {"global.atomic.fadd.v2bf16",
     "declare <2 x bfloat> @llvm.amdgcn.global.atomic.fadd.v2bf16.p1("
     "ptr addrspace(1), <2 x bfloat>)"},
    {"mbcnt.lo", "declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32)"},
    {"mbcnt.hi", "declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32)"},
    {"ds.read.tr16.b64", "declare <4 x i16> @llvm.amdgcn.ds.read.tr16.b64(ptr addrspace(3))"},
    {"ds.read.tr16.b128", "declare <8 x i16> @llvm.amdgcn.ds.read.tr16.b128(ptr addrspace(3))"},
    {"iglp.opt", "declare void @llvm.amdgcn.iglp.opt(i32 immarg)"},
    {"sched.barrier", "declare void @llvm.amdgcn.sched.barrier(i32 immarg)"},
    {"sched.group.barrier",
     "declare void @llvm.amdgcn.sched.group.barrier("
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"s.setprio", "declare void @llvm.amdgcn.s.setprio(i16 immarg)"},
    {"s.waitcnt", "declare void @llvm.amdgcn.s.waitcnt(i32 immarg)"},
    {"make.buffer.rsrc.p1",
     "declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p1("
     "ptr addrspace(1) nocapture readnone, i16, i32, i32)"},
    {"raw.ptr.buffer.load.lds",
     "declare void @llvm.amdgcn.raw.ptr.buffer.load.lds("
     "ptr addrspace(8) nocapture readonly, ptr addrspace(3) nocapture, "
     "i32, i32, i32, i32 immarg, i32 immarg)"},
    {"global.load.lds",
     "declare void @llvm.amdgcn.global.load.lds("
     "ptr addrspace(1) nocapture readonly, ptr addrspace(3) nocapture, "
     "i32 immarg, i32 immarg, i32 immarg)"},
    {"raw.ptr.buffer.load.v2i32",
     "declare <2 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v2i32("
     "ptr addrspace(8) nocapture readonly, i32, i32, i32 immarg)"},
    {"raw.ptr.buffer.load.v4i32",
     "declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32("
     "ptr addrspace(8) nocapture readonly, i32, i32, i32 immarg)"},
    {"raw.ptr.buffer.load.i32",
     "declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32("
     "ptr addrspace(8) nocapture readonly, i32, i32, i32 immarg)"},
    {"raw.ptr.buffer.load.i16",
     "declare i16 @llvm.amdgcn.raw.ptr.buffer.load.i16("
     "ptr addrspace(8) nocapture readonly, i32, i32, i32 immarg)"},
    {"raw.ptr.buffer.store.i32",
     "declare void @llvm.amdgcn.raw.ptr.buffer.store.i32("
     "i32, ptr addrspace(8) nocapture writeonly, i32, i32, i32 immarg)"},
    {"raw.ptr.buffer.store.v2i32",
     "declare void @llvm.amdgcn.raw.ptr.buffer.store.v2i32("
     "<2 x i32>, ptr addrspace(8) nocapture writeonly, i32, i32, i32 immarg)"},
    {"raw.ptr.buffer.store.v4i32",
     "declare void @llvm.amdgcn.raw.ptr.buffer.store.v4i32("
     "<4 x i32>, ptr addrspace(8) nocapture writeonly, i32, i32, i32 immarg)"},
    {"raw.ptr.buffer.store.i16",
     "declare void @llvm.amdgcn.raw.ptr.buffer.store.i16("
     "i16, ptr addrspace(8) nocapture writeonly, i32, i32, i32 immarg)"},
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
     "declare <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.fp8(i32, float, "
     "i1)"},
    {"amdgcn.cvt.scalef32.pk.f32.bf8",
     "declare <2 x float> @llvm.amdgcn.cvt.scalef32.pk.f32.bf8(i32, float, "
     "i1)"},
    {"amdgcn.cvt.scalef32.pk.fp8.f32",
     "declare i32 @llvm.amdgcn.cvt.scalef32.pk.fp8.f32(i32, <2 x float>, "
     "float, i1)"},
    {"amdgcn.cvt.scalef32.pk.bf8.f32",
     "declare i32 @llvm.amdgcn.cvt.scalef32.pk.bf8.f32(i32, <2 x float>, "
     "float, i1)"},
    {"amdgcn.ds.swizzle", "declare i32 @llvm.amdgcn.ds.swizzle(i32, i32 immarg)"},
    {"amdgcn.permlane32.swap",
     "declare { i32, i32 } @llvm.amdgcn.permlane32.swap(i32, i32, i1, i1)"},
    {"amdgcn.permlanex16", "declare i32 @llvm.amdgcn.permlanex16(i32, i32, i32, i32, i1, i1)"},
    {"mfma.f32.32x32x16.bf16",
     "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf16("
     "<8 x bfloat>, <8 x bfloat>, <16 x float>, i32 immarg, i32 immarg, "
     "i32 immarg)"},
    {"mfma.scale.f32.16x16x128.f8f6f4",
     "declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4("
     "<8 x i32>, <8 x i32>, <4 x float>, i32 immarg, i32 immarg, "
     "i32 immarg, i32 immarg, i32, i32 immarg, i32, i32 immarg)"},
    {"mfma.f32.16x16x128.fp4",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x128.fp4("
     "i64, i64, <4 x float>, i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.16x16x96.fp6",
     "declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x96.fp6("
     "<3 x i32>, <3 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)"},
    {"mfma.f32.16x16x128.fp8.hero",
     "declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4("
     "<8 x i32>, <8 x i32>, <4 x float>, i32 immarg, i32 immarg, "
     "i32 immarg, i32, i32 immarg, i32)"},
    /* End of _INTRINSIC_DECLS: all 100 entries transcribed verbatim from
     * rocke/core/lower_llvm.py, in Python dict insertion order. The final
     * entry above ("mfma.f32.16x16x128.fp8.hero") is the last key in the
     * Python dict; there is no tail remaining to port. */
};

const int ROCKE_LL_INTRINSIC_DECLS_COUNT
    = (int)(sizeof(ROCKE_LL_INTRINSIC_DECLS) / sizeof(ROCKE_LL_INTRINSIC_DECLS[0]));

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

} /* namespace ckc */
