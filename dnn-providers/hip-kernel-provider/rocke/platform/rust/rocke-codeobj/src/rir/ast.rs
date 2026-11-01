// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Typed SSA form of the `ck.dsl.ir/v1` subset the direct backend compiles.

use std::borrow::Cow;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scalar {
    I1,
    I32,
    I64,
    F16,
    F32,
    BF16,
    I16,
    I8,
    FP8E4M3,
    BF8E5M2,
    F64,
    /// The 32-bit carrier of a TF32 operand of an emulated MMA atom: only
    /// bitcasts and `tile.mma` read it.
    TF32,
}

impl Scalar {
    pub fn bytes(self) -> u32 {
        match self {
            Scalar::I1 | Scalar::I8 | Scalar::FP8E4M3 | Scalar::BF8E5M2 => 1,
            Scalar::F16 | Scalar::BF16 | Scalar::I16 => 2,
            Scalar::I32 | Scalar::F32 | Scalar::TF32 => 4,
            Scalar::I64 | Scalar::F64 => 8,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, Scalar::F16 | Scalar::F32 | Scalar::BF16 | Scalar::F64)
    }

    /// 16-bit elements, held in one half of a 32-bit register and moved bit
    /// for bit by loads, stores, selects and vector assembly.
    pub fn is_half(self) -> bool {
        matches!(self, Scalar::F16 | Scalar::BF16 | Scalar::I16)
    }

    /// 8-bit elements, packed four to a 32-bit register; fp8 and bf8 are
    /// storage only (the kernels decode them with integer arithmetic).
    pub fn is_byte(self) -> bool {
        matches!(self, Scalar::I8 | Scalar::FP8E4M3 | Scalar::BF8E5M2)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Scalar(Scalar),
    Ptr(Scalar),
    Vec(Scalar, u32),
    Smem(Scalar, Vec<u32>),
}

impl Type {
    pub fn scalar(&self) -> Option<Scalar> {
        match self {
            Type::Scalar(scalar) => Some(*scalar),
            _ => None,
        }
    }

    /// Element type of a vector, or the scalar itself.
    pub fn elem(&self) -> Scalar {
        match self {
            Type::Scalar(scalar) | Type::Ptr(scalar) | Type::Vec(scalar, _) | Type::Smem(scalar, _) => {
                *scalar
            }
        }
    }

    /// Number of 32-bit registers a value of this type occupies.
    pub fn dwords(&self) -> u32 {
        match self {
            Type::Scalar(Scalar::I64) | Type::Scalar(Scalar::F64) | Type::Ptr(_) => 2,
            Type::Scalar(_) => 1,
            Type::Vec(elem, len) => (elem.bytes() * len).div_ceil(4),
            Type::Smem(..) => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub u32);

/// A list of values, stored inline while it has at most `N`.
#[derive(Clone)]
pub struct Ids<const N: usize> {
    len: u32,
    inline: [ValueId; N],
    spill: Option<Box<Vec<ValueId>>>,
}

impl<const N: usize> Ids<N> {
    pub fn new() -> Self {
        Ids { len: 0, inline: [ValueId(0); N], spill: None }
    }

    pub fn push(&mut self, v: ValueId) {
        let len = self.len as usize;
        if len < N {
            self.inline[len] = v;
        } else {
            let mut spill = self.spill.take().unwrap_or_else(|| Box::new(self.inline.to_vec()));
            spill.push(v);
            self.spill = Some(spill);
        }
        self.len += 1;
    }
}

impl<const N: usize> Default for Ids<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> std::ops::Deref for Ids<N> {
    type Target = [ValueId];
    fn deref(&self) -> &[ValueId] {
        match (self.len as usize, &self.spill) {
            (_, Some(spill)) => spill,
            (len, None) => &self.inline[..len],
        }
    }
}

impl<const N: usize> std::ops::DerefMut for Ids<N> {
    fn deref_mut(&mut self) -> &mut [ValueId] {
        match (self.len as usize, &mut self.spill) {
            (_, Some(spill)) => spill,
            (len, None) => &mut self.inline[..len],
        }
    }
}

impl<const N: usize> AsRef<[ValueId]> for Ids<N> {
    fn as_ref(&self) -> &[ValueId] {
        self
    }
}

impl<const N: usize> FromIterator<ValueId> for Ids<N> {
    fn from_iter<I: IntoIterator<Item = ValueId>>(iter: I) -> Self {
        let mut ids = Self::new();
        for v in iter {
            ids.push(v);
        }
        ids
    }
}

impl<const K: usize, const N: usize> From<[ValueId; K]> for Ids<N> {
    fn from(values: [ValueId; K]) -> Self {
        values.into_iter().collect()
    }
}

impl<const N: usize> From<Vec<ValueId>> for Ids<N> {
    fn from(values: Vec<ValueId>) -> Self {
        if values.len() <= N {
            return values.into_iter().collect();
        }
        Ids { len: values.len() as u32, inline: [ValueId(0); N], spill: Some(Box::new(values)) }
    }
}

impl<'a, const N: usize> IntoIterator for &'a Ids<N> {
    type Item = &'a ValueId;
    type IntoIter = std::slice::Iter<'a, ValueId>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<const N: usize, T: AsRef<[ValueId]>> PartialEq<T> for Ids<N> {
    fn eq(&self, other: &T) -> bool {
        **self == *other.as_ref()
    }
}

impl<const N: usize> std::fmt::Debug for Ids<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// Results of an op; all but `scf.for` with iteration arguments have at most one.
pub type Results = Ids<1>;
/// Operands of an op; nearly all have at most three.
pub type Operands = Ids<3>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_spill_past_inline_capacity() {
        let mut ids = Operands::new();
        for i in 0..7 {
            ids.push(ValueId(i));
            assert_eq!(ids.len(), i as usize + 1);
            assert!(ids.iter().enumerate().all(|(k, v)| v.0 == k as u32));
        }
        ids[5] = ValueId(50);
        assert_eq!(ids, [0, 1, 2, 3, 4, 50, 6].map(ValueId));
        let short: Operands = vec![ValueId(9)].into();
        let long: Operands = (0..5).map(ValueId).collect::<Vec<_>>().into();
        assert_eq!((short.len(), long.len()), (1, 5));
        assert_eq!(long, Operands::from([0, 1, 2, 3, 4].map(ValueId)));
    }
}

#[derive(Clone, Debug)]
pub struct ValueDef {
    /// Byte range of the name in `Kernel::names`.
    name: (u32, u32),
    pub ty: Type,
}

impl ValueDef {
    /// A value of type `ty` whose name is appended to `names`.
    pub fn new(names: &mut String, name: &str, ty: Type) -> ValueDef {
        let start = names.len() as u32;
        names.push_str(name);
        ValueDef { name: (start, names.len() as u32), ty }
    }

    /// The name, given the storage it was appended to.
    pub fn name<'a>(&self, names: &'a str) -> &'a str {
        &names[self.name.0 as usize..self.name.1 as usize]
    }
}

/// Attribute keys and string values; the ones the DSL emits are static.
pub type Text = Cow<'static, str>;

#[derive(Clone, Debug, PartialEq)]
pub enum Attr {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Text),
    List(Vec<Attr>),
    Map(Vec<(Text, Attr)>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpKind {
    // structure
    For,
    If,
    IfElse,
    Yield,
    Return,
    // integer / float arithmetic
    Constant,
    ConstantVec,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    Xor,
    Shl,
    Lshr,
    Ashr,
    Smin,
    Smax,
    UmulHi,
    SmulHi,
    Cmp,
    Select,
    Zext,
    Sext,
    Trunc,
    Bitcast,
    Fadd,
    Fsub,
    Fmul,
    Fma,
    Fmax,
    Fcmp,
    CastToF32,
    CastF32To,
    Exp2,
    Log2,
    Rcp,
    // vectors
    Extract,
    Insert,
    Pack,
    Concat,
    TruncF32To,
    // memory and GPU
    ThreadId,
    BlockId,
    GlobalPtrAdd,
    GlobalLoad,
    GlobalLoadVec,
    GlobalStore,
    GlobalStoreVec,
    GlobalAtomicAddF32,
    SmemAlloc,
    SmemLoadVec,
    SmemStore,
    SmemStoreVec,
    BufferRsrc,
    BufferLoadVec,
    BufferStoreVecF16,
    DsSwizzleXor,
    Mma,
    Sync,
    ReadFirstLane,
    InlineAsmIdentity,
    // scalar float ops
    Fneg,
    Fabs,
    Fmin,
    Fmax3,
    Fmin3,
    Fdiv,
    Fdot2,
    Exp2Fast,
    RcpFast,
    Rsqrt,
    Tanh,
    SitofpF32,
    // integer odds and ends
    Not,
    GlobalPtrToI64,
    // element-wise vector ops
    VecAdd,
    VecSub,
    VecMul,
    VecFma,
    VecSplat,
    VecBitcast,
    VecUndef,
    VecSelect,
    VecSum,
    // cross-lane ops
    PermB32,
    Permlanex16,
    Permlane16,
    Permlane64,
    DsBpermute,
    WaveAny,
    WaveAll,
    LaneId,
    // atomics, fences, buffer stores, barriers
    GlobalAtomicAdd,
    Fence,
    SyncLdsOnly,
    TruncF32ToF16,
    Rint,
    SetPrio,
    WaitCnt,
    SchedBarrier,
    SchedGroupBarrier,
    /// `tile.iglp_opt {level}`: an instruction-ordering strategy for the
    /// scheduling region it starts (LLVM's `llvm.amdgcn.iglp.opt`).
    IglpOpt,
    BarrierBare,
    /// `(smem, row, col)`: four f16 elements from `[row][col]` on.
    SmemLoadV4,
    /// `(smem, indices.., value)`, returns the old value.
    LdsAtomicAdd,
    DppXor,
    QuadPerm,
    MovDpp8,
    // internal forms produced by the optimizer (never parsed)
    KernArg,
    /// `(addr: ptr) {offset}`: load from a legalized 64-bit address.
    GLoad,
    /// `(addr: ptr, value) {offset}`.
    GStore,
    /// `(addr: ptr, value: f32) {offset}`.
    GAtomicAddF32,
    /// `(smem, addr: i32) {offset}`: LDS load at byte address + offset.
    LLoad,
    /// `(smem, addr: i32, value) {offset}`.
    LStore,
    /// `(smem, addr: i32, value) {offset}`: LDS atomic add, returns the old value.
    LAtomicAdd,
    // element types beyond f16/f32 (census W02, W09, W17, W18, W19)
    /// `arith.trunc_f32_to_bf16`: `fptrunc float to bfloat`, as `cast_f32_to`.
    TruncF32ToBf16,
    /// `(rsrc, voffset, soffset) {offset}`: one element of the result type,
    /// 2 or 4 bytes, zero when out of bounds.
    BufferLoadScalar,
    /// `(rsrc, voffset, soffset, value) {offset}`: one element, dropped when
    /// out of bounds.
    BufferStoreScalar,
    /// `arith.cvt_f32_to_i8_sat`: round to nearest even, clamp to [-128, 127].
    CvtF32ToI8Sat,
    /// `(addr: ptr, value) {offset}`: legalized `memref.global_atomic_add`;
    /// returns the old value.
    GAtomicAdd,
    /// `tile.smem_addr_of (smem) -> i64`: the allocation's LDS byte address.
    SmemAddrOf,
    /// `tile.smem_ptr_add (addr: i64, offset: i64) -> i64`: an LDS address plus a byte offset.
    SmemPtrAdd,
    /// `tile.async_buffer_load_lds_addr` and `tile.async_buffer_load_lds`
    /// `(rsrc, lds: i64, voffset, soffset) {dwords, aux}`: each lane's
    /// `dwords` dwords at its buffer offset go to LDS at the first lane's
    /// address plus 4 * `dwords` * lane (LDS DMA, gfx950).
    BufferLoadLds,
    /// `tile.global_load_lds (ptr, byte_offset: i32, lds: i64) {size_bytes, aux}`:
    /// as `BufferLoadLds`, from each lane's `ptr + byte_offset`.
    GlobalLoadLds,
    /// `tile.ds_read_tr16_b64 (smem, indices...) -> vec<16-bit x4>`: the
    /// transposing LDS read at the element's address (gfx950).
    DsReadTr16,
    /// `tile.ds_read_tr_b8 (smem, indices...) -> vec<8-bit x8>`: as
    /// `DsReadTr16`, in 8-bit elements.
    DsReadTr8,
    /// Either transposing read at `smem`'s `addr + offset` bytes, `bits` 16
    /// or 8 (`opt` canonical form).
    LdsReadTr,
    /// `tile.permlane32_swap (lo: i32, hi: i32) -> (i32, i32)`:
    /// `llvm.amdgcn.permlane32.swap` (gfx950).
    Permlane32Swap,
    /// `tile.readlane (v: i32|f32, lane: i32)`: lane `lane`'s `v`, uniform.
    ReadLane,
    /// `arith.cvt_fp8_to_f32 (fp8e4m3) -> f32` and its bf8e5m2 twin (gfx950).
    CvtFp8ToF32,
    CvtBf8ToF32,
    /// `arith.cvt_pk_f32_fp8x4 (vec<fp8e4m3x4>) -> vec<f32x4>` and its bf8e5m2 twin (gfx950).
    CvtPkF32Fp8,
    CvtPkF32Bf8,
    /// `arith.cvt_f32_to_fp8 (f32) -> fp8e4m3` and its bf8e5m2 twin: round to
    /// nearest even, no saturation (gfx950).
    CvtF32ToFp8,
    CvtF32ToBf8,
    /// `arith.cvt_pk_fp8_f32x4 (vec<f32x4>) -> vec<fp8e4m3x4>` and its bf8e5m2 twin (gfx950).
    CvtPkFp8F32,
    CvtPkBf8F32,
    /// `vector.and (a, b: vec<i32xN>)`: element-wise and.
    VecAnd,
    /// `tile.s_wait_asynccnt {n}`: gfx1250's wait on its asynchronous LDS
    /// copies; a target without that counter has none to wait for.
    WaitAsyncCnt,
    /// `memref.global_flag_wait_eq (flags, index, value) {scope}`: polls the
    /// i32 `flags[index]` with acquire loads until it holds `value`.
    FlagWaitEq,
    /// `memref.global_flag_store (flags, index, value) {ordering, scope}`: an
    /// atomic i32 store.
    FlagStore,
    // The Qwen3.8 monokernel's persistent-kernel and MX ops (gfx950; the
    // Python engine lowers them, `rocke.core.ir.PYTHON_ONLY_OPCODES`).
    /// `tile.int_to_ptr (addr: i64) -> ptr<T,global>`: a global pointer from
    /// a runtime address; it is based on no argument (alias tag `ANY`).
    IntToPtr,
    /// `tile.buffer_load_cp (rsrc, voffset, soffset) {dwords, aux}`: a raw
    /// buffer load of 1, 2 or 4 dwords with the cache-policy bits `aux`
    /// (sc0 1, nt 2, sc1 16); `i32` or `vec<i32xN>`.
    BufferLoadCp,
    /// `tile.buffer_store_cp (data, rsrc, voffset, soffset) {dwords, aux}`.
    BufferStoreCp,
    /// `tile.tagged_poll (tag, rsrc0, voffset0, ...) {dwords: list, aux, pause}`:
    /// each lane reloads its `(payload, tag)` pairs (buffer loads with `aux`)
    /// until every pair's tag word equals `tag`, executing `pause` between
    /// tries; one `vec<i32xN>` per spec. A leaf: the loop is expanded by
    /// selection, so no pass sees or hoists it.
    TaggedPoll,
    /// `arith.cvt_scalef32_pk_bf16_fp4 (src: i32, scale: f32) {byte}`:
    /// `v_cvt_scalef32_pk_bf16_fp4`, byte `byte` of `src` (two E2M1 codes)
    /// as `vec<bf16x2>`, scaled by the exponent field of `scale`.
    CvtScalePkBf16Fp4,
    /// `tile.permlane16_swap (lo: i32, hi: i32) -> (i32, i32)`:
    /// `v_permlane16_swap_b32` (gfx950).
    Permlane16Swap,
    /// `tile.s_getreg {hwreg}` -> i32: `s_getreg_b32` of `XCC_ID` or `HW_ID`.
    SGetreg,
    /// `tile.s_sleep {units}`: `s_sleep` in steps of at most 127.
    SSleep,
    /// `tile.memrealtime` -> i64: `s_memrealtime`.
    MemRealtime,
    /// `tile.global_atomic_add_scoped (voff: i32, value: i32|i64, base: i64)
    /// {scope, returns}`: `global_atomic_add[_x2]` at `base + voff`, `sc1` at
    /// system scope, `sc0` and the old value when `returns`.
    AtomicAddScoped,
    /// `tile.l2_touch_load (voff: i32, base: i64) -> vec<i32x4>`: a
    /// `global_load_dwordx4` whose value is never read.
    L2TouchLoad,
    /// `tile.l2_touch_drain (values...)`: `s_waitcnt vmcnt(0)`, holding the
    /// `l2_touch_load` destinations allocated until then.
    L2TouchDrain,
    // The rust track's ops (gfx950; `target::gfx9::q38`).
    /// `math.exp (x: f32) -> f32`: LLVM's `llvm.exp.f32` expansion.
    Exp,
    /// `arith.cvt_scalef32_pk_fp8_f32 (old: i32, src0: f32, src1: f32, scale:
    /// f32) {hi} -> i32`: two scaled fp8 bytes into one half of `old`.
    CvtScalePkFp8F32,
    /// `arith.cvt_scalef32_pk_bf8_f32`: as `CvtScalePkFp8F32`, bf8 bytes.
    CvtScalePkBf8F32,
    /// `tile.mfma_scale_mx (a, b, c, a_scale: i32, b_scale: i32) {shape,
    /// fmt_a, fmt_b, opsel_a, opsel_b}`: the block-scaled f8f6f4 MFMA.
    MfmaScaleMx,
    /// `arith.maximum3_f32 (a, b, c) {abs}`: IEEE maximum of three, `abs`
    /// a mask of the sources taken as magnitudes (bit k: source k).
    Maximum3,
    /// `tile.dpp_op (old: i32, src0: i32[, src1: i32]) {op, ctrl, row_mask,
    /// bank_mask, bound_ctrl} -> i32`: `v_mov_b32_dpp` (op `mov`) or
    /// `v_max_u32_dpp` (`max_u32`) of `src0` through the DPP control; lanes
    /// the masks leave unwritten keep `old`.
    DppOp,
    /// `tile.mov_dpp (src: i32) {row_shl | row_shr, bound_ctrl}`: a row shift
    /// (`llvm.amdgcn.update.dpp` with `src` as old).
    MovDpp,
    /// `arith.cvt_f32_to_f64` / `arith.cvt_f64_to_f32`.
    CvtF32ToF64,
    CvtF64ToF32,
    /// `math.sqrt (x: f64) -> f64`: LLVM's expansion (rsq, Goldschmidt steps).
    Sqrt,
    /// `tile.global_store_cp (data, voff: i32, base: i64) | (data, addr: i64)
    /// {aux, offset}`: a global store of 1, 2 or 4 dwords with the cache bits
    /// `aux` (sc0 1, nt 2, sc1 16), through saddr or a per-lane address.
    GlobalStoreCp,
    /// `tile.global_load_cp (voff: i32, base: i64) | (addr: i64) {aux, offset,
    /// dwords}`: a global load (i32, i64 or `vec<i32xN>`) with cache bits.
    GlobalLoadCp,
    /// `tile.bcnt (x: i32) -> i32`: `v_bcnt_u32_b32 x, 0`.
    Bcnt,
    /// `tile.s_ff1 (x: i32|i64) -> i32`: `s_ff1_i32_b32` / `_b64` (uniform).
    SFf1,
    /// `tile.memtime` -> i64: `s_memtime`.
    MemTime,
    /// `tile.s_nop {n}`: `s_nop n`.
    SNop,
    /// `tile.compiler_fence`: no instruction; no memory access moves across it.
    CompilerFence,
    /// `tile.ds_or_b32 (addr: i32, value: i32)`, `tile.ds_write_b32 (addr,
    /// value)`, `tile.ds_read_b32 (addr) -> i32`: LDS at a raw byte address.
    DsOrRaw,
    DsWriteRaw,
    DsReadRaw,
}

impl OpKind {
    pub fn from_name(name: &str) -> Option<OpKind> {
        Some(match name {
            "scf.for" => OpKind::For,
            "scf.if" => OpKind::If,
            "scf.if_else" => OpKind::IfElse,
            "scf.yield" => OpKind::Yield,
            "cf.return" => OpKind::Return,
            "arith.constant" => OpKind::Constant,
            "arith.constant_vec" => OpKind::ConstantVec,
            "arith.add" => OpKind::Add,
            "arith.sub" => OpKind::Sub,
            "arith.mul" => OpKind::Mul,
            "arith.div" => OpKind::Div,
            "arith.mod" => OpKind::Mod,
            "arith.and" => OpKind::And,
            "arith.or" => OpKind::Or,
            "arith.xor" => OpKind::Xor,
            "arith.shl" => OpKind::Shl,
            "arith.lshr" => OpKind::Lshr,
            "arith.ashr" => OpKind::Ashr,
            "arith.smin" => OpKind::Smin,
            "arith.smax" => OpKind::Smax,
            "arith.umul_hi_i32" => OpKind::UmulHi,
            "arith.cmp" => OpKind::Cmp,
            "arith.select" => OpKind::Select,
            "arith.zext" => OpKind::Zext,
            "arith.sext" => OpKind::Sext,
            "arith.trunc" => OpKind::Trunc,
            "arith.bitcast" => OpKind::Bitcast,
            "arith.fadd" => OpKind::Fadd,
            "arith.fsub" => OpKind::Fsub,
            "arith.fmul" => OpKind::Fmul,
            "arith.fma" => OpKind::Fma,
            "arith.fmax" => OpKind::Fmax,
            "arith.fcmp" => OpKind::Fcmp,
            "arith.cast_to_f32" => OpKind::CastToF32,
            "arith.cast_f32_to" => OpKind::CastF32To,
            "math.exp2" => OpKind::Exp2,
            "math.log2" => OpKind::Log2,
            "math.rcp" => OpKind::Rcp,
            "vector.extract" => OpKind::Extract,
            "vector.insert" => OpKind::Insert,
            "vector.pack" => OpKind::Pack,
            "vector.concat" => OpKind::Concat,
            "vector.trunc_f32_to" => OpKind::TruncF32To,
            "gpu.thread_id" => OpKind::ThreadId,
            "gpu.block_id" => OpKind::BlockId,
            "tile.global_ptr_add" => OpKind::GlobalPtrAdd,
            "memref.global_load_typed" => OpKind::GlobalLoad,
            "memref.global_load_vN" => OpKind::GlobalLoadVec,
            "memref.global_store_typed" => OpKind::GlobalStore,
            "memref.global_store_vN" => OpKind::GlobalStoreVec,
            "memref.global_atomic_add_f32" => OpKind::GlobalAtomicAddF32,
            "tile.smem_alloc" => OpKind::SmemAlloc,
            "tile.smem_load_vN" => OpKind::SmemLoadVec,
            "tile.smem_store" => OpKind::SmemStore,
            "tile.smem_store_vN" => OpKind::SmemStoreVec,
            "tile.buffer_rsrc" => OpKind::BufferRsrc,
            "tile.buffer_load_vN" => OpKind::BufferLoadVec,
            "tile.buffer_store_vN_f16" => OpKind::BufferStoreVecF16,
            "tile.ds_swizzle_xor" => OpKind::DsSwizzleXor,
            "tile.mma" => OpKind::Mma,
            "tile.sync" => OpKind::Sync,
            "tile.readfirstlane" => OpKind::ReadFirstLane,
            "tile.inline_asm" => OpKind::InlineAsmIdentity,
            "tile.buffer_load" | "tile.buffer_load_f16" | "tile.buffer_load_f16_d16" => OpKind::BufferLoadScalar,
            "tile.buffer_load_vN_f16" => OpKind::BufferLoadVec,
            "tile.buffer_store_f16" | "tile.buffer_store_f32" => OpKind::BufferStoreScalar,
            "tile.buffer_store_vN_f32" => OpKind::BufferStoreVecF16,
            "arith.trunc_f32_to_f16" => OpKind::TruncF32ToF16,
            "arith.rint_f32" => OpKind::Rint,
            // An SGPR-class hint on a uniform value; LLVM reads the first lane
            // of a divergent one.
            "tile.pin_sgpr" => OpKind::ReadFirstLane,
            "tile.s_setprio" => OpKind::SetPrio,
            "tile.s_waitcnt" => OpKind::WaitCnt,
            "tile.sched_barrier" => OpKind::SchedBarrier,
            "tile.sched_group_barrier" => OpKind::SchedGroupBarrier,
            "tile.iglp_opt" => OpKind::IglpOpt,
            "tile.s_barrier_bare" => OpKind::BarrierBare,
            "tile.smem_load_vN_f32" => OpKind::SmemLoadVec,
            "tile.smem_store_vN_f32" => OpKind::SmemStoreVec,
            "tile.smem_load_v4" => OpKind::SmemLoadV4,
            "tile.lds_atomic_add" => OpKind::LdsAtomicAdd,
            "tile.dpp_xor" => OpKind::DppXor,
            "tile.quad_perm" => OpKind::QuadPerm,
            "tile.mov_dpp8" => OpKind::MovDpp8,
            "arith.trunc_f32_to_bf16" => OpKind::TruncF32ToBf16,
            "tile.buffer_load_bf16" => OpKind::BufferLoadScalar,
            "tile.buffer_store_bf16" => OpKind::BufferStoreScalar,
            "tile.buffer_load_vN_bf16" => OpKind::BufferLoadVec,
            "tile.buffer_store_vN_bf16" => OpKind::BufferStoreVecF16,
            "arith.cvt_f32_to_i8_sat" => OpKind::CvtF32ToI8Sat,
            "arith.fneg" => OpKind::Fneg,
            "arith.fabs" => OpKind::Fabs,
            "arith.fmin" => OpKind::Fmin,
            "arith.fmax3" => OpKind::Fmax3,
            "arith.fmin3" => OpKind::Fmin3,
            "arith.fdiv" => OpKind::Fdiv,
            "arith.fdot2" => OpKind::Fdot2,
            "math.exp2_fast" => OpKind::Exp2Fast,
            "math.rcp_fast" => OpKind::RcpFast,
            "math.rsqrt" => OpKind::Rsqrt,
            "math.tanh" => OpKind::Tanh,
            "arith.sitofp_f32" => OpKind::SitofpF32,
            "arith.not" => OpKind::Not,
            "tile.global_ptr_to_i64" => OpKind::GlobalPtrToI64,
            // The f16 helper forms of the typed global accesses.
            "memref.global_load" => OpKind::GlobalLoad,
            "memref.global_store" => OpKind::GlobalStore,
            "vector.add" => OpKind::VecAdd,
            "vector.sub" => OpKind::VecSub,
            "vector.mul" => OpKind::VecMul,
            "vector.fma" => OpKind::VecFma,
            "vector.splat" => OpKind::VecSplat,
            "vector.bitcast" => OpKind::VecBitcast,
            "vector.undef" => OpKind::VecUndef,
            "vector.select" => OpKind::VecSelect,
            "vector.sum" => OpKind::VecSum,
            "tile.perm_b32" => OpKind::PermB32,
            "tile.permlanex16" => OpKind::Permlanex16,
            "tile.permlane16" => OpKind::Permlane16,
            "tile.permlane64" => OpKind::Permlane64,
            "tile.ds_bpermute" => OpKind::DsBpermute,
            "tile.wave_any" => OpKind::WaveAny,
            "tile.wave_all" => OpKind::WaveAll,
            "tile.lane_id" => OpKind::LaneId,
            "memref.global_atomic_add" => OpKind::GlobalAtomicAdd,
            // `(ptr<f16>, index, vec<f16x2>)`: an atomic add of both halves at an f16 element index.
            "memref.global_atomic_add_pk_f16" => OpKind::GlobalAtomicAdd,
            "memref.fence" => OpKind::Fence,
            "tile.sync_lds_only" => OpKind::SyncLdsOnly,
            "tile.smem_addr_of" => OpKind::SmemAddrOf,
            "tile.smem_ptr_add" => OpKind::SmemPtrAdd,
            "tile.async_buffer_load_lds_addr" | "tile.async_buffer_load_lds" => OpKind::BufferLoadLds,
            "tile.global_load_lds" => OpKind::GlobalLoadLds,
            "tile.ds_read_tr16_b64" => OpKind::DsReadTr16,
            "tile.ds_read_tr_b8" => OpKind::DsReadTr8,
            "tile.permlane32_swap" => OpKind::Permlane32Swap,
            "tile.readlane" => OpKind::ReadLane,
            "arith.cvt_fp8_to_f32" => OpKind::CvtFp8ToF32,
            "arith.cvt_bf8_to_f32" => OpKind::CvtBf8ToF32,
            "arith.cvt_pk_f32_fp8x4" => OpKind::CvtPkF32Fp8,
            "arith.cvt_pk_f32_bf8x4" => OpKind::CvtPkF32Bf8,
            "arith.cvt_f32_to_fp8" => OpKind::CvtF32ToFp8,
            "arith.cvt_f32_to_bf8" => OpKind::CvtF32ToBf8,
            "arith.cvt_pk_fp8_f32x4" => OpKind::CvtPkFp8F32,
            "arith.cvt_pk_bf8_f32x4" => OpKind::CvtPkBf8F32,
            "vector.and" => OpKind::VecAnd,
            "tile.s_wait_asynccnt" => OpKind::WaitAsyncCnt,
            "memref.global_flag_wait_eq" => OpKind::FlagWaitEq,
            "memref.global_flag_store" => OpKind::FlagStore,
            "tile.int_to_ptr" => OpKind::IntToPtr,
            "tile.buffer_load_cp" => OpKind::BufferLoadCp,
            "tile.buffer_store_cp" => OpKind::BufferStoreCp,
            "tile.tagged_poll" => OpKind::TaggedPoll,
            "arith.cvt_scalef32_pk_bf16_fp4" => OpKind::CvtScalePkBf16Fp4,
            "tile.permlane16_swap" => OpKind::Permlane16Swap,
            "tile.s_getreg" => OpKind::SGetreg,
            "tile.s_sleep" => OpKind::SSleep,
            "tile.memrealtime" => OpKind::MemRealtime,
            "tile.global_atomic_add_scoped" => OpKind::AtomicAddScoped,
            "tile.l2_touch_load" => OpKind::L2TouchLoad,
            "tile.l2_touch_drain" => OpKind::L2TouchDrain,
            "math.exp" => OpKind::Exp,
            "arith.cvt_scalef32_pk_fp8_f32" => OpKind::CvtScalePkFp8F32,
            "arith.cvt_scalef32_pk_bf8_f32" => OpKind::CvtScalePkBf8F32,
            "tile.mfma_scale_mx" => OpKind::MfmaScaleMx,
            "arith.maximum3_f32" => OpKind::Maximum3,
            "tile.dpp_op" => OpKind::DppOp,
            "tile.mov_dpp" => OpKind::MovDpp,
            "arith.cvt_f32_to_f64" => OpKind::CvtF32ToF64,
            "arith.cvt_f64_to_f32" => OpKind::CvtF64ToF32,
            "math.sqrt" => OpKind::Sqrt,
            "tile.global_store_cp" => OpKind::GlobalStoreCp,
            "tile.global_load_cp" => OpKind::GlobalLoadCp,
            "tile.bcnt" => OpKind::Bcnt,
            "tile.s_ff1" => OpKind::SFf1,
            "tile.memtime" => OpKind::MemTime,
            "tile.s_nop" => OpKind::SNop,
            "tile.compiler_fence" => OpKind::CompilerFence,
            "tile.ds_or_b32" => OpKind::DsOrRaw,
            "tile.ds_write_b32" => OpKind::DsWriteRaw,
            "tile.ds_read_b32" => OpKind::DsReadRaw,
            _ => return None,
        })
    }

    /// No side effects and no dependence on memory: safe to fold, CSE, hoist, or delete.
    pub fn is_pure(self) -> bool {
        !matches!(
            self,
            OpKind::For
                | OpKind::If
                | OpKind::IfElse
                | OpKind::Yield
                | OpKind::Return
                | OpKind::GlobalLoad
                | OpKind::GlobalLoadVec
                | OpKind::GlobalStore
                | OpKind::GlobalStoreVec
                | OpKind::GlobalAtomicAddF32
                | OpKind::SmemAlloc
                | OpKind::SmemLoadVec
                | OpKind::SmemStore
                | OpKind::SmemStoreVec
                | OpKind::BufferLoadVec
                | OpKind::BufferStoreVecF16
                | OpKind::DsSwizzleXor
                | OpKind::Sync
                | OpKind::InlineAsmIdentity
                | OpKind::GLoad
                | OpKind::GStore
                | OpKind::GAtomicAddF32
                | OpKind::LLoad
                | OpKind::LStore
                | OpKind::SetPrio
                | OpKind::WaitCnt
                | OpKind::SchedBarrier
                | OpKind::SchedGroupBarrier
                | OpKind::IglpOpt
                | OpKind::BarrierBare
                | OpKind::SmemLoadV4
                | OpKind::LdsAtomicAdd
                | OpKind::LAtomicAdd
                | OpKind::DppXor
                | OpKind::QuadPerm
                | OpKind::MovDpp8
                | OpKind::BufferLoadScalar
                | OpKind::BufferStoreScalar
                // Results depend on which lanes are active: no CSE or hoisting.
                | OpKind::Permlanex16
                | OpKind::Permlane16
                | OpKind::Permlane64
                | OpKind::DsBpermute
                | OpKind::WaveAny
                | OpKind::WaveAll
                | OpKind::GlobalAtomicAdd
                | OpKind::GAtomicAdd
                | OpKind::Fence
                | OpKind::SyncLdsOnly
                | OpKind::BufferLoadLds
                | OpKind::GlobalLoadLds
                // A lane's result depends on the other lanes' addresses.
                | OpKind::DsReadTr16
                | OpKind::DsReadTr8
                | OpKind::LdsReadTr
                | OpKind::Permlane32Swap
                | OpKind::ReadLane
                | OpKind::WaitAsyncCnt
                | OpKind::FlagWaitEq
                | OpKind::FlagStore
                | OpKind::BufferLoadCp
                | OpKind::BufferStoreCp
                | OpKind::TaggedPoll
                | OpKind::Permlane16Swap
                | OpKind::SGetreg
                | OpKind::SSleep
                | OpKind::MemRealtime
                | OpKind::AtomicAddScoped
                | OpKind::L2TouchLoad
                | OpKind::L2TouchDrain
                // A lane's result depends on the other lanes.
                | OpKind::DppOp
                | OpKind::MovDpp
                | OpKind::GlobalStoreCp
                | OpKind::GlobalLoadCp
                | OpKind::MemTime
                | OpKind::SNop
                | OpKind::CompilerFence
                | OpKind::DsOrRaw
                | OpKind::DsWriteRaw
                | OpKind::DsReadRaw
        )
    }

    /// Deletable when its results are unused: pure ops and plain loads.
    pub fn is_removable(self) -> bool {
        self.is_pure()
            || matches!(
                self,
                OpKind::GlobalLoad
                    | OpKind::GlobalLoadVec
                    | OpKind::SmemLoadVec
                    | OpKind::BufferLoadVec
                    | OpKind::GLoad
                    | OpKind::LLoad
                    | OpKind::InlineAsmIdentity
                    | OpKind::SmemLoadV4
                    | OpKind::DppXor
                    | OpKind::QuadPerm
                    | OpKind::MovDpp8
                    | OpKind::BufferLoadScalar
                    | OpKind::Permlanex16
                    | OpKind::Permlane16
                    | OpKind::Permlane64
                    | OpKind::DsBpermute
                    | OpKind::WaveAny
                    | OpKind::WaveAll
                    | OpKind::DsReadTr16
                    | OpKind::DsReadTr8
                    | OpKind::LdsReadTr
                    | OpKind::Permlane32Swap
                    | OpKind::ReadLane
                    | OpKind::BufferLoadCp
                    | OpKind::Permlane16Swap
                    | OpKind::DppOp
                    | OpKind::MovDpp
                    | OpKind::GlobalLoadCp
                    | OpKind::DsReadRaw
            )
    }

    pub fn writes_global(self) -> bool {
        matches!(
            self,
            OpKind::GlobalStore
                | OpKind::GlobalStoreVec
                | OpKind::GlobalAtomicAddF32
                | OpKind::BufferStoreVecF16
                | OpKind::GStore
                | OpKind::GAtomicAddF32
                | OpKind::BufferStoreScalar
                | OpKind::GlobalAtomicAdd
                | OpKind::GAtomicAdd
                // Later loads must see other agents' writes: none may use the scalar cache.
                | OpKind::Fence
                | OpKind::FlagWaitEq
                | OpKind::FlagStore
                | OpKind::BufferStoreCp
                | OpKind::TaggedPoll
                | OpKind::AtomicAddScoped
                | OpKind::GlobalStoreCp
                // Orders memory like a write: nothing is reused across it.
                | OpKind::CompilerFence
        )
    }
}

#[derive(Clone, Debug)]
pub struct Op {
    pub kind: OpKind,
    pub results: Results,
    pub operands: Operands,
    pub attrs: Vec<(Text, Attr)>,
    pub regions: Vec<Region>,
    /// Values a region defines on entry: the induction variable and iteration
    /// arguments of `scf.for`.
    pub block_args: Vec<ValueId>,
}

impl Op {
    pub fn attr(&self, key: &str) -> Option<&Attr> {
        self.attrs.iter().find(|(name, _)| name == key).map(|(_, value)| value)
    }

    pub fn int_attr(&self, key: &str) -> Option<i64> {
        match self.attr(key) {
            Some(Attr::Int(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn str_attr(&self, key: &str) -> Option<&str> {
        match self.attr(key) {
            Some(Attr::Str(value)) => Some(value),
            _ => None,
        }
    }

    pub fn float_attr(&self, key: &str) -> Option<f64> {
        match self.attr(key) {
            Some(Attr::Float(value)) => Some(*value),
            Some(Attr::Int(value)) => Some(*value as f64),
            _ => None,
        }
    }

    pub fn bool_attr(&self, key: &str) -> Option<bool> {
        match self.attr(key) {
            Some(Attr::Bool(value)) => Some(*value),
            _ => None,
        }
    }

    /// The register file of an identity pin, `'v'` or `'s'`: a
    /// `tile.inline_asm` of one operand tied to its result (`"=v,0"` or
    /// `"=s,0"`) whose template emits nothing, being empty or `;` comments
    /// only (`rocke.helpers.device_ops.fresh`'s `"; fresh $0"`).
    pub fn asm_pin(&self) -> Option<char> {
        if self.kind != OpKind::InlineAsmIdentity || self.operands.len() != 1 || self.results.len() != 1 {
            return None;
        }
        let silent = self.str_attr("template").is_some_and(|t| t.lines().all(|l| l.trim().is_empty() || l.trim_start().starts_with(';')));
        match self.str_attr("constraints") {
            Some("=v,0") if silent => Some('v'),
            Some("=s,0") if silent => Some('s'),
            _ => None,
        }
    }

    /// The register files (`'v'` / `'s'`) of a general `tile.inline_asm`'s
    /// outputs, from its constraint string (`None` for another op).
    pub fn asm_output_classes(&self) -> Option<Vec<char>> {
        if self.kind != OpKind::InlineAsmIdentity {
            return None;
        }
        let text = self.str_attr("constraints")?;
        Some(
            text.split(',')
                .map(str::trim)
                .filter_map(|p| p.strip_prefix('='))
                .map(|p| if p.trim_start_matches('&').starts_with('s') { 's' } else { 'v' })
                .collect(),
        )
    }

    pub fn set_attr(&mut self, key: &'static str, value: Attr) {
        if let Some(slot) = self.attrs.iter_mut().find(|(name, _)| name == key) {
            slot.1 = value;
        } else {
            self.attrs.push((key.into(), value));
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Region {
    pub ops: Vec<Op>,
}

impl Drop for Region {
    /// Takes nested regions apart from an explicit stack: the derived drop
    /// would recurse once per level of nesting.
    fn drop(&mut self) {
        let mut stack: Vec<Region> = Vec::new();
        for mut op in std::mem::take(&mut self.ops) {
            stack.append(&mut op.regions);
        }
        while let Some(mut region) = stack.pop() {
            for mut op in std::mem::take(&mut region.ops) {
                stack.append(&mut op.regions);
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Param {
    pub value: ValueId,
    pub name: String,
    pub ty: Type,
    /// `noalias = b:true`: memory accessed through a pointer based on this
    /// argument, if the kernel modifies it, is accessed through no pointer
    /// not based on it.
    pub noalias: bool,
    /// `readonly = b:true`: the kernel does not write through this argument.
    pub readonly: bool,
    /// `inreg = b:true`: preload the argument into user SGPRs at dispatch
    /// (gfx950 kernarg preload, `opt::kernarg_preload`).
    pub inreg: bool,
}

#[derive(Clone, Debug)]
pub struct Kernel {
    pub name: String,
    pub max_workgroup_size: u32,
    /// `waves_per_eu = [min, max]`: the occupancy range the kernel asks for
    /// (LLVM's `amdgpu-waves-per-eu`); 0 leaves a bound open. Not an
    /// occupancy cap: trading waves below the achievable count for a wider
    /// scheduling window is slower on gfx1151.
    pub waves_per_eu: (u32, u32),
    /// Lanes per wave of the GPU the kernel compiles for: the parser's 32
    /// (gfx1151) until the compile driver sets its target's.
    pub wave: u32,
    /// VGPRs the allocation may use when fewer than the target's: `None`
    /// from the parsers; `rir::compile_any_target_limited` sets it to test
    /// spilling.
    pub vgpr_limit: Option<u32>,
    /// AGPRs the allocation may use when fewer than the target's:
    /// `agpr_alloc = [0, 0]` (LLVM's `amdgpu-agpr-alloc`) allows none.
    pub agpr_limit: Option<u32>,
    pub params: Vec<Param>,
    pub body: Region,
    pub values: Vec<ValueDef>,
    /// Storage of the value names.
    pub names: String,
}

impl Kernel {
    pub fn ty(&self, value: ValueId) -> &Type {
        &self.values[value.0 as usize].ty
    }

    pub fn name_of(&self, value: ValueId) -> &str {
        self.values[value.0 as usize].name(&self.names)
    }

    pub fn new_value(&mut self, name: &str, ty: Type) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(ValueDef::new(&mut self.names, name, ty));
        id
    }

    /// Renumbers the values the kernel still refers to densely, keeping their
    /// order, and returns each old value's new number (`DROPPED` if unused).
    pub fn compact_values(&mut self) -> Vec<u32> {
        fn mark(region: &Region, map: &mut [u32]) {
            for op in &region.ops {
                for v in op.results.iter().chain(op.operands.iter()).chain(&op.block_args) {
                    map[v.0 as usize] = 0;
                }
                for region in &op.regions {
                    mark(region, map);
                }
            }
        }
        fn rename(region: &mut Region, map: &[u32]) {
            for op in &mut region.ops {
                for v in op.results.iter_mut().chain(op.operands.iter_mut()).chain(op.block_args.iter_mut()) {
                    v.0 = map[v.0 as usize];
                }
                for region in &mut op.regions {
                    rename(region, map);
                }
            }
        }
        let mut map = vec![DROPPED; self.values.len()];
        for param in &self.params {
            map[param.value.0 as usize] = 0;
        }
        mark(&self.body, &mut map);
        let mut next = 0;
        for (old, new) in map.iter_mut().enumerate() {
            if *new != DROPPED {
                *new = next;
                self.values.swap(next as usize, old);
                next += 1;
            }
        }
        self.values.truncate(next as usize);
        for param in &mut self.params {
            param.value.0 = map[param.value.0 as usize];
        }
        rename(&mut self.body, &map);
        map
    }
}

/// A value `Kernel::compact_values` dropped.
pub const DROPPED: u32 = u32::MAX;
