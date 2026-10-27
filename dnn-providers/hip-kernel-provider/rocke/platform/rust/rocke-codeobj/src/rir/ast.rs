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
}

impl Scalar {
    pub fn bytes(self) -> u32 {
        match self {
            Scalar::I1 | Scalar::I8 | Scalar::FP8E4M3 | Scalar::BF8E5M2 => 1,
            Scalar::F16 | Scalar::BF16 | Scalar::I16 => 2,
            Scalar::I32 | Scalar::F32 => 4,
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
            "memref.fence" => OpKind::Fence,
            "tile.sync_lds_only" => OpKind::SyncLdsOnly,
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
