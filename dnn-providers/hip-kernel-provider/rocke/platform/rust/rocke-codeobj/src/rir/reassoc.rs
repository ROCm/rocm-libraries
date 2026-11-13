// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Lever `cmpreassoc` (below) rebuilds compare operands the same way.
//!
//! rustgemm lever `sreassoc`: the constant part of an LDS DMA's scalar offset is added last.
//!
//! rocKE's MXFP6 GEMM computes each K tile's buffer `soffset` as a nest such as
//! `((kt * 12 + j + 3) * 32 + row_block) * 512 + wave_part` for tile `j` of an unrolled loop body: selected as written,
//! every tile pays its own add / shift / add chain (`s_add s46, s17, 0x80; s_lshl s46, s46, 9; s_add s46, s3, s46`).
//! The value is a linear form over the non-constant leaves (`kt`, `row_block`, `wave_part`) plus a constant that is
//! the only thing `j` changes. For the soffset operand of every `tile.async_buffer_load_lds*` whose form has a
//! non-zero constant and at least one leaf, this pass rebuilds the operand as `(invariant leaves + loop-variant leaves)
//! + constant`, each leaf scaled by its coefficient: the sum without the constant is the same op for every tile of the
//! body (the sweep's CSE keeps one) and its invariant half leaves the loop (LICM), so a tile pays one `s_add`.
//!
//! Exact: add, sub, mul and shl of i32 are ring operations modulo 2^32 (the rebuilt ops are `wrap = "modular"`), so
//! the rebuilt value equals the original for every input. Only the soffset operand changes; the old chain is left to
//! DCE. Leaves are values that are not i32 add / sub / mul-by-constant / shl-by-constant / constant (at most `ATOMS`
//! distinct, `DEPTH` levels), so nothing else is duplicated.

use super::ast::*;
use super::ring_form::{collect, Cx, Def, Emit, Form, Limits, Mono};

const ATOMS: usize = 6;
const DEPTH: u32 = 12;
/// Linear forms over at most `ATOMS` leaves: a product of two non-constants is a leaf, a larger form is refused.
const LIMITS: Limits = Limits { products: false, terms: ATOMS, leaf_on_overflow: false };

struct Rw<'a> {
    defs: Vec<Option<Def>>,
    home: Vec<u32>,
    loops: Vec<(u32, u32)>,
    next_loop: u32,
    kernel: &'a mut Kernel,
    rewritten: usize,
}

impl Emit for Rw<'_> {
    const PREFIX: &'static str = "%rsa";

    fn tables(&mut self) -> (&mut Kernel, &mut Vec<Option<Def>>, &mut Vec<u32>) {
        (&mut *self.kernel, &mut self.defs, &mut self.home)
    }
}

impl Rw<'_> {
    fn region(&mut self, region: &mut Region, at: u32) {
        let ops = std::mem::take(&mut region.ops);
        let mut out = Vec::with_capacity(ops.len());
        for mut op in ops {
            let inner = if op.kind == OpKind::For {
                self.next_loop += 1;
                self.next_loop
            } else {
                at
            };
            for r in &mut op.regions {
                self.region(r, inner);
            }
            if op.kind == OpKind::BufferLoadLds && at != 0 && super::op_lever(&op, "sreassoc") {
                if let Some(v) = self.rebuild(op.operands[3], at, &mut out) {
                    op.operands[3] = v;
                    self.rewritten += 1;
                }
            }
            // Lever `cmpreassoc`: both sides of an i32 compare in a loop as (leaves) + constant, so that the
            // constants of a row of masks (a causal mask's column index per accumulator element) stay immediates of
            // one shared base instead of each hoisting a register of its own out of the loop.
            if op.kind == OpKind::Cmp && at != 0 && op.operands.len() == 2 && super::op_lever(&op, "cmpreassoc") {
                for k in 0..2 {
                    if let Some(v) = self.rebuild(op.operands[k], at, &mut out) {
                        op.operands[k] = v;
                        self.rewritten += 1;
                    }
                }
            }
            out.push(op);
        }
        region.ops = out;
    }

    /// The soffset `s` as `(invariant + variant) + constant` for loop `l`, or None when it has no constant part, no
    /// leaf, is already `x + constant` with `x` free of constants, or is not a linear form.
    fn rebuild(&mut self, s: ValueId, l: u32, out: &mut Vec<Op>) -> Option<ValueId> {
        let form = Cx { defs: &self.defs, kernel: &*self.kernel, limits: LIMITS }.form(s, DEPTH)?;
        if form.konst == 0 || form.terms.is_empty() {
            return None;
        }
        if let Some(Some(d)) = self.defs.get(s.0 as usize) {
            if let (OpKind::Add, &[x, c]) = (d.kind, d.operands.as_slice()) {
                let cx = Cx { defs: &self.defs, kernel: &*self.kernel, limits: LIMITS };
                if cx.konst(c).is_some() && cx.form(x, DEPTH).is_some_and(|f| f.konst == 0) {
                    return None;
                }
            }
        }
        let last = self.loops[l as usize].1;
        let inside = |v: ValueId, home: &[u32]| home.get(v.0 as usize).is_some_and(|&h| (l..=last).contains(&h));
        let (var, inv): (Vec<_>, Vec<_>) = form.terms.iter().partition(|t| t.0.iter().any(|&v| inside(ValueId(v), &self.home)));
        let a = self.sum(inv, out);
        let b = self.sum(var, out);
        let base = match (a, b) {
            (Some(a), Some(b)) => self.bin(OpKind::Add, a, b, out),
            (Some(x), None) | (None, Some(x)) => x,
            (None, None) => return None,
        };
        let k = self.konst(form.konst, out);
        Some(self.bin(OpKind::Add, base, k, out))
    }
}

/// Rebuilds the LDS DMA soffsets of `kernel` (lever `sreassoc`, per op); returns how many.
pub fn run(kernel: &mut Kernel) -> usize {
    let n = kernel.values.len();
    let mut defs = vec![None; n];
    let mut home = vec![0u32; n];
    let mut loops = vec![(0u32, 0u32)];
    collect(&kernel.body, &mut defs, &mut home, &mut loops, 0);
    let mut body = std::mem::take(&mut kernel.body);
    let mut rw = Rw { defs, home, loops, next_loop: 0, kernel, rewritten: 0 };
    rw.region(&mut body, 0);
    let rewritten = rw.rewritten;
    kernel.body = body;
    rewritten
}

/// rustgemm lever `m0share`: an LDS DMA whose LDS address is the previous DMA's plus a constant `d` (both of one
/// leaf sum, e.g. the MXFP6 GEMM's A- and B-scale tiles at `alloc + wave * 256`) takes the previous DMA's address
/// with instruction offset `offset + d` and soffset `B + (K - d)` for its `B + K`, when `offset + d` fits the 12-bit
/// field, the soffset is an add of a constant (`sreassoc` makes it one) and `K >= d` (`K - d` stays non-negative, see
/// below). Both DMAs then read one M0 value and `m0cse` drops the second move to M0 (one SALU per K tile in the MXFP6
/// GEMM).
///
/// Exact for any M0 state: the instruction offset adds to both addresses of a buffer DMA (LDS: M0 + offset + 4 *
/// dwords * lane; memory: base + voffset + soffset + offset), so the LDS bytes written and the memory read are the
/// same. LDS addresses use their low dword (modulo 2^32), but the memory address does not wrap the soffset: a
/// rewritten soffset `B + (K - d)` below zero modulo 2^32 (K < d, B + K < d at run time) reads 4 GiB away. Hence
/// `K >= d` (2026-11-08: canonical MXFP6 `canon_dtl="band"` DMAs of one row group, d = 1 KiB, K = 0 at K tile 0
/// read wrong rows; out-68317511 rewrote them). The bounds check sees `voffset + offset`, larger by
/// `d`: the rewrite needs the resource's `num_records` to be a constant of at least 2^30. The DMA keeps its own LDS
/// range for the waits (attribute `dma_tag`, `rir::isel::buffer_load_lds`).
pub fn share_m0(kernel: &mut Kernel, lds: &super::fx::FxHashMap<ValueId, u32>) -> usize {
    let n = kernel.values.len();
    let mut defs = vec![None; n];
    let mut home = vec![0u32; n];
    let mut loops = vec![(0u32, 0u32)];
    collect(&kernel.body, &mut defs, &mut home, &mut loops, 0);
    let mut body = std::mem::take(&mut kernel.body);
    let mut rw = Rw { defs, home, loops, next_loop: 0, kernel, rewritten: 0 };
    rw.share_region(&mut body, lds);
    let shared = rw.rewritten;
    kernel.body = body;
    shared
}

/// An LDS address as (leaf terms, constant) modulo 2^32, and the allocation it is based on.
fn lds_form(defs: &[Option<Def>], lds: &super::fx::FxHashMap<ValueId, u32>, v: ValueId, depth: u32) -> Option<(Form, Option<ValueId>)> {
    let Some(Some(d)) = defs.get(v.0 as usize) else { return Some((Form::leaf(v), None)) };
    if let Some(k) = d.konst {
        return Some((Form { terms: Vec::new(), konst: k as u32 }, None));
    }
    if depth == 0 {
        return Some((Form::leaf(v), None));
    }
    let both = |a: ValueId, b: ValueId, sign: u32| -> Option<(Form, Option<ValueId>)> {
        let (fa, aa) = lds_form(defs, lds, a, depth - 1)?;
        let (fb, ab) = lds_form(defs, lds, b, depth - 1)?;
        let alloc = match (aa, ab) {
            (Some(_), Some(_)) => return None,
            (x, None) | (None, x) => x,
        };
        Some((fa.add(&fb, sign), alloc))
    };
    let konst = |x: ValueId| defs.get(x.0 as usize).and_then(|d| d.as_ref()).and_then(|d| d.konst).map(|k| k as u32);
    let f = match (d.kind, d.operands.as_slice()) {
        (OpKind::SmemAddrOf, &[alloc]) => (Form { terms: Vec::new(), konst: *lds.get(&alloc)? }, Some(alloc)),
        (OpKind::SmemPtrAdd | OpKind::Add, &[a, b]) => both(a, b, 1)?,
        (OpKind::Sub, &[a, b]) => both(a, b, u32::MAX)?,
        (OpKind::Zext, &[a]) => lds_form(defs, lds, a, depth - 1)?,
        (OpKind::Mul, &[a, b]) => match konst(b) {
            Some(k) => {
                let (f, al) = lds_form(defs, lds, a, depth - 1)?;
                if al.is_some() && k != 1 {
                    return None;
                }
                (f.scale(k), al)
            }
            None => (Form::leaf(v), None),
        },
        (OpKind::Shl, &[a, b]) => match konst(b) {
            Some(k) if k < 32 => {
                let (f, al) = lds_form(defs, lds, a, depth - 1)?;
                if al.is_some() && k != 0 {
                    return None;
                }
                (f.scale(1u32 << k), al)
            }
            _ => (Form::leaf(v), None),
        },
        _ => (Form::leaf(v), None),
    };
    (f.0.terms.len() <= ATOMS).then_some(f)
}

impl Rw<'_> {
    fn share_region(&mut self, region: &mut Region, lds: &super::fx::FxHashMap<ValueId, u32>) {
        let ops = std::mem::take(&mut region.ops);
        let mut out = Vec::with_capacity(ops.len());
        // The previous DMA of this region: its LDS address value and form.
        let mut last: Option<(ValueId, Form)> = None;
        for mut op in ops {
            if !op.regions.is_empty() || op.kind == OpKind::InlineAsmIdentity || op.kind == OpKind::GlobalLoadLds {
                for r in &mut op.regions {
                    self.share_region(r, lds);
                }
                last = None;
                out.push(op);
                continue;
            }
            if op.kind != OpKind::BufferLoadLds {
                out.push(op);
                continue;
            }
            let addr = op.operands[1];
            let offset = op.int_attr("offset").unwrap_or(0);
            let Some((form, alloc)) = lds_form(&self.defs, lds, addr, DEPTH) else {
                last = None;
                out.push(op);
                continue;
            };
            let prev = last.replace((addr, form.clone()));
            let Some((paddr, pform)) = prev.filter(|_| super::op_lever(&op, "m0share")) else {
                out.push(op);
                continue;
            };
            let d = form.konst.wrapping_sub(pform.konst) as i32 as i64;
            let new_offset = offset + d;
            let tag = alloc.and_then(|a| {
                let base = *lds.get(&a)?;
                Some(crate::target::gfx9::lds_dma::tag(base, crate::target::gfx9::lds_dma::alloc_bytes(self.kernel.ty(a))))
            });
            let fits = form.terms == pform.terms && paddr != addr && (0..4096).contains(&new_offset) && self.big_records(op.operands[0]);
            // The soffset is an unsigned 32-bit add of its own: the address is base + soffset + voffset + offset
            // without a wrap modulo 2^32, so B + (K - d) must not wrap below zero where B + K does not -- K >= d
            // (B >= 0) -- nor above 2^32 (d < 0: K - d <= u32::MAX).
            let no_wrap = |k: u32| if d >= 0 { k >= d as u32 } else { k.checked_add(d.unsigned_abs() as u32).is_some() };
            if let (true, Some((b, k)), Some(tag)) = (fits, self.add_const(op.operands[3]).filter(|&(_, k)| no_wrap(k)), tag) {
                let kv = self.konst(k.wrapping_sub(d as u32), &mut out);
                let s = self.bin(OpKind::Add, b, kv, &mut out);
                op.operands[1] = paddr;
                op.operands[3] = s;
                op.attrs.retain(|(name, _)| name != "offset" && name != "dma_tag");
                op.attrs.push(("offset".into(), Attr::Int(new_offset)));
                op.attrs.push(("dma_tag".into(), Attr::Int(tag as i64)));
                self.rewritten += 1;
                // M0 keeps the previous DMA's value.
                last = Some((paddr, pform));
            }
            out.push(op);
        }
        region.ops = out;
    }

    /// `(x, k)` when `v` is `x + k` (either order) with `k` a constant.
    fn add_const(&self, v: ValueId) -> Option<(ValueId, u32)> {
        let d = self.defs.get(v.0 as usize)?.as_ref()?;
        let konst = |x: ValueId| self.defs.get(x.0 as usize).and_then(|d| d.as_ref()).and_then(|d| d.konst).map(|k| k as u32);
        match (d.kind, d.operands.as_slice()) {
            (OpKind::Add, &[a, b]) => match (konst(a), konst(b)) {
                (_, Some(k)) => Some((a, k)),
                (Some(k), _) => Some((b, k)),
                _ => None,
            },
            _ => None,
        }
    }

    /// Whether the buffer resource `r` is a `tile.buffer_rsrc` whose `num_records` is a constant of at least 2^30.
    fn big_records(&self, r: ValueId) -> bool {
        let Some(Some(d)) = self.defs.get(r.0 as usize) else { return false };
        let records = d.operands.get(1).and_then(|n| self.defs.get(n.0 as usize)).and_then(|x| x.as_ref()).and_then(|x| x.konst);
        d.kind == OpKind::BufferRsrc && records.is_some_and(|k| (1i64 << 30..1i64 << 32).contains(&k))
    }
}

/// rustgemm lever `soffshare`: in a loop body, an LDS DMA whose soffset differs from an earlier DMA's of the same body
/// only by a loop-invariant amount `D` (the same loop-variant leaves; e.g. the MXFP6 GEMM's second operand chunk at
/// `+ 0x1000`, or its B-scale tile next to the A-scale tile) reads that DMA's soffset register and a buffer resource
/// whose base is `D` bytes further: `buffer_rsrc(ptr + sext(D), records)`, invariant, so LICM builds it once before
/// the loop. One `s_add` per such DMA and K tile goes.
///
/// Exact: the memory address is base + soffset + voffset + offset, and (base + D) + (S - D) = base + S, with each
/// soffset the unwrapped offset of a real DMA; the bounds check (voffset + offset against num_records) and the LDS
/// address do not change. Needs the DMA's resource to be a `tile.buffer_rsrc` defined outside the loop.
pub fn share_soffsets(kernel: &mut Kernel) -> usize {
    let n = kernel.values.len();
    let mut defs = vec![None; n];
    let mut home = vec![0u32; n];
    let mut loops = vec![(0u32, 0u32)];
    collect(&kernel.body, &mut defs, &mut home, &mut loops, 0);
    let mut body = std::mem::take(&mut kernel.body);
    let mut rw = Rw { defs, home, loops, next_loop: 0, kernel, rewritten: 0 };
    rw.soff_region(&mut body, 0);
    let shared = rw.rewritten;
    kernel.body = body;
    shared
}

impl Rw<'_> {
    fn soff_region(&mut self, region: &mut Region, at: u32) {
        let ops = std::mem::take(&mut region.ops);
        let mut out = Vec::with_capacity(ops.len());
        // Earlier DMAs of this body: (soffset value, its variant terms, invariant terms, constant).
        let mut seen: Vec<(ValueId, Vec<(Mono, u32)>, Vec<(Mono, u32)>, u32)> = Vec::new();
        for mut op in ops {
            let inner = if op.kind == OpKind::For {
                self.next_loop += 1;
                self.next_loop
            } else {
                at
            };
            for r in &mut op.regions {
                self.soff_region(r, inner);
            }
            if op.kind == OpKind::BufferLoadLds && at != 0 {
                let s = op.operands[3];
                let last = self.loops[at as usize].1;
                let form = Cx { defs: &self.defs, kernel: &*self.kernel, limits: LIMITS }.form(s, DEPTH);
                if let Some(form) = form {
                    let inside = |v: ValueId| self.home.get(v.0 as usize).is_some_and(|&h| (at..=last).contains(&h));
                    let (var, inv): (Vec<_>, Vec<_>) = form.terms.iter().cloned().partition(|t| t.0.iter().any(|&v| inside(ValueId(v))));
                    // Lever `soffsharek`: only differences that are constants (no new invariant leaves in the resource).
                    let konly = super::op_lever(&op, "soffsharek");
                    if konly || super::op_lever(&op, "soffshare") {
                        // Only the DMA right before this one: a bounded number of rebased resources (one per pair).
                        let hit = seen.last().filter(|(sv, v, si, _)| *v == var && *sv != s && (!konly || *si == inv)).cloned();
                        let rsrc = self.outside_rsrc(op.operands[0], at, last);
                        if let (Some((sv, sv_var, sinv, sk)), Some((ptr, records))) = (hit, rsrc) {
                            // D = (inv - sinv) + (k - sk)
                            let d = Form { terms: inv.clone(), konst: form.konst }.add(&Form { terms: sinv.clone(), konst: sk }, u32::MAX);
                            let r = self.rebased_rsrc(ptr, records, &d, &mut out);
                            op.operands[0] = r;
                            op.operands[3] = sv;
                            self.rewritten += 1;
                            // This DMA now reads the shared soffset: the next one compares with that value.
                            seen.push((sv, sv_var, sinv, sk));
                            out.push(op);
                            continue;
                        }
                    }
                    seen.push((op.operands[3], var, inv, form.konst));
                }
            }
            out.push(op);
        }
        region.ops = out;
    }

    /// `(ptr, records)` when `r` is a `tile.buffer_rsrc ( ptr, records )` whose operands are defined outside loop `at`.
    fn outside_rsrc(&self, r: ValueId, at: u32, last: u32) -> Option<(ValueId, ValueId)> {
        let d = self.defs.get(r.0 as usize)?.as_ref()?;
        let outside = |v: ValueId| self.home.get(v.0 as usize).is_some_and(|&h| !(at..=last).contains(&h));
        match (d.kind, d.operands.as_slice()) {
            (OpKind::BufferRsrc, &[p, n]) if outside(p) && outside(n) && outside(r) => Some((p, n)),
            _ => None,
        }
    }

    fn rebased_rsrc(&mut self, ptr: ValueId, records: ValueId, d: &Form, out: &mut Vec<Op>) -> ValueId {
        let mut sum = self.sum(&d.terms, out);
        if d.konst != 0 || sum.is_none() {
            let k = self.konst(d.konst, out);
            sum = Some(match sum {
                Some(x) => self.bin(OpKind::Add, x, k, out),
                None => k,
            });
        }
        let d32 = sum.expect("a sum");
        let d64 = self.kernel.new_value("%rsd", Type::Scalar(Scalar::I64));
        self.defs.push(None);
        self.home.push(0);
        out.push(Op { kind: OpKind::Sext, results: [d64].into(), operands: [d32].into(), attrs: Vec::new(), regions: Vec::new(), block_args: Vec::new(), loc: None });
        let pty = self.kernel.ty(ptr).clone();
        let p = self.kernel.new_value("%rsp", pty);
        self.defs.push(None);
        self.home.push(0);
        out.push(Op { kind: OpKind::GlobalPtrAdd, results: [p].into(), operands: [ptr, d64].into(), attrs: Vec::new(), regions: Vec::new(), block_args: Vec::new(), loc: None });
        let r = self.kernel.new_value("%rsr", Type::Vec(Scalar::I32, 4));
        self.defs.push(None);
        self.home.push(0);
        out.push(Op { kind: OpKind::BufferRsrc, results: [r].into(), operands: [p, records].into(), attrs: Vec::new(), regions: Vec::new(), block_args: Vec::new(), loc: None });
        r
    }
}

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    /// A loop of `trips` K tiles, `per` LDS DMAs per trip, each with soffset `((kt * per + j) * 32 + %rb) * 512 + %wp`
    /// (the MXFP6 GEMM's scale-tile offset): `%rb` / `%wp` are kernel arguments.
    fn kernel(per: usize) -> String {
        let mut body = String::new();
        for j in 0..per {
            body += &format!(
                "        %cj{j} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{j} }}\n        \
                 %t{j} : i32 = arith.add ( %ktp, %cj{j} )\n        \
                 %u{j} : i32 = arith.mul ( %t{j}, %c32 )\n        \
                 %w{j} : i32 = arith.add ( %u{j}, %rb )\n        \
                 %x{j} : i32 = arith.mul ( %w{j}, %c512 )\n        \
                 %s{j} : i32 = arith.add ( %wp, %x{j} )\n        \
                 %l{j} : i32 = arith.mul ( %cj{j}, %c256 )\n        \
                 %la{j} : i32 = arith.add ( %l{j}, %lw )\n        \
                 %lz{j} : i64 = arith.zext ( %la{j} )\n        \
                 %lp{j} : i64 = tile.smem_ptr_add ( %base, %lz{j} )\n        \
                 tile.async_buffer_load_lds_addr ( %rsrc, %lp{j}, %vo, %s{j} ) {{ aux = i:0, dwords = i:1 }}\n"
            );
        }
        format!(
            "ckdsl.ir v1\nkernel @k {{\n  attrs {{ max_workgroup_size = i:256 }}\n  params {{\n    %X : ptr<i32,global>\n    \
             %Y : ptr<i32,global>\n    %rb : i32\n    %wp : i32\n    %n : i32\n  }}\n  region @entry {{\n    \
             %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    \
             %c1 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:1 }}\n    \
             %c4 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:4 }}\n    \
             %cper : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{per} }}\n    \
             %c32 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:32 }}\n    \
             %c256 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:256 }}\n    \
             %c512 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:512 }}\n    \
             %cbig : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:2147483632 }}\n    \
             %vo : i32 = arith.mul ( %tid, %c4 )\n    \
             %lw0 : i32 = arith.div ( %tid, %c32 )\n    \
             %lw1 : i32 = tile.readfirstlane ( %lw0 )\n    \
             %lw : i32 = arith.mul ( %lw1, %c256 )\n    \
             %t : smem<i32, [4096]> = tile.smem_alloc (  )\n    \
             %base : i64 = tile.smem_addr_of ( %t )\n    \
             %rsrc : vec<i32x4> = tile.buffer_rsrc ( %X, %cbig )\n    \
             scf.for ( %c0, %n, %c1 ) {{ iv = s:\"%kt\", iv_type = s:\"i32\", num_iter_args = i:0, unroll = b:false }}\n      \
             region @body {{\n        %ktp : i32 = arith.mul ( %kt, %cper )\n{body}        scf.yield (  ) {{ num = i:0 }}\n      }}\n    \
             tile.sync (  )\n    \
             %rv : vec<i32x2> = tile.smem_load_vN ( %t, %tid ) {{ elem_type = s:\"i32\", rank = i:1, vec = i:2 }}
    %r : i32 = vector.extract ( %rv ) {{ index = i:0 }}\n    \
             memref.global_store_typed ( %Y, %tid, %r ) {{ align = i:4, elem_type = s:\"i32\" }}\n    \
             cf.return (  )\n  }}\n}}\n"
        )
    }

    fn loop_salu(listing: &str) -> usize {
        // The loop is the block that ends in the backward branch: count its scalar ALU instructions.
        let mut best = 0;
        let mut cur = 0;
        for l in listing.lines() {
            if l.starts_with("bb") {
                cur = 0;
                continue;
            }
            let t = l.trim_start();
            if t.starts_with("s_add") || t.starts_with("s_mul") || t.starts_with("s_lshl") || t.starts_with("s_sub") {
                cur += 1;
            }
            if t.starts_with("s_cbranch_scc") {
                best = best.max(cur);
            }
        }
        best
    }

    fn compiled(levers: &[&str], per: usize) -> rir::CompiledKernel {
        rir::with_levers(levers, || rir::compile_any_target(&kernel(per), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")))
    }

    #[test]
    fn each_tile_pays_one_add_for_its_soffset() {
        let base = compiled(&[], 4);
        let lever = compiled(&["sreassoc"], 4);
        let (b, l) = (loop_salu(&base.listing()), loop_salu(&lever.listing()));
        // Without the lever every tile builds its own add / shift / add chain; with it each tile adds its constant to
        // one per-trip base (plus the trip's counter and base updates).
        assert!(l + 4 <= b, "salu per trip {b} -> {l}\n{}", lever.listing());
        assert!(base.hsaco != lever.hsaco);
    }

    #[test]
    fn the_rebuilt_offset_is_the_same_linear_form() {
        use super::*;
        let mut k = rir::parse::parse(&kernel(2)).unwrap_or_else(|e| panic!("{e}"));
        let before: Vec<Form> = soffset_forms(&k);
        let n = rir::with_levers(&["sreassoc"], || run(&mut k));
        // Tile 0's offset has no constant part: only tile 1's is rebuilt.
        assert_eq!(n, 1);
        let after: Vec<Form> = soffset_forms(&k);
        assert_eq!(before, after);
        assert!(before.iter().all(|f| f.terms.len() == 3) && before[0].konst == 0 && before[1].konst == 32 * 512, "{before:?}");
    }

    /// A loop whose body compares `col + (lane + k)` (col per trip) with a row for `per` constants `k`, as a causal mask
    /// does for each accumulator element, and counts the true ones.
    fn mask_kernel(per: usize) -> String {
        let mut body = String::new();
        for k in 0..per {
            body += &format!(
                "        %ck{k} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n        \
                 %o{k} : i32 = arith.add ( %lane, %ck{k} )\n        \
                 %t{k} : i32 = arith.add ( %col, %o{k} )\n        \
                 %le{k} : i1 = arith.cmp ( %t{k}, %row ) {{ pred = s:\"le\" }}\n        \
                 %s{k} : i32 = arith.select ( %le{k}, %c1, %c0 )\n        \
                 %a{} : i32 = arith.add ( %a{k}, %s{k} )\n",
                k * 3 + 1,
                k + 1
            );
        }
        format!(
            "ckdsl.ir v1\nkernel @maskk {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %Y : ptr<i32,global>\n    \
             %row : i32\n    %n : i32\n  }}\n  region @entry {{\n    \
             %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    \
             %c1 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:1 }}\n    \
             %c4 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:4 }}\n    \
             %c128 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:128 }}\n    \
             %lane : i32 = arith.mul ( %tid, %c4 )\n    \
             %r : i32 = scf.for ( %c0, %n, %c1, %c0 ) {{ iter_args = l:[ {{ name = s:\"%a0\", type = s:\"i32\" }} ], iv = s:\"%kt\", \
             iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:false }}\n      \
             region @body {{\n        %it : i32 = arith.mul ( %kt, %c128 )\n        %col : i32 = arith.add ( %it, %tid )\n{body}        \
             scf.yield ( %a{per} ) {{ num = i:1 }}\n      }}\n    \
             memref.global_store_typed ( %Y, %tid, %r ) {{ align = i:4, elem_type = s:\"i32\" }}\n    \
             cf.return (  )\n  }}\n}}\n"
        )
    }

    #[test]
    fn cmpreassoc_keeps_the_constants_of_a_row_of_compares_out_of_registers() {
        let build = |levers: &[&str]| rir::with_levers(levers, || rir::compile_any_target(&mask_kernel(24), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")));
        let (base, lever) = (build(&[]), build(&["cmpreassoc"]));
        assert!(base.hsaco != lever.hsaco);
        // Without the lever each `lane + k` is hoisted into a register of its own; with it, one base per trip.
        assert!(lever.stats.vgprs + 12 <= base.stats.vgprs, "vgprs {} -> {}", base.stats.vgprs, lever.stats.vgprs);
        // The rewrite is exact: the same compares on the same sums.
        let mut k = rir::parse::parse(&mask_kernel(3)).unwrap_or_else(|e| panic!("{e}"));
        let n = rir::with_levers(&["cmpreassoc"], || super::run(&mut k));
        // Each compare's left side has a constant part (k * 3 + 1); the right side has none.
        assert_eq!(n, 3);
    }

    #[test]
    fn rematv_recomputes_a_row_of_hoisted_offsets_instead_of_spilling_them() {
        // 24 `lane + k` values live through the loop in 20 VGPRs: spilled without the lever, recomputed with it.
        let build = |levers: &[&str]| {
            rir::with_levers(levers, || rir::compile_any_target_limited(&mask_kernel(24), Target::Gfx950, 20).unwrap_or_else(|e| panic!("{e}")))
        };
        let (base, lever) = (build(&[]), build(&["rematv"]));
        let moved = |s: &rir::regalloc::SpillStats| s.lds_dwords + s.scratch_dwords + s.agpr_dwords;
        assert!(moved(&base.stats.spill) > 0, "{:?}", base.stats.spill);
        assert!(lever.stats.spill.remat > base.stats.spill.remat, "{:?} vs {:?}", base.stats.spill, lever.stats.spill);
        assert!(moved(&lever.stats.spill) < moved(&base.stats.spill), "{:?} vs {:?}", base.stats.spill, lever.stats.spill);
    }

    /// Two DMAs of one wave-uniform LDS part into the 1 KiB allocations %A and %B (records `records`; the second
    /// DMA's soffset `%sy + k2`), then a read of %B with no barrier, stored to %Y.
    fn two_dmas(records: i64, k2: i64) -> String {
        format!(
            "ckdsl.ir v1\nkernel @m0s {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    \
             %X : ptr<f32,global> {{ align = i:16, noalias = b:true, readonly = b:true }}\n    %Y : ptr<i32,global>\n    \
             %w : i32\n    %sx : i32\n    %sy : i32\n  }}\n  region @entry {{\n    \
             %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %c4 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:4 }}\n    \
             %c16 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:16 }}\n    \
             %c32 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{k2} }}\n    \
             %size : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{records} }}\n    \
             %A : smem<i32, [256]> = tile.smem_alloc (  )\n    %B : smem<i32, [256]> = tile.smem_alloc (  )\n    \
             %rsrc : vec<i32x4> = tile.buffer_rsrc ( %X, %size )\n    \
             %a0 : i64 = tile.smem_addr_of ( %A )\n    %b0 : i64 = tile.smem_addr_of ( %B )\n    \
             %wz : i64 = arith.zext ( %w )\n    \
             %a1 : i64 = tile.smem_ptr_add ( %a0, %wz )\n    %b1 : i64 = tile.smem_ptr_add ( %b0, %wz )\n    \
             %voff : i32 = arith.mul ( %tid, %c4 )\n    \
             %s1 : i32 = arith.add ( %sx, %c16 )\n    %s2 : i32 = arith.add ( %sy, %c32 )\n    \
             tile.async_buffer_load_lds_addr ( %rsrc, %a1, %voff, %s1 ) {{ aux = i:0, dwords = i:1 }}\n    \
             tile.async_buffer_load_lds_addr ( %rsrc, %b1, %voff, %s2 ) {{ aux = i:0, dwords = i:1 }}\n    \
             %rv : vec<i32x2> = tile.smem_load_vN ( %B, %tid ) {{ elem_type = s:\"i32\", rank = i:1, vec = i:2 }}\n    \
             %r : i32 = vector.extract ( %rv ) {{ index = i:0 }}\n    \
             memref.global_store_typed ( %Y, %tid, %r ) {{ align = i:4, elem_type = s:\"i32\" }}\n    \
             cf.return (  )\n  }}\n}}\n"
        )
    }

    /// vmcnt of each `s_waitcnt` in `lines`.
    fn vmcnts(lines: &[&str]) -> Vec<u32> {
        lines
            .iter()
            .filter(|l| l.trim_start().starts_with("s_waitcnt"))
            .map(|l| {
                let imm = u32::from_str_radix(l.rsplit(": ").next().unwrap_or("").trim(), 16).expect("a hex word") & 0xffff;
                imm & 0xf | (imm >> 14 & 3) << 4
            })
            .collect()
    }

    #[test]
    fn m0share_reuses_the_neighbours_m0_and_keeps_the_wait_on_its_own_range() {
        let build = |levers: &[&str], records: i64| {
            rir::with_levers(levers, || rir::compile_any_target(&two_dmas(records, 2048), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing())
        };
        let m0 = |l: &str| l.lines().filter(|x| x.trim_start().starts_with("s_mov_b32 m0")).count();
        let base = build(&["m0cse"], 0x7fff_fff0);
        let shared = build(&["m0cse", "m0share"], 0x7fff_fff0);
        assert_eq!(m0(&base), 2, "{base}");
        assert_eq!(m0(&shared), 1, "{shared}");
        // The second DMA: %B's offset from %A in its instruction offset (and in neither address register).
        let dmas: Vec<&str> = shared.lines().filter(|l| l.contains("buffer_load_dword")).collect();
        assert_eq!(dmas.len(), 2, "{shared}");
        assert!(dmas[1].contains("imm:1024") || dmas[1].contains("imm:0x400"), "{shared}");
        // The read of %B still waits for the DMA into %B (the youngest): vmcnt(0) between them.
        let lines: Vec<&str> = shared.lines().collect();
        let dma = lines.iter().rposition(|l| l.contains("buffer_load_dword")).expect("a DMA");
        let read = lines.iter().position(|l| l.trim_start().starts_with("ds_read")).expect("the read");
        assert!(dma < read && vmcnts(&lines[dma..read]).contains(&0), "{shared}");
        // A resource with a small record count keeps both moves.
        assert_eq!(m0(&build(&["m0cse", "m0share"], 65536)), 2);
    }

    /// Per DMA: (memory offset form = soffset + instruction offset, LDS address form + instruction offset).
    fn dma_addresses(k: &super::Kernel, lds: &crate::rir::fx::FxHashMap<super::ValueId, u32>) -> Vec<(super::Form, super::Form)> {
        use super::*;
        let n = k.values.len();
        let mut defs = vec![None; n];
        let mut home = vec![0u32; n];
        let mut loops = vec![(0u32, 0u32)];
        collect(&k.body, &mut defs, &mut home, &mut loops, 0);
        let cx = Cx { defs: &defs, kernel: k, limits: LIMITS };
        let mut out = Vec::new();
        rir::analysis::preorder(&k.body, |op| {
            if op.kind == OpKind::BufferLoadLds {
                let off = Form { terms: Vec::new(), konst: op.int_attr("offset").unwrap_or(0) as u32 };
                let mem = cx.form(op.operands[3], DEPTH).expect("a linear soffset").add(&off, 1);
                let (l, _) = lds_form(&defs, lds, op.operands[1], DEPTH).expect("a linear LDS address");
                out.push((mem, l.add(&off, 1)));
            }
        });
        out
    }

    #[test]
    fn m0share_keeps_both_addresses_of_every_dma() {
        let mut k = rir::parse::parse(&two_dmas(0x7fff_fff0, 2048)).unwrap_or_else(|e| panic!("{e}"));
        let (lds, _) = rir::analysis::lds_layout(&k);
        let before = dma_addresses(&k, &lds);
        assert_eq!(rir::with_levers(&["m0share"], || super::share_m0(&mut k, &lds)), 1);
        let after = dma_addresses(&k, &lds);
        assert_eq!(before, after);
        // the rewritten soffset keeps a non-negative constant: no wrap below zero for any %sy
        let ks: Vec<u32> = after.iter().map(|(m, _)| m.konst).collect();
        assert!(ks.iter().all(|&c| c < 1 << 31), "{ks:?}");
    }

    /// Regression (canonical MXFP6 `canon_dtl="band"` + m0share, out-68317511: wrong rows at K tile 0): a DMA whose
    /// soffset constant K is below the LDS distance d keeps its own M0 -- `B + (K - d)` would wrap below zero, and the
    /// memory address does not wrap the soffset (it read 4 GiB away). From K = d on it shares.
    #[test]
    fn m0share_never_wraps_the_soffset_below_zero() {
        for (k2, want) in [(32, 0), (1023, 0), (1024, 1), (4096, 1)] {
            let mut k = rir::parse::parse(&two_dmas(0x7fff_fff0, k2)).unwrap_or_else(|e| panic!("{e}"));
            let (lds, _) = rir::analysis::lds_layout(&k);
            assert_eq!(rir::with_levers(&["m0share"], || super::share_m0(&mut k, &lds)), want, "K = {k2}");
        }
    }

    /// Per DMA of `k`: (base pointer, form of soffset + the resource's offset from that pointer + instruction offset).
    fn dma_memory(k: &super::Kernel) -> Vec<(super::ValueId, super::Form)> {
        use super::*;
        let n = k.values.len();
        let mut defs = vec![None; n];
        let mut home = vec![0u32; n];
        let mut loops = vec![(0u32, 0u32)];
        collect(&k.body, &mut defs, &mut home, &mut loops, 0);
        let cx = Cx { defs: &defs, kernel: k, limits: LIMITS };
        let def = |v: ValueId| defs[v.0 as usize].clone().expect("defined");
        let mut out = Vec::new();
        rir::analysis::preorder(&k.body, |op| {
            if op.kind == OpKind::BufferLoadLds {
                let off = Form { terms: Vec::new(), konst: op.int_attr("offset").unwrap_or(0) as u32 };
                let mut f = cx.form(op.operands[3], DEPTH).expect("a linear soffset").add(&off, 1);
                let r = def(op.operands[0]);
                assert_eq!(r.kind, OpKind::BufferRsrc);
                let mut ptr = r.operands[0];
                if let Some(Some(p)) = defs.get(ptr.0 as usize) {
                    if p.kind == OpKind::GlobalPtrAdd {
                        let s = def(p.operands[1]);
                        assert_eq!(s.kind, OpKind::Sext);
                        f = f.add(&cx.form(s.operands[0], DEPTH).expect("a linear delta"), 1);
                        ptr = p.operands[0];
                    }
                }
                out.push((ptr, f));
            }
        });
        out
    }

    #[test]
    fn soffshare_moves_the_difference_into_the_resource_and_keeps_every_address() {
        let mut k = rir::parse::parse(&kernel(4)).unwrap_or_else(|e| panic!("{e}"));
        let before = dma_memory(&k);
        assert_eq!(rir::with_levers(&["soffshare"], || super::share_soffsets(&mut k)), 3);
        let after = dma_memory(&k);
        assert_eq!(before, after);
        // Every DMA of the trip reads tile 0's soffset value.
        let mut soffsets = Vec::new();
        rir::analysis::preorder(&k.body, |op| {
            if op.kind == super::OpKind::BufferLoadLds {
                soffsets.push(op.operands[3]);
            }
        });
        assert!(soffsets.iter().all(|&s| s == soffsets[0]), "{soffsets:?}");
        // Compiled: the loop pays no soffset arithmetic per tile beyond tile 0's.
        let base = compiled(&[], 4);
        let shared = compiled(&["soffshare"], 4);
        assert!(loop_salu(&shared.listing()) + 3 <= loop_salu(&base.listing()), "{}", shared.listing());
    }

    #[test]
    fn soffsharek_takes_constant_differences_only() {
        // Tile 1's offset uses %n where tile 0's uses %rb: the difference has an invariant leaf.
        let text = kernel(2).replacen("%w1 : i32 = arith.add ( %u1, %rb )", "%w1 : i32 = arith.add ( %u1, %n )", 1);
        let shares = |lever: &str, text: &str| {
            let mut k = rir::parse::parse(text).unwrap_or_else(|e| panic!("{e}"));
            let before = dma_memory(&k);
            let n = rir::with_levers(&[lever], || super::share_soffsets(&mut k));
            assert_eq!(before, dma_memory(&k));
            n
        };
        assert_eq!((shares("soffshare", &text), shares("soffsharek", &text)), (1, 0));
        assert_eq!((shares("soffshare", &kernel(2)), shares("soffsharek", &kernel(2))), (1, 1));
    }

    #[test]
    fn soffshare_pairs_each_dma_with_the_one_right_before() {
        // A DMA of another stream (soffset `kt * 4`) first in the body: the tile DMAs still pair with their
        // predecessor, and that first DMA (a different loop-variant part) pairs with nothing.
        let text = kernel(3).replacen(
            "%ktp : i32 = arith.mul ( %kt, %cper )\n",
            "%ktp : i32 = arith.mul ( %kt, %cper )\n        \
             tile.async_buffer_load_lds_addr ( %rsrc, %base, %vo, %ktp ) { aux = i:0, dwords = i:1 }\n",
            1,
        );
        let mut k = rir::parse::parse(&text).unwrap_or_else(|e| panic!("{e}"));
        let before = dma_memory(&k);
        assert_eq!(rir::with_levers(&["soffshare"], || super::share_soffsets(&mut k)), 2);
        assert_eq!(before, dma_memory(&k));
    }

    #[test]
    fn ldsgroup_puts_the_small_allocations_first() {
        use super::ValueId;
        let text = "ckdsl.ir v1\nkernel @g {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %Y : ptr<i32,global>\n  }\n  \
             region @entry {\n    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
             %A : smem<i32, [256]> = tile.smem_alloc (  )\n    %G : smem<i32, [2048]> = tile.smem_alloc (  )\n    \
             %B : smem<i32, [256]> = tile.smem_alloc (  )\n    \
             %x : vec<i32x2> = tile.smem_load_vN ( %A, %tid ) { elem_type = s:\"i32\", rank = i:1, vec = i:2 }\n    \
             %y : vec<i32x2> = tile.smem_load_vN ( %G, %tid ) { elem_type = s:\"i32\", rank = i:1, vec = i:2 }\n    \
             %z : vec<i32x2> = tile.smem_load_vN ( %B, %tid ) { elem_type = s:\"i32\", rank = i:1, vec = i:2 }\n    \
             %e : i32 = vector.extract ( %z ) { index = i:0 }\n    \
             memref.global_store_typed ( %Y, %tid, %e ) { align = i:4, elem_type = s:\"i32\" }\n    cf.return (  )\n  }\n}\n";
        let k = rir::parse::parse(text).unwrap_or_else(|e| panic!("{e}"));
        let offsets = |levers: &[&str]| {
            let (lds, size) = rir::with_levers(levers, || rir::analysis::lds_layout(&k));
            let id = |n: &str| (0..k.values.len() as u32).map(ValueId).find(|&v| k.name_of(v) == n).expect("a value");
            let mut v: Vec<u32> = ["%A", "%G", "%B"].iter().map(|n| lds[&id(n)]).collect();
            v.push(size);
            v
        };
        assert_eq!(offsets(&[]), [0, 1024, 9216, 10240]);
        assert_eq!(offsets(&["ldsgroup"]), [0, 2048, 1024, 10240]);
    }

    fn soffset_forms(k: &super::Kernel) -> Vec<super::Form> {
        use super::*;
        let n = k.values.len();
        let mut defs = vec![None; n];
        let mut home = vec![0u32; n];
        let mut loops = vec![(0u32, 0u32)];
        collect(&k.body, &mut defs, &mut home, &mut loops, 0);
        let cx = Cx { defs: &defs, kernel: k, limits: LIMITS };
        let mut out = Vec::new();
        rir::analysis::preorder(&k.body, |op| {
            if op.kind == OpKind::BufferLoadLds {
                out.push(cx.form(op.operands[3], DEPTH).expect("a linear soffset"));
            }
        });
        out
    }
}
