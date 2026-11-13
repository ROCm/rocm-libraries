// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! An LDS DMA's lane offset as `shared + per-load scalar` (gfx950 attention K/V tiles).
//!
//! rocKE writes the K and V tile loads of an unrolled loop body one by one: load `j` of a buffer reads lane
//! offset `2 * (((tile * 64 + wave * 8 + j) * stride + base) + lane_part)`, where `stride` is a kernel argument. Selected
//! as written every load pays its own `v_add`, `v_mul_lo_u32` (quarter rate), `v_add3` and `v_lshlrev`: sixteen
//! times per loop iteration.
//!
//! The offset is a polynomial over its leaves (`tile`, `wave`, `stride`, ...) in which the unrolled index `j` only
//! changes the coefficient of the uniform `stride` by `2 * j`. Here every such offset inside a loop is rebuilt as
//! `(invariant terms + variant terms) + (uniform terms)`: the sum without the uniform single-leaf terms
//! (`2 * j * stride`) is the same op for every load of the body (the sweep's CSE keeps one, and its invariant half
//! leaves the loop), so a load pays one `v_add_u32` of a scalar it computes on the SALU.
//!
//! Exact: add, sub and mul of i32 are ring operations modulo 2^32 and the rebuilt ops are `wrap = "modular"`.
//! Only the voffset operand of a buffer DMA load inside a loop changes (`ROCKE_DIRECT_DEBUG=novoffsplit` skips it).

use super::analysis::classify;
use super::ast::*;
use super::fx::FxHashMap;
use super::ring_form::{collect, Cx, Def, Emit, Limits, Mono};

const TERMS: usize = 12;
const DEPTH: u32 = 12;
/// Loads of one loop that share a base and differ in their uniform part before it pays: the K and V tiles unroll 4 to 16; the
/// split-KV segment kernels' paged loads differ in their soffset only (and got 6-12 more instructions).
const MIN_SHARED: usize = 4;

/// Degree-2 forms of up to `TERMS` terms; a larger sub-expression is one leaf.
const LIMITS: Limits = Limits { products: true, terms: TERMS, leaf_on_overflow: true };

/// Per (loop, shared terms): the distinct (uniform terms, constant) of the loads that have them.
type Counts = FxHashMap<(u32, Vec<(Mono, u32)>), Vec<(Vec<(Mono, u32)>, u32)>>;

struct Rw<'a> {
    defs: Vec<Option<Def>>,
    home: Vec<u32>,
    loops: Vec<(u32, u32)>,
    uniform: Vec<bool>,
    next_loop: u32,
    kernel: &'a mut Kernel,
    rewritten: usize,
}

impl Rw<'_> {
}

impl Emit for Rw<'_> {
    const PREFIX: &'static str = "%vsp";

    fn tables(&mut self) -> (&mut Kernel, &mut Vec<Option<Def>>, &mut Vec<u32>) {
        (&mut *self.kernel, &mut self.defs, &mut self.home)
    }
}

impl Rw<'_> {
    fn region(&mut self, region: &mut Region, at: u32, counts: &Counts) {
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
                self.region(r, inner, counts);
            }
            if op.kind == OpKind::BufferLoadLds && at != 0 && op.operands.len() > 3 {
                if let Some(v) = self.rebuild(op.operands[2], at, counts, &mut out) {
                    op.operands[2] = v;
                    self.rewritten += 1;
                }
            }
            out.push(op);
        }
        region.ops = out;
    }

    /// The split of lane offset `s` in loop `l`: (the terms that are not uniform single leaves next to a product, those
    /// that are, the constant), or None when there is no such uniform term or `s` is not a polynomial.
    fn split(&self, s: ValueId, l: u32) -> Option<(Vec<(Mono, u32)>, Vec<(Mono, u32)>, u32)> {
        let poly = Cx { defs: &self.defs, kernel: &*self.kernel, limits: LIMITS }.form(s, DEPTH)?;
        let last = self.loops[l as usize].1;
        let inside = |v: u32| self.home.get(v as usize).is_some_and(|&h| (l..=last).contains(&h));
        // Leaves that multiply another leaf: the strides.
        let factors: Vec<u32> = poly.terms.iter().filter(|t| t.0.len() == 2).flat_map(|t| t.0.clone()).collect();
        let is_part = |t: &(Mono, u32)| t.0.len() == 1 && factors.contains(&t.0[0]) && self.uniform.get(t.0[0] as usize).copied().unwrap_or(false) && !inside(t.0[0]);
        let (part, rest): (Vec<_>, Vec<_>) = poly.terms.iter().cloned().partition(|t| is_part(t));
        (!part.is_empty() && poly.terms.iter().any(|t| t.0.len() == 2)).then_some((rest, part, poly.konst))
    }

    /// Counts, per loop, the loads that share the same non-uniform terms.
    fn count(&self, region: &Region, at: u32, next: &mut u32, counts: &mut Counts) {
        for op in &region.ops {
            let inner = if op.kind == OpKind::For {
                *next += 1;
                *next
            } else {
                at
            };
            for r in &op.regions {
                self.count(r, inner, next, counts);
            }
            if op.kind == OpKind::BufferLoadLds && at != 0 && op.operands.len() > 3 {
                if let Some((rest, part, konst)) = self.split(op.operands[2], at) {
                    let sigs = counts.entry((at, rest)).or_default();
                    if !sigs.contains(&(part.clone(), konst)) {
                        sigs.push((part, konst));
                    }
                }
            }
        }
    }

    /// The lane offset `s` as `(invariant + variant) + uniform` for loop `l`.
    fn rebuild(&mut self, s: ValueId, l: u32, counts: &Counts, out: &mut Vec<Op>) -> Option<ValueId> {
        let (rest, part, konst) = self.split(s, l)?;
        // Already one vector add of a uniform per load (the split-KV kernels step each load from the previous one's offset).
        if let Some(Some(d)) = self.defs.get(s.0 as usize) {
            if d.kind == OpKind::Add && d.operands.iter().any(|v| self.uniform.get(v.0 as usize).copied().unwrap_or(false)) {
                return None;
            }
        }
        // A base that few other loads of the loop share is no cheaper than the offset as written.
        if counts.get(&(l, rest.clone())).map_or(0, |v| v.len()) < MIN_SHARED {
            return None;
        }
        let last = self.loops[l as usize].1;
        let inside = |v: u32, home: &[u32]| home.get(v as usize).is_some_and(|&h| (l..=last).contains(&h));
        let (var, inv): (Vec<_>, Vec<_>) = rest.iter().partition(|t| t.0.iter().any(|&v| inside(v, &self.home)));
        let a = self.sum(inv, out);
        let b = self.sum(var, out);
        let base = match (a, b) {
            (Some(a), Some(b)) => Some(self.bin(OpKind::Add, a, b, out)),
            (x, None) | (None, x) => x,
        };
        let mut uni = self.sum(&part, out)?;
        if konst != 0 {
            let k = self.konst(konst, out);
            uni = self.bin(OpKind::Add, uni, k, out);
        }
        Some(match base {
            Some(b) => self.bin(OpKind::Add, b, uni, out),
            None => uni,
        })
    }
}

/// Rebuilds the lane offsets of the buffer DMA loads in loops of `kernel`; returns how many.
pub fn run(kernel: &mut Kernel) -> usize {
    if super::debug_flag("novoffsplit") || kernel.wave != 64 {
        return 0;
    }
    let n = kernel.values.len();
    let mut defs = vec![None; n];
    let mut home = vec![0u32; n];
    let mut loops = vec![(0u32, 0u32)];
    collect(&kernel.body, &mut defs, &mut home, &mut loops, 0);
    let uniform = classify(kernel);
    let mut body = std::mem::take(&mut kernel.body);
    let mut rw = Rw { defs, home, loops, uniform, next_loop: 0, kernel, rewritten: 0 };
    let mut counts = FxHashMap::default();
    rw.count(&body, 0, &mut 0, &mut counts);
    rw.region(&mut body, 0, &counts);
    let rewritten = rw.rewritten;
    kernel.body = body;
    rewritten
}

#[cfg(test)]
mod tests {
    use super::super::ring_form::Form;
    use super::*;
    use crate::rir;
    use crate::target::Target;

    /// `loads` DMA loads per trip, load `j` at lane offset `((kt * 8 + j) * %S + tid) * 4` (`%S`: a kernel argument).
    fn kernel(loads: usize) -> String {
        let mut body = String::new();
        for j in 0..loads {
            body += &format!(
                "        %c{j} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{j} }}\n        \
                 %row{j} : i32 = arith.add ( %kt8, %c{j} )\n        \
                 %rs{j} : i32 = arith.mul ( %row{j}, %S )\n        \
                 %at{j} : i32 = arith.add ( %rs{j}, %tid )\n        \
                 %vo{j} : i32 = arith.mul ( %at{j}, %c4 )\n        \
                 %wo{j} : i32 = arith.mul ( %c{j}, %c256 )\n        \
                 %zo{j} : i64 = arith.zext ( %wo{j} )\n        \
                 %lp{j} : i64 = tile.smem_ptr_add ( %base, %zo{j} )\n        \
                 tile.async_buffer_load_lds_addr ( %rsrc, %lp{j}, %vo{j}, %c0 ) {{ aux = i:0, dwords = i:1 }}\n"
            );
        }
        format!(
            "ckdsl.ir v1\nkernel @k {{\n  attrs {{ max_workgroup_size = i:256 }}\n  params {{\n    %X : ptr<i32,global>\n    \
             %Y : ptr<i32,global>\n    %S : i32\n    %n : i32\n  }}\n  region @entry {{\n    \
             %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    \
             %c1 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:1 }}\n    \
             %c4 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:4 }}\n    \
             %c8 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:8 }}\n    \
             %c256 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:256 }}\n    \
             %cbig : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:2147483632 }}\n    \
             %t : smem<i32, [4096]> = tile.smem_alloc (  )\n    \
             %base : i64 = tile.smem_addr_of ( %t )\n    \
             %rsrc : vec<i32x4> = tile.buffer_rsrc ( %X, %cbig )\n    \
             scf.for ( %c0, %n, %c1 ) {{ iv = s:\"%kt\", iv_type = s:\"i32\", num_iter_args = i:0, unroll = b:false }}\n      \
             region @body {{\n        %kt8 : i32 = arith.mul ( %kt, %c8 )\n{body}        scf.yield (  ) {{ num = i:0 }}\n      }}\n    \
             tile.sync (  )\n    \
             %v : i32 = tile.smem_load_vN ( %t, %tid ) {{ elem_type = s:\"i32\", rank = i:1, vec = i:1 }}\n    \
             memref.global_store_typed ( %Y, %tid, %v ) {{ align = i:4, elem_type = s:\"i32\" }}\n    \
             cf.return (  )\n  }}\n}}\n"
        )
    }

    fn polys(k: &Kernel) -> Vec<Form> {
        let n = k.values.len();
        let (mut defs, mut home, mut loops) = (vec![None; n], vec![0u32; n], vec![(0u32, 0u32)]);
        collect(&k.body, &mut defs, &mut home, &mut loops, 0);
        let mut out = Vec::new();
        crate::rir::analysis::preorder(&k.body, |op| {
            if op.kind == OpKind::BufferLoadLds {
                out.push(Cx { defs: &defs, kernel: k, limits: LIMITS }.form(op.operands[2], DEPTH).expect("a polynomial"));
            }
        });
        out
    }

    #[test]
    fn the_rebuilt_offset_is_the_same_polynomial() {
        let mut k = rir::parse::parse(&kernel(5)).unwrap_or_else(|e| panic!("{e}"));
        k.wave = 64;
        let before = polys(&k);
        // Load 0's row has no constant, so no uniform term: loads 1 to 4.
        assert_eq!(run(&mut k), 4);
        assert_eq!(before, polys(&k));
        assert!(before[1].terms.iter().any(|t| t.0.len() == 1 && t.1 == 4));
    }

    #[test]
    fn each_load_pays_one_vector_add_and_the_multiply_is_shared() {
        let l = rir::compile_any_target(&kernel(4), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
        let muls = l.lines().filter(|x| x.trim_start().starts_with("v_mul_lo_u32")).count();
        assert!(muls <= 1, "{muls} v_mul_lo_u32\n{l}");
    }
}
