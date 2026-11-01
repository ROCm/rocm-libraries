// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Cross-lane ops computed once per run of active lanes on gfx950.
//!
//! A cross-lane op's result depends on which lanes are active, so the
//! optimizer never merges two of them (`OpKind::is_pure`). In a straight-line
//! run of a region's ops the active lanes do not change: they change only on
//! entering and leaving a nested region, or in inline assembly. Two such ops
//! of the same kind, operands and attributes there give the same values, as
//! LLVM's CSE finds for COMGR's code: the second goes and its readers take
//! the first's results. The pure ops of the run merge the same way in the
//! same walk, so that the exchanges of two merged values merge in turn: the
//! attention tunings move their softmax probabilities to the PV operand's
//! layout through stages of exchanges and selects that repeat this way.

use crate::rir::ast::*;
use crate::rir::fx::FxHashMap;

/// The ops whose result reads other lanes' registers (and nothing else).
fn exchanges(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::DsSwizzleXor
            | OpKind::DsBpermute
            | OpKind::DppXor
            | OpKind::QuadPerm
            | OpKind::MovDpp8
            | OpKind::Permlanex16
            | OpKind::Permlane16
            | OpKind::Permlane64
            | OpKind::Permlane32Swap
            | OpKind::Permlane16Swap
            | OpKind::ReadLane
            | OpKind::WaveAny
            | OpKind::WaveAll
    )
}

fn attr_hash(attrs: &[(Text, Attr)]) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for (name, attr) in attrs {
        name.hash(&mut h);
        match attr {
            Attr::Int(v) => v.hash(&mut h),
            Attr::Bool(v) => v.hash(&mut h),
            Attr::Str(v) => v.hash(&mut h),
            _ => return None,
        }
    }
    Some(h.finish())
}

struct Merge<'a> {
    types: &'a [ValueDef],
    pure: bool,
    /// Per value, the value its readers take: itself unless its op went.
    renames: Vec<ValueId>,
    merged: usize,
}

impl Merge<'_> {
    /// Where `region` files an op of attribute hash `attrs` (`attr_hash`):
    /// a hash of its key, its kind, operands, `attrs` and results' types.
    fn key_hash(&self, op: &Op, attrs: u64) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = crate::rir::fx::FxHasher::default();
        op.kind.hash(&mut h);
        op.operands[..].hash(&mut h);
        attrs.hash(&mut h);
        for r in op.results.iter() {
            self.types[r.0 as usize].ty.hash(&mut h);
        }
        h.finish()
    }

    /// Whether ops `a` and `b`, of attribute hashes `a_attrs` and `b_attrs`,
    /// have one key.
    fn same_key(&self, a: &Op, a_attrs: u64, b: &Op, b_attrs: u64) -> bool {
        let ty = |r: &ValueId| &self.types[r.0 as usize].ty;
        a.kind == b.kind
            && a.operands[..] == b.operands[..]
            && a_attrs == b_attrs
            && a.results.len() == b.results.len()
            && a.results.iter().zip(b.results.iter()).all(|(x, y)| ty(x) == ty(y))
    }

    fn region(&mut self, region: &mut Region) {
        // The run's ops a later one may merge into, by `key_hash`: their
        // places in `out` and attribute hashes.
        let mut seen: FxHashMap<u64, Vec<(usize, u64)>> = FxHashMap::default();
        let old = std::mem::take(&mut region.ops);
        let mut out = Vec::with_capacity(old.len());
        for mut op in old {
            for v in op.operands.iter_mut() {
                *v = self.renames[v.0 as usize];
            }
            if !op.regions.is_empty() || op.kind == OpKind::InlineAsmIdentity {
                for inner in op.regions.iter_mut() {
                    self.region(inner);
                }
                // The lanes change in there and back: past it, a run starts again.
                seen.clear();
                out.push(op);
                continue;
            }
            let exchange = exchanges(op.kind);
            if (exchange || (self.pure && op.kind.is_pure())) && !op.results.is_empty() {
                if let Some(attrs) = attr_hash(&op.attrs) {
                    let h = self.key_hash(&op, attrs);
                    let first = seen.get(&h).and_then(|list| list.iter().find(|&&(at, a)| self.same_key(&out[at], a, &op, attrs)));
                    if let Some(&(at, _)) = first {
                        for (&r, &f) in op.results.iter().zip(out[at].results.iter()) {
                            self.renames[r.0 as usize] = f;
                        }
                        self.merged += usize::from(exchange);
                        continue;
                    }
                    seen.entry(h).or_default().push((out.len(), attrs));
                }
            }
            out.push(op);
        }
        region.ops = out;
    }
}

/// Merges the repeated cross-lane ops of `kernel`'s runs, with the run's pure
/// ops when `pure` (not after the unrolling, whose copies keep some of those
/// apart on purpose); returns how many exchanges went. A kernel without
/// cross-lane ops is left to the optimizer's own merging of pure ops.
/// `ROCKE_DIRECT_DEBUG=nolanecse` skips it.
pub fn merge_exchanges(kernel: &mut Kernel, pure: bool) -> usize {
    if crate::rir::debug_flag("nolanecse") {
        return 0;
    }
    let mut any = false;
    crate::rir::analysis::preorder(&kernel.body, |op| any |= exchanges(op.kind));
    if !any || !pure && !repeats_an_exchange(&kernel.body) {
        return 0;
    }
    let mut body = std::mem::take(&mut kernel.body);
    let merged = {
        let renames = (0..kernel.values.len() as u32).map(ValueId).collect();
        let mut merge = Merge { types: &kernel.values, pure, renames, merged: 0 };
        merge.region(&mut body);
        merge.merged
    };
    kernel.body = body;
    merged
}

/// Whether a run of `region`'s ops (or of a region in them) has two
/// cross-lane ops of one kind: `Merge::region` without the pure ops merges
/// nothing else, and no less than that.
fn repeats_an_exchange(region: &Region) -> bool {
    let mut kinds: Vec<OpKind> = Vec::new();
    for op in &region.ops {
        if !op.regions.is_empty() || op.kind == OpKind::InlineAsmIdentity {
            if op.regions.iter().any(repeats_an_exchange) {
                return true;
            }
            kinds.clear();
        } else if exchanges(op.kind) {
            if kinds.contains(&op.kind) {
                return true;
            }
            kinds.push(op.kind);
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    /// Two row maxima of `x` over four lanes, the second `wrap`ped, stored.
    fn kernel(wrap: (&str, &str)) -> String {
        let mut s = String::from(
            "ckdsl.ir v1\nkernel @k {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %X : ptr<f32,global>\n    \
             %Y : ptr<f32,global>\n    %n : i32\n  }\n  region @entry {\n    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
             %c : i1 = arith.cmp ( %tid, %n ) { pred = s:\"lt\" }\n    \
             %x : f32 = memref.global_load ( %X, %tid ) { align = i:4 }\n    %xi : i32 = arith.bitcast ( %x ) { target = s:\"i32\" }\n",
        );
        for (k, (open, close)) in [("", ""), wrap].iter().enumerate() {
            s += &format!(
                "{open}    %s{k} : i32 = tile.ds_swizzle_xor ( %xi ) {{ xor_mask = i:1 }}\n    %f{k} : f32 = arith.bitcast ( %s{k} ) {{ target = s:\"f32\" }}\n    \
                 %m{k} : f32 = arith.fmax ( %x, %f{k} )\n    %o{k} : i32 = arith.add ( %tid, %tid )\n    \
                 memref.global_store_typed ( %Y, %o{k}, %m{k} ) {{ align = i:4, elem_type = s:\"f32\" }}\n{close}"
            );
        }
        s + "    cf.return (  )\n  }\n}\n"
    }

    fn swizzles(text: &str) -> usize {
        let listing = rir::compile_any_target(text, Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
        listing.lines().filter(|l| l.split_whitespace().next() == Some("ds_swizzle_b32")).count()
    }

    #[test]
    fn a_repeated_exchange_in_one_run_goes() {
        assert_eq!(swizzles(&kernel(("", ""))), 1);
    }

    #[test]
    fn an_exchange_under_other_lanes_stays() {
        let wrap = ("    scf.if ( %c )\n      region @then {\n", "      }\n");
        assert_eq!(swizzles(&kernel(wrap)), 2);
    }

    /// `merge_exchanges` as it was before its dense renames and its check for
    /// repeated exchanges, kept verbatim to compare against.
    mod reference {
        use super::super::{attr_hash, exchanges};
        use crate::rir::ast::*;
        use std::collections::HashMap;

        type Key = (OpKind, Vec<ValueId>, u64, Vec<Type>);

        struct Merge<'a> {
            types: &'a [ValueDef],
            pure: bool,
            renames: HashMap<ValueId, ValueId>,
            merged: usize,
        }

        impl Merge<'_> {
            fn region(&mut self, region: &mut Region) {
                let mut seen: HashMap<Key, Vec<ValueId>> = HashMap::new();
                let old = std::mem::take(&mut region.ops);
                let mut out = Vec::with_capacity(old.len());
                for mut op in old {
                    for v in op.operands.iter_mut() {
                        if let Some(&r) = self.renames.get(v) {
                            *v = r;
                        }
                    }
                    if !op.regions.is_empty() || op.kind == OpKind::InlineAsmIdentity {
                        for inner in op.regions.iter_mut() {
                            self.region(inner);
                        }
                        seen.clear();
                        out.push(op);
                        continue;
                    }
                    let exchange = exchanges(op.kind);
                    if (exchange || (self.pure && op.kind.is_pure())) && !op.results.is_empty() {
                        if let Some(attrs) = attr_hash(&op.attrs) {
                            let types = op.results.iter().map(|r| self.types[r.0 as usize].ty.clone()).collect();
                            let key = (op.kind, op.operands.iter().copied().collect(), attrs, types);
                            if let Some(first) = seen.get(&key) {
                                for (&r, &f) in op.results.iter().zip(first) {
                                    self.renames.insert(r, f);
                                }
                                self.merged += usize::from(exchange);
                                continue;
                            }
                            seen.insert(key, op.results.iter().copied().collect());
                        }
                    }
                    out.push(op);
                }
                region.ops = out;
            }
        }

        pub fn merge_exchanges(kernel: &mut Kernel, pure: bool) -> usize {
            if crate::rir::debug_flag("nolanecse") {
                return 0;
            }
            let mut any = false;
            crate::rir::analysis::preorder(&kernel.body, |op| any |= exchanges(op.kind));
            if !any {
                return 0;
            }
            let mut body = std::mem::take(&mut kernel.body);
            let merged = {
                let mut merge = Merge { types: &kernel.values, pure, renames: HashMap::new(), merged: 0 };
                merge.region(&mut body);
                merge.merged
            };
            kernel.body = body;
            merged
        }
    }

    /// Swizzles, quad permutes and row reads of loaded values and of each
    /// other's results, with masks from a few, bitcasts and maxima between,
    /// some in an `scf.if`, some after a barrier, and their results stored.
    fn random_exchanges(seed: u64) -> String {
        let mut s = seed.wrapping_mul(0x2545_f491_4f6c_dd1d) | 1;
        let mut next = |n: u64| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s % n
        };
        let mut t = String::from(
            "ckdsl.ir v1\nkernel @k {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %X : ptr<f32,global>\n    \
             %Y : ptr<f32,global>\n    %n : i32\n  }\n  region @entry {\n    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
             %c : i1 = arith.cmp ( %tid, %n ) { pred = s:\"lt\" }\n    \
             %x : f32 = memref.global_load ( %X, %tid ) { align = i:4 }\n    %v0 : i32 = arith.bitcast ( %x ) { target = s:\"i32\" }\n",
        );
        let mut visible = vec![0usize];
        let ops = 4 + next(24) as usize;
        for k in 1..=ops {
            let a = visible[next(visible.len() as u64) as usize];
            let line = match next(5) {
                0 | 1 => format!("%v{k} : i32 = tile.ds_swizzle_xor ( %v{a} ) {{ xor_mask = i:{} }}", 1 << next(3)),
                2 => format!("%v{k} : i32 = arith.add ( %v{a}, %v{} )", visible[next(visible.len() as u64) as usize]),
                3 => format!("%v{k} : i32 = arith.xor ( %v{a}, %v{a} )"),
                _ => format!("%v{k} : i32 = arith.add ( %v{a}, %tid )"),
            };
            if next(6) == 0 {
                t += &format!(
                    "    scf.if ( %c )\n      region @then {{\n    {line}\n    \
                     memref.global_store_typed ( %Y, %tid, %v{k} ) {{ align = i:4, elem_type = s:\"i32\" }}\n      }}\n"
                );
            } else {
                t += &format!("    {line}\n");
                visible.push(k);
            }
            if next(10) == 0 {
                t += "    tile.sync (  )\n";
            }
        }
        for k in visible {
            t += &format!("    %o{k} : i32 = arith.add ( %tid, %v{k} )\n");
            t += &format!("    memref.global_store_typed ( %Y, %o{k}, %v{k} ) {{ align = i:4, elem_type = s:\"i32\" }}\n");
        }
        t + "    cf.return (  )\n  }\n}\n"
    }

    #[test]
    fn exchange_merging_changes_what_it_changed() {
        let (mut merged, mut kept) = (0, 0);
        for seed in 0..300u64 {
            let text = random_exchanges(seed);
            let parsed = rir::parse::parse(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
            let mut optimized = parsed.clone();
            let (mut lds, _) = rir::analysis::lds_layout(&optimized);
            rir::opt::optimize(&mut optimized, &mut lds).unwrap_or_else(|e| panic!("{e}"));
            for mut k in [parsed, optimized] {
                k.wave = 64;
                for pure in [true, false] {
                    let mut old = k.clone();
                    let n = reference::merge_exchanges(&mut old, pure);
                    let mut new = k.clone();
                    assert_eq!(super::merge_exchanges(&mut new, pure), n, "seed {seed}");
                    assert_eq!(rir::print::dump(&new), rir::print::dump(&old), "seed {seed} pure {pure}");
                    if n > 0 {
                        merged += 1;
                    } else {
                        kept += 1;
                    }
                }
            }
        }
        assert!(merged > 100 && kept > 100, "{merged} merged, {kept} kept");
    }
}
