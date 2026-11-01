// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Long accumulator chains issued round-robin, on gfx1151. A run of
//! consecutive f32 `arith.fma` ops in a loop body can hold independent chains
//! one after another (an FMA whose addend is its chain's previous FMA
//! continues the chain): a tiled depthwise convolution accumulates each
//! output over all its taps before the next output. The window scheduler
//! (`post::schedule_window`, at most 64 places) then overlaps only the two or
//! three chains its window holds, and each FMA waits for the one before it
//! (`s_delay_alu`); LLVM's scheduler, which sees the whole block, interleaves
//! every chain. This pass reorders such a run so the chains take turns, one
//! FMA of each in run order, when its longest chain has at least
//! `MIN_CHAIN` FMAs. Only the order inside the run changes, and only where
//! every FMA reads values defined before the run or its own chain's previous
//! FMA. `ROCKE_DIRECT_DEBUG=nointerleave` turns it off.

use super::ast::{Kernel, Op, OpKind, Region, Scalar, Type, ValueId};
use super::fx::FxHashMap;

/// The shortest longest chain worth reordering. Chains of 15 and 27 FMAs
/// stalled on every FMA and ran 10 to 14% faster interleaved; the shorter
/// chains of other depthwise kernels already overlap enough in the window,
/// and reordering them cost some kernels a wave.
const MIN_CHAIN: usize = 15;

/// Interleaves every qualifying run of the kernel; returns how many.
pub(super) fn interleave(kernel: &mut Kernel) -> usize {
    if super::debug_flag("nointerleave") {
        return 0;
    }
    let f32s: Vec<bool> = (0..kernel.values.len()).map(|v| *kernel.ty(ValueId(v as u32)) == Type::Scalar(Scalar::F32)).collect();
    let mut body = std::mem::take(&mut kernel.body);
    let n = walk(&mut body, false, &f32s);
    kernel.body = body;
    n
}

fn walk(region: &mut Region, in_loop: bool, f32s: &[bool]) -> usize {
    let mut n = 0;
    for op in region.ops.iter_mut() {
        let inner = in_loop || op.kind == OpKind::For;
        for r in op.regions.iter_mut() {
            n += walk(r, inner, f32s);
        }
    }
    if !in_loop {
        return n;
    }
    let is_fma = |op: &Op| op.kind == OpKind::Fma && op.results.len() == 1 && f32s[op.results[0].0 as usize];
    let mut i = 0;
    while i < region.ops.len() {
        if !is_fma(&region.ops[i]) {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < region.ops.len() && is_fma(&region.ops[j]) {
            j += 1;
        }
        if let Some(order) = round_robin(&region.ops[i..j]) {
            let mut slots: Vec<Option<Op>> = region.ops.drain(i..j).map(Some).collect();
            let reordered: Vec<Op> = order.into_iter().map(|k| slots[k].take().expect("each FMA once")).collect();
            region.ops.splice(i..i, reordered);
            n += 1;
        }
        i = j;
    }
    n
}

/// The run's FMAs in round-robin order of their chains, if the run holds at
/// least two chains, its longest has `MIN_CHAIN` FMAs, and no FMA reads
/// another FMA of the run except as its chain's addend.
fn round_robin(run: &[Op]) -> Option<Vec<usize>> {
    let mut chain_of: FxHashMap<ValueId, usize> = FxHashMap::default();
    let mut chains: Vec<Vec<usize>> = Vec::new();
    for (k, op) in run.iter().enumerate() {
        if op.operands.len() != 3 {
            return None;
        }
        let inside: Vec<usize> = (0..3).filter(|&s| chain_of.contains_key(&op.operands[s])).collect();
        let c = match inside[..] {
            [] => {
                chains.push(Vec::new());
                chains.len() - 1
            }
            [2] => {
                let c = chain_of[&op.operands[2]];
                let last = *chains[c].last().expect("a chain has an FMA");
                if run[last].results[0] != op.operands[2] {
                    return None;
                }
                c
            }
            _ => return None,
        };
        chains[c].push(k);
        chain_of.insert(op.results[0], c);
    }
    let longest = chains.iter().map(Vec::len).max().unwrap_or(0);
    if chains.len() < 2 || longest < MIN_CHAIN {
        return None;
    }
    let order: Vec<usize> = (0..longest).flat_map(|t| chains.iter().filter_map(move |c| c.get(t).copied())).collect();
    (order.iter().enumerate().any(|(p, &k)| p != k)).then_some(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `chains` outputs, each accumulated over `taps` products `w[t] * x[c + t]`,
    /// written one output after another (`round_robin`: one tap of each in
    /// turn), in an `scf.for` body (`looped`) or straight-line code. `cross`
    /// makes the second chain's first FMA multiply by the first chain's result.
    fn kernel(chains: usize, taps: usize, looped: bool, round_robin: bool, cross: bool) -> String {
        let mut body = String::new();
        let ind = if looped { "        " } else { "    " };
        for t in 0..chains + taps {
            body += &format!("{ind}%k{t} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{t} }}\n");
            body += &format!("{ind}%j{t} : i32 = arith.add ( %base, %k{t} )\n");
            body += &format!(
                "{ind}%x{t} : f32 = memref.global_load_typed ( %X, %j{t} ) {{ align = i:4, elem_type = s:\"f32\" }}\n"
            );
            body += &format!(
                "{ind}%w{t} : f32 = memref.global_load_typed ( %W, %k{t} ) {{ align = i:4, elem_type = s:\"f32\" }}\n"
            );
        }
        let fma = |c: usize, t: usize| {
            let acc = if t == 0 { format!("%a{c}") } else { format!("%f{c}_{}", t - 1) };
            let x = if cross && c == 1 && t == 0 { format!("%f0_{}", taps - 1) } else { format!("%x{}", c + t) };
            format!("{ind}%f{c}_{t} : f32 = arith.fma ( %w{t}, {x}, {acc} )\n")
        };
        let order: Vec<(usize, usize)> = if round_robin {
            (0..taps).flat_map(|t| (0..chains).map(move |c| (c, t))).collect()
        } else {
            (0..chains).flat_map(|c| (0..taps).map(move |t| (c, t))).collect()
        };
        for (c, t) in order {
            body += &fma(c, t);
        }
        let finals: Vec<String> = (0..chains).map(|c| format!("%f{c}_{}", taps - 1)).collect();
        let mut text = String::from(
            "ckdsl.ir v1\nkernel @probe {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %X : ptr<f32,global>\n    \
             %W : ptr<f32,global>\n    %Y : ptr<f32,global>\n    %n : i32\n  }\n  region @entry {\n    \
             %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
             %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    \
             %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n    \
             %z : f32 = arith.constant (  ) { ity = s:\"f32\", value = f:0.0 }\n",
        );
        let results: Vec<String> = (0..chains).map(|c| format!("%r{c}")).collect();
        if looped {
            let heads: Vec<String> = results.iter().map(|r| format!("{r} : f32")).collect();
            let args: Vec<String> = (0..chains).map(|c| format!("{{ name = s:\"%a{c}\", type = s:\"f32\" }}")).collect();
            text += &format!(
                "    {} = scf.for ( %c0, %n, %c1, {} ) {{ iter_args = l:[ {} ], iv = s:\"%i\", iv_type = s:\"i32\", \
                 num_iter_args = i:{chains}, unroll = b:false }}\n      region @body {{\n        \
                 %base : i32 = arith.add ( %tid, %i )\n{body}        scf.yield ( {} ) {{ num = i:{chains} }}\n      }}\n",
                heads.join(", "),
                vec!["%z"; chains].join(", "),
                args.join(", "),
                finals.join(", ")
            );
        } else {
            for c in 0..chains {
                text += &format!("    %a{c} : f32 = arith.add ( %z, %z )\n");
            }
            text += &format!("    %base : i32 = arith.add ( %tid, %c0 )\n{body}");
            for c in 0..chains {
                text += &format!("    %r{c} : f32 = arith.add ( %f{c}_{}, %z )\n", taps - 1);
            }
        }
        for (c, r) in results.iter().enumerate() {
            text += &format!(
                "    %q{c} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n    \
                 %o{c} : i32 = arith.add ( %tid, %q{c} )\n    \
                 memref.global_store_typed ( %Y, %o{c}, {r} ) {{ align = i:4, elem_type = s:\"f32\" }}\n",
                64 * c
            );
        }
        text + "    cf.return (  )\n  }\n}\n"
    }

    fn parsed(text: &str) -> Kernel {
        crate::rir::parse::parse(text).unwrap_or_else(|e| panic!("{e}"))
    }

    /// The chain of each FMA in program order, from the names `%f<chain>_<tap>`.
    fn chain_order(k: &Kernel) -> Vec<usize> {
        fn visit(region: &Region, k: &Kernel, out: &mut Vec<usize>) {
            for op in &region.ops {
                if op.kind == OpKind::Fma {
                    let name = k.name_of(op.results[0]);
                    let c = name.trim_start_matches(['%', 'f']).split('_').next().and_then(|c| c.parse().ok());
                    out.push(c.expect("an FMA of the test kernel"));
                }
                for r in &op.regions {
                    visit(r, k, out);
                }
            }
        }
        let mut out = Vec::new();
        visit(&k.body, k, &mut out);
        out
    }

    #[test]
    fn long_chains_in_a_loop_take_turns() {
        let mut k = parsed(&kernel(4, 15, true, false, false));
        assert_eq!(interleave(&mut k), 1);
        let want: Vec<usize> = (0..15).flat_map(|_| 0..4).collect();
        assert_eq!(chain_order(&k), want);
        assert_eq!(interleave(&mut k), 0, "a run already round-robin stays");
    }

    #[test]
    fn short_chains_straight_line_code_and_shared_results_stay() {
        for (chains, taps, looped, cross) in [(4, 14, true, false), (4, 15, false, false), (4, 15, true, true)] {
            let text = kernel(chains, taps, looped, false, cross);
            let mut k = parsed(&text);
            let before = chain_order(&k);
            assert_eq!(interleave(&mut k), 0, "{chains} x {taps} looped {looped} cross {cross}");
            assert_eq!(chain_order(&k), before);
        }
    }

    #[test]
    fn the_compile_for_gfx1151_interleaves_and_still_builds() {
        // The rewritten run selects, allocates and encodes; the order is the pass's.
        let text = kernel(6, 16, true, false, false);
        let mut k = parsed(&text);
        assert_eq!(interleave(&mut k), 1);
        for target in [crate::target::Target::Gfx1151, crate::target::Target::Gfx950] {
            crate::rir::compile_any_target(&text, target).unwrap_or_else(|e| panic!("{target:?}: {e}"));
        }
    }
}
