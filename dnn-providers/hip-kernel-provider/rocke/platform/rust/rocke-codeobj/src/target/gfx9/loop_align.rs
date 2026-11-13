// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! rust-compiler-v2 lever `loopalign`: every loop header (a block that a later block branches back to) starts at a
//! 64-byte boundary of the code, as the instruction cache lines and fetch blocks of gfx950 are laid out; `s_nop 0`
//! words fill the block before it (executed once on the way in, never by the loop). The last pass before encoding:
//! hazards already counted, and an `s_nop` adds wait states only.
//!
//! Branches (`s_cbranch_*` / `s_branch`) keep a 16-bit word offset; a target past it is relaxed by the encoder
//! (`asm::assemble_relaxed`), after this pass: the GEMM loops here are below 2,000 instructions, so the alignment
//! padding is not recomputed for the few kernels with a loop body past 128 KiB.

use crate::error::CompileError;
use crate::rir::mir::*;

/// Code alignment of a loop header, in dwords (64 bytes).
const ALIGN_WORDS: usize = 16;

/// Pads before every loop header of `f`; returns the `s_nop` words added.
pub fn run(f: &mut MFunc) -> Result<usize, CompileError> {
    let n = f.blocks.len();
    let mut header = vec![false; n];
    for (b, block) in f.blocks.iter().enumerate() {
        for inst in block.insts.iter().filter(|i| i.is_branch()) {
            let t = inst.imm as usize;
            if t <= b && t < n {
                header[t] = true;
            }
        }
    }
    let mut added = 0;
    let mut at = 0usize;
    for b in 0..n {
        if header[b] && b > 0 && at % ALIGN_WORDS != 0 {
            let pad = ALIGN_WORDS - at % ALIGN_WORDS;
            // Into the block before, ahead of its branch when it ends in one (an `s_nop` is one word).
            let prev = &mut f.blocks[b - 1].insts;
            let pos = prev.iter().rposition(|i| !i.is_branch()).map_or(0, |i| i + 1);
            for _ in 0..pad {
                prev.insert(pos, MInst::new(Opc::SNop, [], []));
            }
            added += pad;
            at += pad;
        }
        for inst in &f.blocks[b].insts {
            at += super::asm::encode_words(inst)?.len();
        }
    }
    Ok(added)
}

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    /// A loop of `n` trips adding loads of X into an accumulator, after `pre` extra stores (to move the header).
    fn kernel(pre: usize) -> String {
        let stores: String = (0..pre)
            .map(|k| format!("    %o{k} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n    memref.global_store_typed ( %Y, %o{k}, %tid ) {{ align = i:4, elem_type = s:\"i32\" }}\n", 64 * (k + 1)))
            .collect();
        format!(
            "ckdsl.ir v1\nkernel @la {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %X : ptr<i32,global>\n    \
             %Y : ptr<i32,global>\n    %n : i32\n  }}\n  region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n{stores}    \
             %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    \
             %c1 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:1 }}\n    \
             %s : i32 = scf.for ( %c0, %n, %c1, %c0 ) {{ iter_args = l:[ {{ name = s:\"%a\", type = s:\"i32\" }} ], iv = s:\"%k\", \
             iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:false }}\n      region @body {{\n        \
             %p : i32 = arith.add ( %k, %tid )\n        \
             %x : i32 = memref.global_load_typed ( %X, %p ) {{ align = i:4, elem_type = s:\"i32\" }}\n        \
             %b : i32 = arith.add ( %a, %x )\n        scf.yield ( %b ) {{ num = i:1 }}\n      }}\n    \
             memref.global_store_typed ( %Y, %tid, %s ) {{ align = i:4, elem_type = s:\"i32\" }}\n    cf.return (  )\n  }}\n}}\n"
        )
    }

    /// Byte offsets of the targets of the backward branches of a listing (a branch's imm is its word delta).
    fn back_targets(listing: &str) -> Vec<i64> {
        listing
            .lines()
            .filter(|l| l.trim_start().starts_with("s_cbranch") || l.trim_start().starts_with("s_branch"))
            .filter_map(|l| {
                let off = i64::from_str_radix(l.rsplit("// ").next()?.split(':').next()?.trim(), 16).ok()?;
                let imm: i64 = l.split("imm:").nth(1)?.split_whitespace().next()?.parse().ok()?;
                let delta = imm as u16 as i16 as i64;
                (delta < 0).then_some(off + 4 + 4 * delta)
            })
            .collect()
    }

    #[test]
    fn loop_headers_start_on_64_bytes() {
        for pre in 0..4 {
            let plain = rir::compile_any_target(&kernel(pre), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
            let aligned = rir::with_levers(&["loopalign"], || rir::compile_any_target(&kernel(pre), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")));
            let back = back_targets(&aligned.listing());
            assert!(!back.is_empty(), "{}", aligned.listing());
            assert!(back.iter().all(|t| t % 64 == 0), "pre {pre}: {back:?}
{}", aligned.listing());
            // Only s_nop words were added.
            let count = |l: &str| l.lines().filter(|x| x.starts_with("  ") && !x.trim_start().starts_with("s_nop")).count();
            assert_eq!(count(&plain.listing()), count(&aligned.listing()), "pre {pre}");
            if pre == 1 {
                assert!(back_targets(&plain.listing()).iter().any(|t| t % 64 != 0), "{}", plain.listing());
            }
        }
    }
}
