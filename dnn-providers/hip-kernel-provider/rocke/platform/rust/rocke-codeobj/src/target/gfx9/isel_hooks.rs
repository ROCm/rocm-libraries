// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! What instruction selection picks differently on gfx9 (wave64). The gfx11
//! choices stay in `rir::isel`; it calls these where the targets differ.

use super::gen_isa::Gfx9Op;
use crate::error::CompileError;
use crate::rir::mir::{MInst, Opc};
use crate::target::Target;

/// A memory fence at `scope` on gfx950, as its memory model writes it
/// outside threadgroup-split mode: `buffer_wbl2` writes L2 back at agent
/// (sc1) and system (sc0 sc1) scope, then `s_waitcnt vmcnt(0) lgkmcnt(0)`
/// completes every earlier access (at workgroup scope too, where the model
/// asks only for lgkmcnt: the conservative wait); an acquire then
/// invalidates the caches below the scope (`buffer_inv`; a workgroup shares
/// one CU's).
///
/// Lever `wgfence` (the rust track): a workgroup-scope fence waits as the
/// model asks, as LLVM's `SIGfx90ACacheControl` and COMGR do outside
/// threadgroup split mode: a release waits for LDS (`lgkmcnt(0)`) and for the
/// LDS DMA loads and stores in flight (the bytes other waves read next), not
/// for plain loads; an acquire waits for nothing. All waves of a workgroup run
/// on one CU and share its vL1D; vector loads in flight (weights issued ahead
/// of a barrier) are no longer drained by it.
pub fn fence(scope: &str, acquire: bool) -> Result<Vec<MInst>, CompileError> {
    if scope == "workgroup" && crate::rir::lever("wgfence") {
        if acquire {
            return Ok(Vec::new());
        }
        // LDS, and the LDS DMA loads (other waves read their bytes) and stores in flight
        // (`waitcnt` places the wait: `WAIT_RELEASE`).
        return Ok(vec![MInst::new(Opc::Gfx9(Gfx9Op::WaitRelease), [], [])]);
    }
    let cpol = match scope {
        "workgroup" => None,
        "agent" => Some(2),
        "system" => Some(3),
        other => return Err(CompileError::new("ROCKE_DIRECT_UNSUPPORTED", format!("memory fence scope {other}"))),
    };
    let mut out = Vec::new();
    if let Some(c) = cpol {
        out.push(MInst::new(Opc::Gfx9(Gfx9Op::BufferWbl2), [], []).with_imm(c));
    }
    out.push(MInst::new(Opc::SWaitcntExplicit, [], []).with_imm(super::waitcnt::waitcnt_imm(0, 0)));
    if let (true, Some(c)) = (acquire, cpol) {
        out.push(MInst::new(Opc::Gfx9(Gfx9Op::BufferInv), [], []).with_imm(c));
    }
    Ok(out)
}

/// An acquire-only fence (lever `acqfence`): the waits, then `buffer_inv` at
/// agent (sc1) or system (sc0 sc1) scope, as LLVM's `SIGfx940CacheControl::
/// insertAcquire`; no `buffer_wbl2` (that is a release's). A workgroup
/// acquire is nothing (one CU's vL1D).
pub fn acquire_fence(scope: &str) -> Result<Vec<MInst>, CompileError> {
    let cpol = match scope {
        "workgroup" => return Ok(Vec::new()),
        "agent" => 2,
        "system" => 3,
        other => return Err(CompileError::new("ROCKE_DIRECT_UNSUPPORTED", format!("memory fence scope {other}"))),
    };
    Ok(vec![
        MInst::new(Opc::SWaitcntExplicit, [], []).with_imm(super::waitcnt::waitcnt_imm(0, 0)),
        MInst::new(Opc::Gfx9(Gfx9Op::BufferInv), [], []).with_imm(cpol),
    ])
}

/// Whether a d16 load on `target` keeps the other half of its destination,
/// which selection relies on to load two 16-bit values into the halves of
/// one register. gfx950 code objects run in either SRAM ECC mode (`sramecc
/// any`), and with ECC on a d16 load does not keep it (LLVM's
/// `d16PreservesUnusedBits`): the halves are loaded whole and packed.
pub fn d16_keeps_other_half(target: Target) -> bool {
    target != Target::Gfx950
}

/// The wave64 form of the wave32 lane-mask opcode `op`: lane masks, EXEC and
/// VCC are SGPR pairs.
pub fn wide(op: Opc) -> Opc {
    match op {
        Opc::SAndB32 => Opc::SAndB64,
        Opc::SOrB32 => Opc::SOrB64,
        Opc::SXorB32 => Opc::SXorB64,
        Opc::SAndNot1B32 => Opc::SAndNot1B64,
        Opc::SNotB32 => Opc::SNotB64,
        Opc::SMovB32 => Opc::SMovB64,
        Opc::SCselectB32 => Opc::SCselectB64,
        Opc::SCmpLgU32 => Opc::SCmpLgU64,
        Opc::SCmpEqU32 => Opc::SCmpEqU64,
        Opc::SAndSaveexecB32 => Opc::SAndSaveexecB64,
        other => unreachable!("{other:?} is not a lane-mask opcode"),
    }
}

/// `tile.dpp_xor` on gfx9, which has no DPP `row_xmask`: lane `L` reads lane
/// `L ^ mask` (mask 1..15, within a row of 16).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XorShuffle {
    /// A DPP `quad_perm` control (masks below 4 stay within a quad).
    QuadPerm(u32),
    /// A `ds_swizzle_b32` offset in bitmask mode, which permutes within 32 lanes.
    Swizzle(u32),
}

pub fn xor_shuffle(mask: u32) -> XorShuffle {
    xor_shuffle_with(mask, crate::rir::lever("dppmirror"))
}

/// `xor_shuffle`; with `mirror` (lever `dppmirror`) the masks a 16-lane row
/// permutes without row_xmask take their register-only DPP control too, as
/// the LLVM lowering (`rocke.core.ir.DPP_XOR_CTRL_NO_ROW_XMASK`): 7
/// row_half_mirror, 8 row_ror:8, 15 row_mirror. (`QuadPerm` holds any DPP16
/// control.)
pub fn xor_shuffle_with(mask: u32, mirror: bool) -> XorShuffle {
    match mask {
        1 => XorShuffle::QuadPerm(0xb1),
        2 => XorShuffle::QuadPerm(0x4e),
        3 => XorShuffle::QuadPerm(0x1b),
        7 if mirror => XorShuffle::QuadPerm(0x141),
        8 if mirror => XorShuffle::QuadPerm(0x128),
        15 if mirror => XorShuffle::QuadPerm(0x140),
        m => XorShuffle::Swizzle(m << 10 | 0x1f),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CompileError;
    use crate::rir::mir::{Class, MFunc, MInst, Opnd, Phys};
    use crate::rir::{analysis, isel, opt, parse};
    use crate::target::Target;

    fn select(text: &str, target: Target) -> Result<MFunc, CompileError> {
        let mut kernel = parse::parse(text)?;
        kernel.wave = target.info().wave;
        let (mut lds, lds_size) = analysis::lds_layout(&kernel);
        let facts = opt::optimize(&mut kernel, &mut lds)?;
        let uniform = analysis::classify(&kernel);
        Ok(isel::select_for(target, &kernel, &facts, uniform, &lds, lds_size)?.func)
    }

    fn insts(text: &str, target: Target) -> Vec<MInst> {
        let func = select(text, target).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(func.target, target);
        func.blocks.into_iter().flat_map(|b| b.insts).collect()
    }

    fn kernel(wg: u32, body: &str) -> String {
        format!(
            "ckdsl.ir v1\nkernel @probe {{\n  attrs {{ max_workgroup_size = i:{wg} }}\n  params {{\n    \
             %Y : ptr<i32,global>\n    %n : i32\n  }}\n  region @entry {{\n    \
             %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n{body}  }}\n}}\n"
        )
    }

    const GUARDED_STORE: &str = "    %lt : i1 = arith.cmp ( %tid, %n ) { pred = s:\"lt\" }\n    scf.if ( %lt )\n      \
        region @then {\n        memref.global_store_typed ( %Y, %tid, %tid ) { align = i:4, elem_type = s:\"i32\" }\n      }\n";

    fn count(insts: &[MInst], op: Opc) -> usize {
        insts.iter().filter(|i| i.op == op).count()
    }

    #[test]
    fn lane_masks_are_sgpr_pairs_on_gfx950() {
        let text = kernel(256, GUARDED_STORE);
        let func = select(&text, Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        let code: Vec<&MInst> = func.blocks.iter().flat_map(|b| &b.insts).collect();
        let cmp = code.iter().find(|i| matches!(i.op, Opc::VCmp(_))).expect("a compare");
        let Opnd::R(mask) = cmp.defs[0] else { panic!("{cmp:?}") };
        let info = func.vregs[mask.reg as usize];
        assert_eq!((info.class, info.size, info.align), (Class::S, 2, 2));
        let exec = Opnd::P(Phys::ExecLo, 2);
        let save = code.iter().find(|i| i.op == Opc::SAndSaveexecB64).expect("s_and_saveexec_b64");
        assert_eq!((save.defs[1], save.uses[0], save.uses[1]), (exec, cmp.defs[0], exec));
        let restore = code.iter().find(|i| i.op == Opc::SOrB64).expect("s_or_b64 exec");
        assert_eq!((restore.defs[0], restore.uses[0]), (exec, exec));
        assert!(!code.iter().any(|i| matches!(i.op, Opc::SAndSaveexecB32 | Opc::SOrB32)));
        // gfx1151 keeps its wave32 selection.
        let code = insts(&text, Target::Gfx1151);
        assert_eq!((count(&code, Opc::SAndSaveexecB32), count(&code, Opc::SAndSaveexecB64)), (1, 0));
        let save = code.iter().find(|i| i.op == Opc::SAndSaveexecB32).expect("s_and_saveexec_b32");
        assert_eq!(save.defs[1], Opnd::P(Phys::ExecLo, 1));
    }

    #[test]
    fn lane_ids_count_64_lanes_on_gfx950() {
        // lane_id < 32 holds for every lane of a wave32 only.
        let body = "    %l : i32 = tile.lane_id (  )\n    %c32 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:32 }\n    \
            %lt : i1 = arith.cmp ( %l, %c32 ) { pred = s:\"lt\" }\n    scf.if ( %lt )\n      region @then {\n        \
            memref.global_store_typed ( %Y, %tid, %l ) { align = i:4, elem_type = s:\"i32\" }\n      }\n";
        let text = kernel(256, body);
        let wave64 = insts(&text, Target::Gfx950);
        assert_eq!((count(&wave64, Opc::VMbcntLoU32B32), count(&wave64, Opc::VMbcntHiU32B32)), (1, 1));
        let hi = wave64.iter().find(|i| i.op == Opc::VMbcntHiU32B32).expect("v_mbcnt_hi");
        let lo = wave64.iter().find(|i| i.op == Opc::VMbcntLoU32B32).expect("v_mbcnt_lo");
        assert_eq!((hi.uses[0], hi.uses[1]), (Opnd::Imm(u32::MAX), lo.defs[0]));
        assert_eq!(count(&wave64, Opc::SAndSaveexecB64), 1);
        let wave32 = insts(&text, Target::Gfx1151);
        assert_eq!((count(&wave32, Opc::VMbcntLoU32B32), count(&wave32, Opc::VMbcntHiU32B32)), (1, 0));
        assert_eq!(count(&wave32, Opc::SAndSaveexecB32), 0);
    }

    #[test]
    fn one_wave_workgroups_follow_the_wave_size() {
        let text = kernel(64, "    tile.sync (  )\n    memref.global_store_typed ( %Y, %tid, %tid ) { align = i:4, elem_type = s:\"i32\" }\n");
        assert_eq!(count(&insts(&text, Target::Gfx1151), Opc::SBarrier), 1);
        assert_eq!(count(&insts(&text, Target::Gfx950), Opc::SBarrier), 0);
    }

    #[test]
    fn gfx950_refuses_what_it_does_not_lower_yet() {
        let body = "    %p : i32 = tile.permlane64 ( %tid )\n    memref.global_store_typed ( %Y, %tid, %p ) { align = i:4, elem_type = s:\"i32\" }\n";
        let text = kernel(256, body);
        assert!(select(&text, Target::Gfx1151).is_ok());
        let error = select(&text, Target::Gfx950).err().expect("refused");
        assert!(error.to_string().contains("tile.permlane64 on gfx950"), "{error}");
    }

    #[test]
    fn fences_follow_the_gfx950_memory_model() {
        let fence = |scope: &str, ordering: &str| {
            kernel(
                256,
                &format!(
                    "    memref.global_store_typed ( %Y, %tid, %tid ) {{ align = i:4, elem_type = s:\"i32\" }}\n    \
                     memref.fence (  ) {{ scope = s:\"{scope}\", ordering = s:\"{ordering}\" }}\n    \
                     memref.global_store_typed ( %Y, %n, %tid ) {{ align = i:4, elem_type = s:\"i32\" }}\n"
                ),
            )
        };
        let seq = |text: &str| -> Vec<(Opc, i64)> {
            insts(text, Target::Gfx950)
                .into_iter()
                .filter(|i| matches!(i.op, Opc::Gfx9(_) | Opc::SWaitcntExplicit))
                .map(|i| (i.op, i.imm))
                .collect()
        };
        let (wb, inv, wait) = (Opc::Gfx9(Gfx9Op::BufferWbl2), Opc::Gfx9(Gfx9Op::BufferInv), Opc::SWaitcntExplicit);
        // vmcnt(0) lgkmcnt(0)
        assert_eq!(seq(&fence("agent", "acq_rel")), [(wb, 2), (wait, 0x70), (inv, 2)]);
        assert_eq!(seq(&fence("system", "acquire")), [(wb, 3), (wait, 0x70), (inv, 3)]);
        assert_eq!(seq(&fence("agent", "release")), [(wb, 2), (wait, 0x70)]);
        assert_eq!(seq(&fence("workgroup", "acq_rel")), [(wait, 0x70)]);
        // Through the encoder; gfx1151 keeps its own invalidations.
        let compiled = crate::rir::compile_for(&fence("agent", "acq_rel"), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        let listing = compiled.listing();
        for wanted in ["buffer_wbl2", "buffer_inv", "s_waitcnt"] {
            assert!(listing.contains(wanted), "{wanted} missing:\n{listing}");
        }
        let gfx11 = insts(&fence("agent", "acq_rel"), Target::Gfx1151);
        assert!(gfx11.iter().any(|i| i.op == Opc::BufferGl1Inv) && !gfx11.iter().any(|i| matches!(i.op, Opc::Gfx9(_))));
    }

    #[test]
    fn xor_shuffles_read_lane_l_xor_mask() {
        for mask in 1..16u32 {
            for lane in 0..64u32 {
                let read = match xor_shuffle(mask) {
                    // quad_perm: lane L of each quad reads lane ctrl[2L+1:2L] of it.
                    XorShuffle::QuadPerm(ctrl) => lane & !3 | ctrl >> (2 * (lane & 3)) & 3,
                    // Bitmask mode: ((L & and) | or) ^ xor within each 32-lane half.
                    XorShuffle::Swizzle(offset) => {
                        let (and, or, xor) = (offset & 0x1f, offset >> 5 & 0x1f, offset >> 10 & 0x1f);
                        lane & !31 | ((lane & 31 & and) | or) ^ xor
                    }
                };
                assert_eq!(read, lane ^ mask, "mask {mask} lane {lane}");
            }
        }
    }

    #[test]
    fn gfx950_loads_both_halves_whole() {
        // Two f16 loads into the halves of one register: gfx1151 loads them
        // straight into the halves (d16), gfx950 loads each whole and packs.
        let text = "ckdsl.ir v1\nkernel @halves {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %X : ptr<f16,global>\n    \
                    %Y : ptr<i32,global>\n  }\n  region @entry {\n    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
                    %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:64 }\n    %t1 : i32 = arith.add ( %tid, %c1 )\n    \
                    %z : vec<f16x2> = arith.constant_vec (  ) { elem = s:\"f16\", fill = f:0.0, vec = i:2 }\n    \
                    %a : f16 = memref.global_load_typed ( %X, %tid ) { align = i:2, elem_type = s:\"f16\" }\n    \
                    %v0 : vec<f16x2> = vector.insert ( %z, %a ) { index = i:0 }\n    \
                    %b : f16 = memref.global_load_typed ( %X, %t1 ) { align = i:2, elem_type = s:\"f16\" }\n    \
                    %v1 : vec<f16x2> = vector.insert ( %v0, %b ) { index = i:1 }\n    \
                    %w : i32 = arith.bitcast ( %v1 )\n    \
                    memref.global_store_typed ( %Y, %tid, %w ) { align = i:4, elem_type = s:\"i32\" }\n    cf.return (  )\n  }\n}\n";
        let d16 = |code: &[MInst]| count(code, Opc::GLoadD16B16) + count(code, Opc::GLoadD16HiB16);
        assert!(d16(&insts(text, Target::Gfx1151)) > 0);
        let wave64 = insts(text, Target::Gfx950);
        assert_eq!((d16(&wave64), count(&wave64, Opc::GLoadU16)), (0, 2));
    }

    #[test]
    fn byte_inserts_keep_every_byte() {
        // Eight loaded bytes inserted into a vec<i8x8> and stored: on either
        // target each byte is merged into its dword by a v_perm_b32.
        let mut text = String::from(
            "ckdsl.ir v1\nkernel @bytes {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %X : ptr<i8,global>\n    \
             %Y : ptr<i8,global>\n  }\n  region @entry {\n    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
             %c8 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:8 }\n    %o : i32 = arith.mul ( %tid, %c8 )\n    \
             %v0 : vec<i8x8> = arith.constant_vec (  ) { elem = s:\"i8\", fill = i:0, vec = i:8 }\n",
        );
        for k in 0..8 {
            text += &format!(
                "    %c{k}k : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{k} }}\n    %a{k} : i32 = arith.add ( %o, %c{k}k )\n    \
                 %b{k} : i8 = memref.global_load_typed ( %X, %a{k} ) {{ align = i:1, elem_type = s:\"i8\" }}\n    \
                 %v{} : vec<i8x8> = vector.insert ( %v{k}, %b{k} ) {{ index = i:{k} }}\n",
                k + 1
            );
        }
        text += "    memref.global_store_vN ( %Y, %o, %v8 ) { align = i:8, elem_type = s:\"i8\", vec = i:8 }\n    cf.return (  )\n  }\n}\n";
        for target in [Target::Gfx1151, Target::Gfx950] {
            let code = insts(&text, target);
            let loads: Vec<Opnd> = code.iter().filter(|i| i.op == Opc::GLoadU8).map(|i| i.defs[0]).collect();
            assert_eq!(loads.len(), 8, "{target:?}");
            for byte in loads {
                assert!(code.iter().any(|i| i.op == Opc::VPermB32 && i.uses.contains(&byte)), "{target:?}: {byte:?} is not merged");
            }
        }
    }

    #[test]
    fn wide_forms_are_the_64_bit_ones() {
        for op in [Opc::SAndB32, Opc::SAndNot1B32, Opc::SAndSaveexecB32, Opc::SCmpLgU32, Opc::SCselectB32] {
            let wide = wide(op);
            assert_ne!(wide, op);
            assert!(wide.name().ends_with("_b64") || wide.name().ends_with("_u64"), "{}", wide.name());
        }
    }
}
