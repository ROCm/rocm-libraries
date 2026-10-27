// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! What instruction selection picks differently on gfx9 (wave64). The gfx11
//! choices stay in `rir::isel`; it calls these where the targets differ.

use crate::rir::mir::Opc;

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
    match mask {
        1 => XorShuffle::QuadPerm(0xb1),
        2 => XorShuffle::QuadPerm(0x4e),
        3 => XorShuffle::QuadPerm(0x1b),
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
    fn wide_forms_are_the_64_bit_ones() {
        for op in [Opc::SAndB32, Opc::SAndNot1B32, Opc::SAndSaveexecB32, Opc::SCmpLgU32, Opc::SCselectB32] {
            let wide = wide(op);
            assert_ne!(wide, op);
            assert!(wide.name().ends_with("_b64") || wide.name().ends_with("_u64"), "{}", wide.name());
        }
    }
}
