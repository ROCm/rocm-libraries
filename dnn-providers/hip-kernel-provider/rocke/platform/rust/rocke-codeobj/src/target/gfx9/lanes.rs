// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Cross-lane ops of gfx950's attention kernels: `tile.permlane32_swap`
//! (`v_permlane32_swap_b32`: the upper half of the first operand's lanes
//! trades places with the lower half of the second's, as
//! `llvm.amdgcn.permlane32.swap` with `fi` and `bound_ctrl` off) and
//! `tile.readlane` (`v_readlane_b32`). The swap reads and writes both
//! registers (the row is tied), so selection swaps fresh copies of its
//! operands; h14 keeps two wait states after a VALU write of either.

use super::gen_isa::Gfx9Op;
use crate::rir::mir::Opc;

pub fn permlane32_swap() -> Opc {
    Opc::Gfx9(Gfx9Op::VPermlane32SwapB32)
}

/// `tile.permlane16_swap` (`v_permlane16_swap_b32`: the odd 16-lane rows of
/// the first operand trade places with the even rows of the second, as
/// `llvm.amdgcn.permlane16.swap` with `fi` and `bound_ctrl` off).
pub fn permlane16_swap() -> Opc {
    Opc::Gfx9(Gfx9Op::VPermlane16SwapB32)
}

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    fn kernel(body: &str) -> String {
        format!(
            "ckdsl.ir v1\nkernel @lanes {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %Y : ptr<i32,global>\n    %n : i32\n  }}\n  \
             region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %c3 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:3 }}\n    \
             %c5 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:5 }}\n    \
             %x : i32 = arith.add ( %tid, %c3 )\n    %y : i32 = arith.mul ( %tid, %c3 )\n{body}    \
             memref.global_store_typed ( %Y, %tid, %z ) {{ align = i:4, elem_type = s:\"i32\" }}\n    cf.return (  )\n  }}\n}}\n"
        )
    }

    const SWAP_AND_READ: &str = "    %p : i32, %q : i32 = tile.permlane32_swap ( %x, %y )\n    \
                                 %s : i32 = tile.readlane ( %p, %c5 )\n    %z : i32 = arith.add ( %q, %s )\n";

    /// The encoded words of each listing line containing `needle`.
    fn words(listing: &str, needle: &str) -> Vec<Vec<u32>> {
        listing
            .lines()
            .filter(|l| l.contains(needle))
            .map(|l| l.rsplit(": ").next().unwrap_or("").split_whitespace().map(|w| u32::from_str_radix(w, 16).expect("a hex word")).collect())
            .collect()
    }

    #[test]
    fn swaps_and_lane_reads_encode_as_llvm_mc_does() {
        let listing = rir::compile_any_target(&kernel(SWAP_AND_READ), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
        // VOP1 0x5a with VGPR operands (`v_permlane32_swap_b32 v0, v1` is 7e00b501).
        let swap = words(&listing, "v_permlane32_swap_b32");
        assert!(swap.len() == 1 && swap[0][0] & 0xfe01_ff00 == 0x7e00_b500, "{listing}");
        // VOP3 0x289, lane 5 as an inline constant (`v_readlane_b32 s4, v1, 5` is d2890004 00010b01).
        let read = words(&listing, "v_readlane_b32");
        assert!(read.len() == 1 && read[0][0] & 0xffff_ff00 == 0xd289_0000 && read[0][1] >> 9 & 0x1ff == 0x85, "{listing}");
        // h14: two wait states between the copies into the swapped registers and the swap.
        let lines: Vec<&str> = listing.lines().collect();
        let at = lines.iter().position(|l| l.contains("v_permlane32_swap_b32")).expect("the swap");
        assert!(lines[at - 1].contains("s_nop"), "{listing}");
    }

    #[test]
    fn a_uniform_value_is_its_own_lane_read_and_gfx1151_refuses_both() {
        let uniform = "    %s : i32 = tile.readlane ( %n, %c5 )\n    %z : i32 = arith.add ( %x, %s )\n";
        let listing = rir::compile_any_target(&kernel(uniform), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
        assert!(!listing.contains("v_readlane_b32"), "{listing}");
        for body in [SWAP_AND_READ, uniform] {
            let e = rir::compile_any_target(&kernel(body), Target::Gfx1151).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains("on gfx1151"), "{e}");
        }
    }
}
