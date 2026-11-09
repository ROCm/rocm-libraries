// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! The transposing LDS reads of gfx950: `tile.ds_read_tr16_b64` and
//! `tile.ds_read_tr_b8` as `ds_read_b64_tr_b16` and `ds_read_b64_tr_b8`.
//!
//! Each lane gives the address of an element of the allocation (the LLVM
//! lowering's GEP: `rir::opt` turns the indices into a byte address and an
//! offset, `LdsReadTr`) and gets 64 bits gathered across the lanes of its
//! group: four 16-bit or eight 8-bit elements. They are plain DS loads to the
//! rest of the back end: counted on lgkmcnt, ordered with the other LDS
//! accesses, and tagged with their allocation's bytes for the LDS DMA waits
//! (`lds_dma`). Their results depend on the other lanes, so `rir::opt` never
//! merges or hoists them.
//!
//! The LLVM lowering emits `ds_read_b64_tr_b8` as inline assembly (ROCm's
//! LLVM has no intrinsic for it), past LLVM's waitcnt pass: COMGR's code reads
//! the result one wait state after the read, with no lgkmcnt wait.

use super::gen_isa::Gfx9Op;
use crate::rir::mir::Opc;

/// The read of `bits`-bit elements: 16 or 8.
pub fn opcode(bits: i64) -> Opc {
    Opc::Gfx9(if bits == 8 { Gfx9Op::DsReadB64TrB8 } else { Gfx9Op::DsReadB64TrB16 })
}

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    /// A 64-lane kernel that fills a 16x64 allocation of `elem` from `%X`,
    /// then stores the transposing read `read` (an op line of `%t`) as `%r`.
    fn kernel(elem: &str, read: &str, store_elem: &str, vec: u32) -> String {
        format!(
            "ckdsl.ir v1\nkernel @tr {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    \
             %X : ptr<{elem},global>\n    %Y : ptr<{store_elem},global>\n  }}\n  region @entry {{\n    \
             %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    \
             %c1 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:1 }}\n    \
             %c8 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:8 }}\n    \
             %c16 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:16 }}\n    \
             %t : smem<{elem}, [16x64]> = tile.smem_alloc (  )\n    \
             %x : {elem} = memref.global_load_typed ( %X, %tid ) {{ align = i:1, elem_type = s:\"{elem}\" }}\n    \
             tile.smem_store ( %t, %c0, %tid, %x ) {{ elem_type = s:\"{elem}\", rank = i:2 }}\n    \
             tile.sync (  )\n    \
             %row : i32 = arith.div ( %tid, %c16 )\n    \
             %col : i32 = arith.mod ( %tid, %c16 )\n    \
             {read}\n    \
             %o : i32 = arith.mul ( %tid, %c8 )\n    \
             memref.global_store_vN ( %Y, %o, %r ) {{ align = i:8, elem_type = s:\"{store_elem}\", vec = i:{vec} }}\n    \
             cf.return (  )\n  }}\n}}\n"
        )
    }

    fn tr16(rows: &str) -> String {
        kernel("f16", &format!("%r : vec<f16x4> = tile.ds_read_tr16_b64 ( %t, {rows} ) {{ elem_type = s:\"f16\", rank = i:2 }}"), "f16", 4)
    }

    fn tr8(rows: &str) -> String {
        let read = format!("%r : vec<fp8e4m3x8> = tile.ds_read_tr_b8 ( %t, {rows} ) {{ dtype = s:\"fp8e4m3\" }}");
        kernel("fp8e4m3", &read, "fp8e4m3", 8)
    }

    /// The encoded words of the first listing line containing `needle`.
    fn words(listing: &str, needle: &str) -> Vec<u32> {
        let line = listing.lines().find(|x| x.contains(needle)).unwrap_or_else(|| panic!("{needle} missing:\n{listing}"));
        line.rsplit(": ").next().unwrap_or("").split_whitespace().map(|w| u32::from_str_radix(w, 16).expect("a hex word")).collect()
    }

    /// (vmcnt, lgkmcnt) of each `s_waitcnt` in `lines`.
    fn waits(lines: &[&str]) -> Vec<(u32, u32)> {
        lines
            .iter()
            .filter(|l| l.trim_start().starts_with("s_waitcnt"))
            .map(|l| {
                let imm = u32::from_str_radix(l.rsplit(": ").next().unwrap_or("").trim(), 16).expect("a hex word") & 0xffff;
                (imm & 0xf | (imm >> 14 & 3) << 4, imm >> 8 & 0xf)
            })
            .collect()
    }

    #[test]
    fn transposing_reads_encode_as_llvm_mc_does_and_wait_on_lgkmcnt() {
        // Element (1, 8) of a 16x64 allocation: byte 2 * 72 for 16-bit
        // elements, 72 for 8-bit ones, in the offset field.
        for (text, mnemonic, word0) in [(tr16("%c1, %c8"), "ds_read_b64_tr_b16", 0xd9c6_0090), (tr8("%c1, %c8"), "ds_read_b64_tr_b8", 0xd9c4_0048)] {
            let listing = rir::compile_any_target(&text, Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
            assert_eq!(words(&listing, mnemonic)[0], word0, "{listing}");
            // The store of the result waits for it: lgkmcnt(0) between them.
            let lines: Vec<&str> = listing.lines().collect();
            let read = lines.iter().position(|l| l.contains(mnemonic)).expect("the read");
            let store = lines.iter().position(|l| l.contains("global_store_dwordx2")).unwrap_or_else(|| panic!("{listing}"));
            assert!(read < store && waits(&lines[read..store]).iter().any(|&(_, lgkm)| lgkm == 0), "{listing}");
        }
        // Per-lane indices: the address in a VGPR, no offset past the base.
        let listing = rir::compile_any_target(&tr16("%row, %col"), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
        assert!(listing.contains("ds_read_b64_tr_b16"), "{listing}");
    }

    #[test]
    fn transposing_reads_wait_for_the_lds_dma_into_their_allocation_only() {
        // A DMA fills %A; the read of `smem` follows with no barrier.
        let text = |smem: &str| {
            format!(
                "ckdsl.ir v1\nkernel @trdma {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    \
                 %X : ptr<f32,global> {{ align = i:16, noalias = b:true, readonly = b:true }}\n    %Y : ptr<f16,global>\n  }}\n  \
                 region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
                 %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    \
                 %c4 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:4 }}\n    \
                 %c16 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:16 }}\n    \
                 %size : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:65536 }}\n    \
                 %A : smem<f16, [16x64]> = tile.smem_alloc (  )\n    %B : smem<f16, [16x64]> = tile.smem_alloc (  )\n    \
                 %rsrc : vec<i32x4> = tile.buffer_rsrc ( %X, %size )\n    %a0 : i64 = tile.smem_addr_of ( %A )\n    \
                 %voff : i32 = arith.mul ( %tid, %c16 )\n    \
                 tile.async_buffer_load_lds_addr ( %rsrc, %a0, %voff, %c0 ) {{ aux = i:0, dwords = i:4 }}\n    \
                 %row : i32 = arith.div ( %tid, %c16 )\n    %col : i32 = arith.mod ( %tid, %c16 )\n    \
                 %r : vec<f16x4> = tile.ds_read_tr16_b64 ( {smem}, %row, %col ) {{ elem_type = s:\"f16\", rank = i:2 }}\n    \
                 %o : i32 = arith.mul ( %tid, %c4 )\n    \
                 memref.global_store_vN ( %Y, %o, %r ) {{ align = i:8, elem_type = s:\"f16\", vec = i:4 }}\n    \
                 cf.return (  )\n  }}\n}}\n"
            )
        };
        for (smem, waits_for_dma) in [("%A", true), ("%B", false)] {
            let listing = rir::compile_any_target(&text(smem), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
            let lines: Vec<&str> = listing.lines().collect();
            let dma = lines.iter().position(|l| l.contains("buffer_load_dwordx4")).unwrap_or_else(|| panic!("{listing}"));
            let read = lines.iter().position(|l| l.contains("ds_read_b64_tr_b16")).unwrap_or_else(|| panic!("{listing}"));
            assert!(dma < read, "{listing}");
            let vm = waits(&lines[dma..read]).iter().any(|&(vm, _)| vm == 0);
            assert_eq!(vm, waits_for_dma, "{smem}:\n{listing}");
        }
    }

    #[test]
    fn transposing_reads_are_gfx950_and_typed() {
        let e = rir::compile_any_target(&tr16("%row, %col"), Target::Gfx1151).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("transposing LDS read on gfx1151"), "{e}");
        // 16-bit reads of an allocation of other elements, or a result other
        // than four 16-bit elements, do not validate.
        let f32_smem = kernel("f32", "%r : vec<f16x4> = tile.ds_read_tr16_b64 ( %t, %row, %col ) { elem_type = s:\"f16\", rank = i:2 }", "f16", 4);
        let wide = kernel("f16", "%r : vec<f16x8> = tile.ds_read_tr16_b64 ( %t, %row, %col ) { elem_type = s:\"f16\", rank = i:2 }", "f16", 8);
        let rank = kernel("f16", "%r : vec<f16x4> = tile.ds_read_tr16_b64 ( %t, %row ) { elem_type = s:\"f16\", rank = i:1 }", "f16", 4);
        for text in [f32_smem, wide, rank] {
            assert!(rir::compile_any_target(&text, Target::Gfx950).is_err(), "{text}");
        }
    }
}
