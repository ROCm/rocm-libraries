// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! What gfx950's convolution kernels need beyond the GEMM ops:
//!
//! * The kernel attribute `lds_alias_scopes` asks rocKE's LLVM lowering for
//!   one alias scope per LDS allocation, so that LLVM's waitcnt pass waits
//!   only for the LDS DMA into a read's own allocation. The direct
//!   compiler's waitcnt pass does that always (each LDS access carries the
//!   KiB range of its allocation, `lds_dma`), so the attribute changes
//!   nothing here.
//! * `arith.fdot2` selects `v_dot2_f32_f16` / `v_dot2_f32_bf16` (VOP3P), as
//!   on gfx1151; its hazards are M2 and M8 of `hazards`. COMGR emits the
//!   VOP2 `v_dot2c_*` forms of the same operation, which accumulate into
//!   their destination.
//! * `memref.global_atomic_add_pk_f16 (ptr<f16>, index, vec<f16x2>)` is the
//!   `memref.global_atomic_add` of a `vec<f16x2>` at an f16 element index:
//!   `global_atomic_pk_add_f16`, the returning form (GLC) only when the old
//!   value is used, as `llvm.amdgcn.global.atomic.fadd.v2f16` lowers.
//! * `tile.buffer_store_vN_f32` stores its dwords as the f16 and bf16 forms
//!   do (`raw.ptr.buffer.store` of the dwords), on either target.

use super::gen_isa::Gfx9Op;
use crate::rir::mir::Opc;

/// The packed f16 atomic add: the plain and the returning form.
pub fn pk_add_f16() -> (Opc, Opc) {
    (Opc::Gfx9(Gfx9Op::GAtomicPkAddF16), Opc::Gfx9(Gfx9Op::GAtomicPkAddF16Rtn))
}

#[cfg(test)]
mod f32_buffer_store {
    use crate::rir;
    use crate::target::Target;

    /// `tile.buffer_store_vN_f32` of four f32: one `buffer_store_dwordx4`, as
    /// the 16-bit forms of the same dwords.
    #[test]
    fn is_a_dword_buffer_store() {
        let text = "ckdsl.ir v1\nkernel @bstore {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    \
                    %S : ptr<f32,global>\n    %D : ptr<f32,global>\n    %D_bytes : i32\n  }\n  region @entry {\n    \
                    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
                    %c4 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:4 }\n    \
                    %c16 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:16 }\n    \
                    %c0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n    \
                    %i : i32 = arith.mul ( %tid, %c4 )\n    %off : i32 = arith.mul ( %tid, %c16 )\n    \
                    %v : vec<f32x4> = memref.global_load_vN ( %S, %i ) { align = i:16, elem_type = s:\"f32\", vec = i:4 }\n    \
                    %rsrc : vec<i32x4> = tile.buffer_rsrc ( %D, %D_bytes )\n    \
                    tile.buffer_store_vN_f32 ( %rsrc, %off, %c0, %v ) { dwords = i:4 }\n    cf.return (  )\n  }\n}\n";
        for target in [Target::Gfx950, Target::Gfx1151] {
            let listing = rir::compile_any_target(text, target).unwrap_or_else(|e| panic!("{target:?}: {e}")).listing();
            assert!(listing.contains("buffer_store_dwordx4") || listing.contains("buffer_store_b128"), "{listing}");
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    fn kernel(attrs: &str, params: &str, body: &str) -> String {
        format!(
            "ckdsl.ir v1\nkernel @conv {{\n  attrs {{ {attrs}max_workgroup_size = i:64 }}\n  params {{\n{params}  }}\n  \
             region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n{body}    cf.return (  )\n  }}\n}}\n"
        )
    }

    fn dot(elem: &str) -> String {
        kernel(
            "",
            &format!("    %A : ptr<{elem},global>\n    %D : ptr<f32,global>\n"),
            &format!(
                "    %c2 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:2 }}\n    %i : i32 = arith.mul ( %tid, %c2 )\n    \
                 %v : vec<{elem}x2> = memref.global_load_vN ( %A, %i ) {{ align = i:4, elem_type = s:\"{elem}\", vec = i:2 }}\n    \
                 %c : f32 = memref.global_load_typed ( %D, %tid ) {{ align = i:4, elem_type = s:\"f32\" }}\n    \
                 %d : f32 = arith.fdot2 ( %v, %v, %c )\n    \
                 memref.global_store_typed ( %D, %tid, %d ) {{ align = i:4, elem_type = s:\"f32\" }}\n"
            ),
        )
    }

    #[test]
    fn lds_alias_scopes_changes_no_byte() {
        let params = "    %X : ptr<f32,global>\n";
        let body = "    %x : f32 = memref.global_load_typed ( %X, %tid ) { align = i:4, elem_type = s:\"f32\" }\n    \
                    %y : f32 = arith.fadd ( %x, %x )\n    \
                    memref.global_store_typed ( %X, %tid, %y ) { align = i:4, elem_type = s:\"f32\" }\n";
        let plain = rir::compile_any_target(&kernel("", params, body), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        let scoped = rir::compile_any_target(&kernel("lds_alias_scopes = b:true, ", params, body), Target::Gfx950)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(plain.hsaco, scoped.hsaco);
    }

    #[test]
    fn the_packed_f16_atomic_returns_only_when_its_old_value_is_used() {
        let params = "    %W : ptr<f16,global>\n    %O : ptr<f16,global>\n";
        let add = "    %c2 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:2 }\n    %i : i32 = arith.mul ( %tid, %c2 )\n    \
                   %v : vec<f16x2> = memref.global_load_vN ( %O, %i ) { align = i:4, elem_type = s:\"f16\", vec = i:2 }\n    \
                   %old : vec<f16x2> = memref.global_atomic_add_pk_f16 ( %W, %i, %v ) { elem_type = s:\"f16\", ordering = s:\"monotonic\", vec = i:2 }\n";
        let kept = format!(
            "{add}    memref.global_store_vN ( %O, %i, %old ) {{ align = i:4, elem_type = s:\"f16\", vec = i:2 }}\n"
        );
        for (body, glc) in [(add.to_string(), false), (kept, true)] {
            let listing = rir::compile_any_target(&kernel("", params, &body), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
            let line = listing.lines().find(|l| l.trim_start().starts_with("global_atomic_pk_add_f16")).unwrap_or_else(|| panic!("{listing}"));
            let words: Vec<u32> = line.rsplit(": ").next().unwrap().split_whitespace().map(|w| u32::from_str_radix(w, 16).unwrap()).collect();
            // GLOBAL (segment 2) op 0x4e (dd388000 007f0308: global_atomic_pk_add_f16 v[8:9], v3, off); sc0 returns.
            assert_eq!((words[0] >> 26, words[0] >> 14 & 3, words[0] >> 18 & 0x7f), (0x37, 2, 0x4e), "{line}");
            assert_eq!(words[0] >> 16 & 1 == 1, glc, "{line}");
        }
        let e = rir::compile_any_target(&kernel("", params, add), Target::Gfx1151).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("F16"), "{e}");
    }

    #[test]
    fn fdot2_is_the_vop3p_dot_on_gfx950() {
        for (elem, mnemonic, opcode) in [("f16", "v_dot2_f32_f16", 0x23), ("bf16", "v_dot2_f32_bf16", 0x1a)] {
            let listing = rir::compile_any_target(&dot(elem), Target::Gfx950).unwrap_or_else(|e| panic!("{elem}: {e}")).listing();
            let line = listing.lines().find(|l| l.trim_start().starts_with(mnemonic)).unwrap_or_else(|| panic!("{listing}"));
            let word = u32::from_str_radix(line.rsplit(": ").next().unwrap().split_whitespace().next().unwrap(), 16).unwrap();
            // VOP3P (0xd38 in bits 31:23), the dot's opcode in bits 22:16.
            assert_eq!((word >> 23, word >> 16 & 0x7f), (0x1a7, opcode), "{line}");
        }
    }
}
