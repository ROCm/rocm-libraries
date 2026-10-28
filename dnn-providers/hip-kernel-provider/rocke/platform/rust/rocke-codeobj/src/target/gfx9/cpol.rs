// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! gfx950 cache policy of global loads and stores, as `MInst` modifier bits
//! the rows with `m=sc1` and `m=nt` take (`isa/gfx950.isa`; past `mir.rs`'s
//! `M_*`, which gfx11 shares):
//!
//! * sc1: the access is coherent at agent scope. An atomic load or store at
//!   agent scope sets it, as gfx950's memory model writes them.
//! * nt: nontemporal (the IR's `temporal_hint STREAMING`). gfx950 is one of
//!   the targets whose reference lowering marks it, so a vector access's
//!   `nontemporal` becomes the `nt` attribute (`take_nontemporal`), which
//!   legalization keeps and selection turns into this bit; on gfx1151
//!   `validate` refuses it, as the reference does.
//!
//! The flag ops of stream-K's partial-tile handoff (`rir::isel`), at agent
//! scope as LLVM lowers them on gfx950:
//!
//! * `memref.global_flag_wait_eq (flags, index, value)`: each lane polls its
//!   flag until it holds `value`, an acquire load per trip
//!   (`global_load_dword ... sc1`, `s_waitcnt vmcnt(0)`, `buffer_inv sc1`);
//!   a lane leaves EXEC once its flag matched, and EXEC is restored after.
//! * `memref.global_flag_store (flags, index, value)`: `global_store_dword
//!   ... sc1`, after the release fence (`buffer_wbl2 sc1`, `s_waitcnt
//!   vmcnt(0) lgkmcnt(0)`) unless the ordering is monotonic.

use crate::rir::ast::{Attr, Kernel, OpKind, Region};

pub const M_SC1: u32 = 1 << 13;
pub const M_NT: u32 = 1 << 14;

/// Renames `nontemporal = true` on `kernel`'s vector global loads and stores
/// to `nt` (gfx950).
pub fn take_nontemporal(kernel: &mut Kernel) {
    fn walk(region: &mut Region) {
        for op in region.ops.iter_mut() {
            if matches!(op.kind, OpKind::GlobalLoadVec | OpKind::GlobalStoreVec) {
                for (key, value) in op.attrs.iter_mut() {
                    if &**key == "nontemporal" && matches!(value, Attr::Bool(true)) {
                        *key = "nt".into();
                    }
                }
            }
            op.regions.iter_mut().for_each(walk);
        }
    }
    walk(&mut kernel.body);
}

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    /// The census probe's copy of a vector with nontemporal load and store.
    const NONTEMPORAL: &str = "ckdsl.ir v1\nkernel @nontemporal {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    \
                               %S : ptr<bf16,global>\n    %D : ptr<bf16,global>\n  }\n  region @entry {\n    \
                               %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
                               %c8 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:8 }\n    \
                               %i : i32 = arith.mul ( %tid, %c8 )\n    \
                               %v : vec<bf16x8> = memref.global_load_vN ( %S, %i ) { align = i:16, elem_type = s:\"bf16\", nontemporal = b:true, vec = i:8 }\n    \
                               memref.global_store_vN ( %D, %i, %v ) { align = i:16, elem_type = s:\"bf16\", nontemporal = b:true, vec = i:8 }\n    \
                               cf.return (  )\n  }\n}\n";

    #[test]
    fn nontemporal_accesses_set_nt_on_gfx950_only() {
        let listing = rir::compile_any_target(NONTEMPORAL, Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
        let nt = |m: &str| words_of(&listing, m).iter().all(|(_, w)| w >> 17 & 1 == 1);
        assert!(nt("global_load_dwordx4") && nt("global_store_dwordx4"), "{listing}");
        let e = rir::compile_any_target(NONTEMPORAL, Target::Gfx1151).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("nontemporal"), "{e}");
    }

    fn kernel(body: &str) -> String {
        format!(
            "ckdsl.ir v1\nkernel @flags {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %F : ptr<i32,global>\n    \
             %X : ptr<f32,global>\n    %Y : ptr<f32,global>\n  }}\n  region @entry {{\n    \
             %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %one : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:1 }}\n{body}    cf.return (  )\n  }}\n}}\n"
        )
    }

    /// Waits for its flag, copies a partial result, and hands the flag on.
    const HANDOFF: &str = "    memref.global_flag_wait_eq ( %F, %tid, %one ) { scope = s:\"agent\" }\n    \
                           %x : f32 = memref.global_load_typed ( %X, %tid ) { align = i:4, elem_type = s:\"f32\" }\n    \
                           memref.global_store_typed ( %Y, %tid, %x ) { align = i:4, elem_type = s:\"f32\" }\n    \
                           memref.fence (  ) { ordering = s:\"release\", scope = s:\"agent\" }\n    \
                           memref.global_flag_store ( %F, %tid, %one ) { ordering = s:\"release\", scope = s:\"agent\" }\n";

    /// The listing lines of `mnemonic` with their first encoded word.
    fn words_of<'a>(listing: &'a str, mnemonic: &str) -> Vec<(&'a str, u32)> {
        listing
            .lines()
            .filter(|l| l.trim_start().starts_with(mnemonic))
            .map(|l| {
                let hex = l.rsplit(": ").next().unwrap_or("").split_whitespace().next().unwrap_or("");
                (l, u32::from_str_radix(hex, 16).unwrap_or_else(|_| panic!("no encoding in {l}")))
            })
            .collect()
    }

    #[test]
    fn the_flag_handoff_is_an_acquire_spin_and_a_release_store() {
        let listing = rir::compile_any_target(&kernel(HANDOFF), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
        let sc1 = |w: u32| w >> 25 & 1 == 1;
        let loads = words_of(&listing, "global_load_dword ");
        assert_eq!(loads.iter().filter(|(_, w)| sc1(*w)).count(), 1, "one acquire load: {listing}");
        let stores = words_of(&listing, "global_store_dword ");
        assert_eq!(stores.iter().filter(|(_, w)| sc1(*w)).count(), 1, "one flag store: {listing}");
        assert!(stores.last().is_some_and(|(_, w)| sc1(*w)), "the flag store is the last: {listing}");
        let order: Vec<&str> = listing
            .lines()
            .map(str::trim_start)
            .filter(|l| ["global_load_dword ", "buffer_inv", "s_cbranch_execnz", "buffer_wbl2", "global_store_dword "].iter().any(|m| l.starts_with(m)))
            .map(|l| l.split_whitespace().next().unwrap_or(""))
            .collect();
        let at = |m: &str| order.iter().position(|x| *x == m).unwrap_or_else(|| panic!("no {m}: {listing}"));
        assert!(at("global_load_dword") < at("buffer_inv") && at("buffer_inv") < at("s_cbranch_execnz"), "{listing}");
        assert!(at("buffer_wbl2") < order.iter().rposition(|x| *x == "global_store_dword").unwrap(), "{listing}");
        assert!(rir::compile_any_target(&kernel(HANDOFF), Target::Gfx1151).is_err());
    }

    #[test]
    fn only_agent_scope_flags_are_selected() {
        let wait = |scope: &str| kernel(&format!("    memref.global_flag_wait_eq ( %F, %tid, %one ) {{ scope = s:\"{scope}\" }}\n"));
        assert!(rir::compile_any_target(&wait("agent"), Target::Gfx950).is_ok());
        for scope in ["workgroup", "system"] {
            let e = rir::compile_any_target(&wait(scope), Target::Gfx950).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains(scope), "{scope}: {e}");
        }
    }
}
