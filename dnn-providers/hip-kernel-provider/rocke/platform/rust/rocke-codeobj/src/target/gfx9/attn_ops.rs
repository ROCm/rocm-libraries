// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Ops gfx950's attention kernels and probes need beyond the ones attention
//! backward brought, selected in `rir::isel`:
//!
//! * `vector.and (vec<i32xN>, vec<i32xN>)`: `v_and_b32` per dword (a splat
//!   operand stays a scalar), as the vector arithmetic.
//! * `vector.lshr (vec<i32xN>, vec<i32xN>)`: `v_lshrrev_b32` per dword, the
//!   amount first; a splat operand stays a scalar.
//! * `tile.writelane (value, lane, passthrough)`: a copy of `passthrough`
//!   with `v_writelane_b32` writing the uniform value (an SGPR or an inline
//!   constant) into a constant lane; any other lane is refused.
//! * `tile.byte_perm (a, b) {sel}`: `v_perm_b32` with the selector in an SGPR
//!   (gfx9's VOP3 takes no literal).
//! * A compare of two i8 values: each widened to 32 bits, sign-extended for
//!   the signed predicates and zero-extended for `eq`, `ne` and `ult` (the
//!   i8 is one byte of a word whose other bytes hold anything), then the
//!   i32 compare (gfx950; gfx1151 keeps its refusal).
//! * `tile.s_wait_asynccnt`: nothing on gfx950, which has no asynchronous
//!   LDS copy counter (the reference lowering emits nothing on any target
//!   without one); gfx1151 keeps its refusal.

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    fn kernel(params: &str, body: &str) -> String {
        format!(
            "ckdsl.ir v1\nkernel @attn {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n{params}  }}\n  \
             region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n{body}    cf.return (  )\n  }}\n}}\n"
        )
    }

    #[test]
    fn vector_and_is_one_and_per_dword() {
        let params = "    %X : ptr<i32,global>\n    %m : i32\n";
        let body = "    %c2 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:2 }\n    %i : i32 = arith.mul ( %tid, %c2 )\n    \
                    %x : vec<i32x2> = memref.global_load_vN ( %X, %i ) { align = i:8, elem_type = s:\"i32\", vec = i:2 }\n    \
                    %s : vec<i32x2> = vector.splat ( %m ) { vec = i:2 }\n    %a : vec<i32x2> = vector.and ( %x, %s )\n    \
                    memref.global_store_vN ( %X, %i, %a ) { align = i:8, elem_type = s:\"i32\", vec = i:2 }\n";
        let listing = rir::compile_any_target(&kernel(params, body), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
        assert_eq!(listing.matches("v_and_b32").count(), 2, "{listing}");
    }

    #[test]
    fn vector_lshr_is_one_reversed_shift_per_dword() {
        let params = "    %X : ptr<i32,global>\n    %m : i32\n";
        let body = "    %c2 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:2 }\n    %i : i32 = arith.mul ( %tid, %c2 )\n    \
                    %x : vec<i32x2> = memref.global_load_vN ( %X, %i ) { align = i:8, elem_type = s:\"i32\", vec = i:2 }\n    \
                    %s : vec<i32x2> = vector.splat ( %m ) { vec = i:2 }\n    %a : vec<i32x2> = vector.lshr ( %x, %s )\n    \
                    memref.global_store_vN ( %X, %i, %a ) { align = i:8, elem_type = s:\"i32\", vec = i:2 }\n";
        let listing = rir::compile_any_target(&kernel(params, body), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
        assert_eq!(listing.matches("v_lshrrev_b32").count(), 2, "{listing}");
    }

    #[test]
    fn byte_perm_is_one_v_perm_b32_with_its_selector_in_an_sgpr() {
        let params = "    %X : ptr<i32,global>\n";
        let body = "    %a : i32 = memref.global_load_typed ( %X, %tid ) { align = i:4, elem_type = s:\"i32\" }\n    \
                    %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n    %i : i32 = arith.add ( %tid, %c1 )\n    \
                    %b : i32 = memref.global_load_typed ( %X, %i ) { align = i:4, elem_type = s:\"i32\" }\n    \
                    %p : i32 = tile.byte_perm ( %a, %b ) { sel = i:117834498 }\n    \
                    memref.global_store_typed ( %X, %tid, %p ) { align = i:4, elem_type = s:\"i32\" }\n";
        let listing = rir::compile_any_target(&kernel(params, body), Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
        assert_eq!(listing.matches("v_perm_b32").count(), 1, "{listing}");
        assert!(listing.contains("0x7060302") || listing.contains("117506818"), "{listing}");
    }

    #[test]
    fn writelane_writes_a_constant_lane_of_a_copy() {
        let params = "    %X : ptr<i32,global>\n    %m : i32\n";
        let write = |value: &str, lane: i64| {
            format!(
                "    %x : i32 = memref.global_load_typed ( %X, %tid ) {{ align = i:4, elem_type = s:\"i32\" }}\n    \
                 %c7 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:7 }}\n    \
                 %l : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{lane} }}\n    \
                 %w : i32 = tile.writelane ( {value}, %l, %x )\n    \
                 memref.global_store_typed ( %X, %tid, %w ) {{ align = i:4, elem_type = s:\"i32\" }}\n"
            )
        };
        for value in ["%m", "%c7"] {
            let listing = rir::compile_any_target(&kernel(params, &write(value, 5)), Target::Gfx950).unwrap_or_else(|e| panic!("{value}: {e}")).listing();
            assert_eq!(listing.matches("v_writelane_b32").count(), 1, "{listing}");
            assert!(listing.contains(", 5"), "{listing}");
        }
        // A lane that is not a constant is refused, not miscompiled.
        let dynamic = write("%m", 5).replace("%l : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:5 }", "%l : i32 = arith.add ( %tid, %c7 )");
        assert!(rir::compile_any_target(&kernel(params, &dynamic), Target::Gfx950).is_err());
    }

    /// `x <pred> -2` for an i8 `x` loaded from memory: the bytes 0x7f and 0x80
    /// order differently signed (127 > -128) and unsigned (127 < 128).
    fn byte_compare(pred: &str) -> String {
        kernel(
            "    %X : ptr<i8,global>\n    %Y : ptr<i32,global>\n",
            &format!(
                "    %x : i8 = memref.global_load_typed ( %X, %tid ) {{ align = i:1, elem_type = s:\"i8\" }}\n    \
                 %k : i8 = arith.constant (  ) {{ ity = s:\"i8\", value = i:-2 }}\n    \
                 %p : i1 = arith.cmp ( %x, %k ) {{ pred = s:\"{pred}\" }}\n    \
                 %one : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:1 }}\n    \
                 %zero : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    \
                 %y : i32 = arith.select ( %p, %one, %zero )\n    \
                 memref.global_store_typed ( %Y, %tid, %y ) {{ align = i:4, elem_type = s:\"i32\" }}\n"
            ),
        )
    }

    #[test]
    fn byte_compares_widen_by_the_predicates_signedness() {
        for (pred, widen, cmp) in [("lt", "v_bfe_i32", "v_cmp_lt_i32"), ("ne", "v_and_b32", "v_cmp_ne_u32"), ("ult", "v_and_b32", "v_cmp_lt_u32")] {
            let listing = rir::compile_any_target(&byte_compare(pred), Target::Gfx950).unwrap_or_else(|e| panic!("{pred}: {e}")).listing();
            assert!(listing.contains(widen), "{pred}: {listing}");
            assert_eq!(listing.matches("v_cmp_").count(), 1, "{pred}: {listing}");
            assert!(listing.contains(cmp) || pred == "ne" && listing.contains("v_cmp_ne_i32"), "{pred}: {listing}");
        }
        let e = rir::compile_any_target(&byte_compare("lt"), Target::Gfx1151).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("I8) compare"), "{e}");
    }

    #[test]
    fn wait_asynccnt_changes_no_byte_on_gfx950() {
        let wait = "    tile.s_waitcnt (  ) { expcnt = i:-1, lgkmcnt = i:0, vmcnt = i:0 }\n";
        let async_wait = |n: u32| format!("    tile.s_wait_asynccnt (  ) {{ n = i:{n} }}\n");
        let with = kernel("", &format!("{}{wait}{}", async_wait(0), async_wait(3)));
        let listing = |text: &str| rir::compile_any_target(text, Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing();
        assert_eq!(listing(&with), listing(&kernel("", wait)));
        assert!(rir::compile_any_target(&with, Target::Gfx1151).is_err());
    }
}
