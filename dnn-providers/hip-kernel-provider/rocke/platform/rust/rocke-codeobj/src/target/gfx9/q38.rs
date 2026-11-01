// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! The Qwen3.8 monokernel's persistent-kernel and MX ops on gfx950 (R2 of the
//! direct-compiler pilot): what selection makes of them, checked here against
//! the words `llvm-mc -mcpu=gfx950` and COMGR give the same instructions.
//!
//! - `tile.buffer_load_cp` / `tile.buffer_store_cp`: MUBUF loads and stores
//!   with the cache policy `aux` (sc0 1, nt 2, sc1 16) in the format's sc0,
//!   nt and sc1 bits (`rir::isel::cache_policy_mods`).
//! - `tile.tagged_poll`: EXEC saved; per try the buffer loads (cache bits
//!   `aux`), one `v_cmp_ne` per tag word, EXEC narrowed to the lanes still
//!   waiting, `s_cbranch_execz` out; the pause; a branch back; EXEC restored.
//! - `tile.int_to_ptr`: the address's registers, as a pointer.
//! - `arith.cvt_scalef32_pk_bf16_fp4`: `v_cvt_scalef32_pk_bf16_fp4`, the
//!   byte select in op_sel[1:0].
//! - `tile.permlane16_swap`: `v_permlane16_swap_b32` on copies (tied, h14).
//! - `tile.s_getreg`: `s_getreg_b32` of `hwreg(id, 0, 32)`; `tile.s_sleep`:
//!   `s_sleep` steps of at most 127; `tile.memrealtime`: `s_memrealtime` (its
//!   lgkmcnt wait is the first read's).
//! - `tile.global_atomic_add_scoped`: `global_atomic_add[_x2]` through saddr,
//!   sc1 at system scope, sc0 when it returns, between scheduling fences.
//! - `tile.l2_touch_load` / `tile.l2_touch_drain`: `global_load_dwordx4`
//!   through saddr into registers held (`KEEP_ALIVE`) until the drain's
//!   `s_waitcnt vmcnt(0)`.
//! - kernarg preload (`inreg`): `rir::opt::kernarg_preload`, the precolored
//!   user SGPRs (`rir::isel`), `super::with_preload` and the 256-byte
//!   `super::preload_entry` for firmware that does not preload.
//!
//! The rust track (decision 6, accuracy-gated) adds:
//!
//! - `math.exp` (f32): LLVM's expansion with fast FMA (`ph = x log2e`, its
//!   FMA residual and the low part of log2e, `v_rndne`, `v_exp_f32` of the
//!   fraction, `v_ldexp`, then 0 below the denormal range and +inf above
//!   the f32 range).
//! - `arith.cvt_scalef32_pk_{fp8,bf8}_f32`: `v_cvt_scalef32_pk_{fp8,bf8}_f32`
//!   on a copy of `old` (tied), `hi` the destination op_sel bit.
//! - `tile.mfma_scale_mx`: `v_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4`, the
//!   formats in cbsz / blgp and the scale byte selects in the
//!   `v_mfma_ld_scale_b32` word's op_sel / op_sel_hi (`super::mfma::select_mx`).
//! - `arith.maximum3_f32 {abs}`: `v_maximum3_f32` with VOP3 abs (`M_ABS0`).
//! - `tile.dpp_op (old, src0[, src1]) {op, ctrl, row_mask, bank_mask,
//!   bound_ctrl}`: `v_mov_b32_dpp` / `v_max_u32_dpp` on a copy of `old`, the
//!   lanes the masks (or an out-of-row source without bound_ctrl) leave
//!   unwritten keeping it; a move writing every lane is the untied row.

/// VOP3 abs of source 0 (`<< 1`, `<< 2`: sources 1, 2): the `abs` modifier
/// set of `isa/gfx950.isa`, past `mir.rs`'s `M_*` and `cpol.rs`'s bits.
pub const M_ABS0: u32 = 1 << 20;
pub const M_ABS1: u32 = 1 << 21;
pub const M_ABS2: u32 = 1 << 22;

/// The f32 constants of `math.exp` (LLVM `SITargetLowering::lowerFEXP`).
pub mod exp_consts {
    /// log2(e) rounded to f32.
    pub const LOG2E: u32 = 0x3fb8_aa3b;
    /// 0x1.4ae0bep-26: log2(e) - LOG2E, the low part.
    pub const LOG2E_LO: u32 = 0x32a5_705f;
    /// -0x1.9d1da0p+6: below it the result is 0.
    pub const UNDERFLOW: u32 = 0xc2ce_8ed0;
    /// 0x1.62e430p+6: above it the result is +inf.
    pub const OVERFLOW: u32 = 0x42b1_7218;
}

/// Whether `ctrl` is a gfx9 DPP16 control (`tile.dpp_op`): quad_perm,
/// row_shl / row_shr / row_ror 1-15, wave shifts and rotates, the mirrors
/// and row_bcast 15 / 31 (as the encoder's `dpp_control_on_gfx9`).
pub fn dpp_ctrl_ok(ctrl: i64) -> bool {
    matches!(ctrl, 0x000..=0x0ff | 0x101..=0x10f | 0x111..=0x11f | 0x121..=0x12f | 0x130 | 0x134 | 0x138 | 0x13c | 0x140..=0x143)
}

/// The DPP word bits 31:8 (`imm` of a dpp16 row) of `tile.dpp_op`'s
/// attributes: control, bound_ctrl (bit 11), bank mask (16), row mask (20).
pub fn dpp_imm(ctrl: i64, row_mask: i64, bank_mask: i64, bound_ctrl: bool) -> i64 {
    (ctrl & 0x1ff) | (bound_ctrl as i64) << 11 | (bank_mask & 15) << 16 | (row_mask & 15) << 20
}

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::rir::mir::*;
    use crate::target::Target;

    fn kernel(params: &str, body: &str) -> String {
        format!(
            "ckdsl.ir v1\nkernel @q38 {{\n  attrs {{ max_workgroup_size = i:256 }}\n  params {{\n    %Y : ptr<i32,global>\n{params}  }}\n  \
             region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %c64 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:4096 }}\n    \
             %rs : vec<i32x4> = tile.buffer_rsrc ( %Y, %c64 )\n    %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    \
             %c8 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:8 }}\n    %off : i32 = arith.mul ( %tid, %c8 )\n\
             {body}    memref.global_store_typed ( %Y, %tid, %r ) {{ align = i:4, elem_type = s:\"i32\" }}\n    cf.return (  )\n  }}\n}}\n"
        )
    }

    fn listing(text: &str) -> String {
        rir::compile_any_target(text, Target::Gfx950).unwrap_or_else(|e| panic!("{e}\n{text}")).listing()
    }

    /// `text` with lever scope `scope` (`dlevers`) on every op of its body.
    fn stamp(text: &str, scope: &str) -> String {
        let mut body = false;
        let mut out = String::new();
        for line in text.lines() {
            let tr = line.trim_start();
            body |= tr.starts_with("region @entry");
            if body && tr.contains(" ( ") && !tr.starts_with("region") {
                if let Some(at) = line.find(" ) { ") {
                    out += &format!("{} ) {{ dlevers = s:\"{scope}\", {}", &line[..at], &line[at + 5..]);
                } else {
                    out += &format!("{line} {{ dlevers = s:\"{scope}\" }}");
                }
            } else {
                out += line;
            }
            out.push('\n');
        }
        out
    }

    /// The listings of `text` with `levers` as the kernel's levers and as every op's scope are the same, and a
    /// scope turning them off gives the plain listing back.
    fn scopes_match_kernel_levers(text: &str, levers: &[&str]) {
        let plain = listing(text);
        let lev = rir::with_levers(levers, || listing(text));
        assert_eq!(listing(&stamp(text, &levers.join(","))), lev, "scoped {levers:?}");
        let off: Vec<String> = levers.iter().map(|l| format!("-{l}")).collect();
        assert_eq!(rir::with_levers(levers, || listing(&stamp(text, &off.join(",")))), plain, "scoped off {levers:?}");
    }

    /// (mnemonic + operands, first word, second word) of every listing line starting with `mnemonic`.
    fn lines<'a>(listing: &'a str, mnemonic: &str) -> Vec<(&'a str, u32, u32)> {
        listing
            .lines()
            .map(str::trim_start)
            .filter(|l| l.starts_with(mnemonic))
            .map(|l| {
                let mut words = l.rsplit(": ").next().unwrap_or("").split_whitespace().map(|h| u32::from_str_radix(h, 16).unwrap_or(0));
                let w0 = words.next().unwrap_or(0);
                (l, w0, words.next().unwrap_or(0))
            })
            .collect()
    }

    #[test]
    fn buffer_accesses_carry_their_cache_policy() {
        let body = "    %v : vec<i32x2> = tile.buffer_load_cp ( %rs, %off, %c0 ) { aux = i:19, dwords = i:2 }\n    \
                    %a : i32 = vector.extract ( %v ) { index = i:0 }\n    \
                    tile.buffer_store_cp ( %v, %rs, %off, %c0 ) { aux = i:18, dwords = i:2 }\n    \
                    %w : i32 = tile.buffer_load_cp ( %rs, %off, %c0 ) { aux = i:0, dwords = i:1 }\n    \
                    %r : i32 = arith.add ( %a, %w )\n";
        let l = listing(&kernel("", body));
        // MUBUF word 0: sc0 bit 14, sc1 bit 15, nt bit 17 (llvm-mc: `buffer_load_dwordx2 ... sc0 nt sc1`).
        let bits = |w: u32| (w >> 14 & 1, w >> 15 & 1, w >> 17 & 1);
        let x2 = lines(&l, "buffer_load_dwordx2");
        assert_eq!(x2.len(), 1, "{l}");
        assert_eq!(bits(x2[0].1), (1, 1, 1), "{l}");
        let st = lines(&l, "buffer_store_dwordx2");
        assert_eq!(bits(st[0].1), (0, 1, 1), "{l}");
        let x1 = lines(&l, "buffer_load_dword ");
        assert_eq!(bits(x1[0].1), (0, 0, 0), "{l}");
    }

    #[test]
    fn a_tagged_poll_narrows_exec_until_every_tag_matches() {
        let body = "    %p0 : vec<i32x4>, %p1 : vec<i32x2> = tile.tagged_poll ( %n, %rs, %off, %rs, %c0 ) \
                    { aux = i:16, dwords = l:[ i:4, i:2 ], pause = s:\"s_sleep 1\" }\n    \
                    %a : i32 = vector.extract ( %p0 ) { index = i:2 }\n    %b : i32 = vector.extract ( %p1 ) { index = i:0 }\n    \
                    %r : i32 = arith.add ( %a, %b )\n";
        let l = listing(&kernel("    %n : i32\n", body));
        let seq: Vec<String> = l
            .lines()
            .map(str::trim_start)
            .filter(|x| {
                ["s_mov_b64 s", "buffer_load_dwordx4", "buffer_load_dwordx2", "v_cmp_ne_i32", "s_and_b64 exec", "s_cbranch_execz", "s_sleep", "s_branch", "s_mov_b64 exec"]
                    .iter()
                    .any(|m| x.starts_with(m))
            })
            .map(|x| x.split(" //").next().unwrap_or(x).split_whitespace().take(2).collect::<Vec<_>>().join(" "))
            .collect();
        let at = |m: &str| seq.iter().position(|x| x.starts_with(m)).unwrap_or_else(|| panic!("no {m}: {l}"));
        // Saved EXEC, both loads (sc1), three tag compares (dwords 1, 3 of the x4; 1 of the x2), the narrowing,
        // the exit, the pause, the back edge, the restore.
        assert!(at("s_mov_b64 s") < at("buffer_load_dwordx4"), "{seq:?}");
        assert_eq!(seq.iter().filter(|x| x.starts_with("v_cmp_ne_i32")).count(), 3, "{seq:?}");
        assert!(at("buffer_load_dwordx2") < at("v_cmp_ne_i32") && at("v_cmp_ne_i32") < at("s_and_b64 exec"), "{seq:?}");
        assert!(at("s_and_b64 exec") < at("s_cbranch_execz") && at("s_cbranch_execz") < at("s_sleep"), "{seq:?}");
        assert!(at("s_sleep") < at("s_branch") && at("s_branch") < at("s_mov_b64 exec"), "{seq:?}");
        for (line, w, _) in lines(&l, "buffer_load_dwordx").iter().copied() {
            assert_eq!(w >> 15 & 1, 1, "sc1: {line}");
        }
        // The pause `s_nop 0` (default) and none.
        let nop = listing(&kernel("    %n : i32\n", &body.replace("s_sleep 1", "s_nop 0")));
        assert!(!nop.contains("s_sleep") && nop.contains("s_branch"), "{nop}");
    }

    #[test]
    fn scaled_converts_select_their_byte_with_op_sel() {
        let mut body = String::from("    %h : f32 = arith.bitcast ( %n )\n    %r0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }\n");
        for byte in 0..4 {
            body += &format!(
                "    %d{byte} : vec<bf16x2> = arith.cvt_scalef32_pk_bf16_fp4 ( %tid, %h ) {{ byte = i:{byte} }}\n    \
                 %i{byte} : i32 = vector.bitcast ( %d{byte} )\n    %r{} : i32 = arith.xor ( %r{byte}, %i{byte} )\n",
                byte + 1
            );
        }
        body += "    %r : i32 = arith.add ( %r4, %tid )\n";
        let l = listing(&kernel("    %n : i32\n", &body));
        let cvt = lines(&l, "v_cvt_scalef32_pk_bf16_fp4");
        assert_eq!(cvt.len(), 4, "{l}");
        // VOP3 0x251; op_sel bits 11 and 12 of word 0 are the byte (llvm-mc: op_sel:[1,0,0] = 0x0800).
        let mut sel: Vec<u32> = cvt.iter().map(|(_, w, _)| w >> 11 & 3).collect();
        sel.sort_unstable();
        assert_eq!(sel, vec![0, 1, 2, 3], "{l}");
        assert!(cvt.iter().all(|(_, w, _)| w >> 16 & 0x3ff == 0x251), "{l}");
    }

    #[test]
    fn swaps_reads_and_clocks_encode_as_llvm_mc_does() {
        let body = "    %x : i32, %y : i32 = tile.permlane16_swap ( %tid, %n )\n    \
                    %g : i32 = tile.s_getreg (  ) { hwreg = s:\"XCC_ID\" }\n    tile.s_sleep (  ) { units = i:200 }\n    \
                    %t : i64 = tile.memrealtime (  )\n    %tl : i32 = arith.trunc ( %t )\n    \
                    %s : i32 = arith.add ( %x, %y )\n    %s2 : i32 = arith.add ( %s, %g )\n    %r : i32 = arith.add ( %s2, %tl )\n";
        let l = listing(&kernel("    %n : i32\n", body));
        // v_permlane16_swap_b32_e32 (VOP1 0x59), s_getreg_b32 hwreg(HW_REG_XCC_ID) = SOPK 0x11 imm 0xf814,
        // s_sleep 127 then 73 (SOPP 0xe), s_memrealtime (SMEM 0x25, imm bit clear).
        let p = lines(&l, "v_permlane16_swap_b32");
        assert!(p.len() == 1 && p[0].1 >> 9 & 0xff == 0x59 && p[0].1 >> 25 == 0x3f, "{l}");
        let g = lines(&l, "s_getreg_b32");
        assert!(g.len() == 1 && g[0].1 & 0xffff == 0xf814 && g[0].1 >> 23 & 0x1f == 0x11, "{l}");
        let sl: Vec<u32> = lines(&l, "s_sleep").iter().map(|(_, w, _)| *w).collect();
        assert_eq!(sl, vec![0xbf8e_007f, 0xbf8e_0049], "{l}");
        let m = lines(&l, "s_memrealtime");
        assert!(m.len() == 1 && m[0].1 & 0xfffc_003f == 0xc094_0000 && m[0].2 == 0, "{l}");
    }

    #[test]
    fn scoped_atomics_return_with_sc0_and_scope_sc1() {
        let body = "    %w : i64 = arith.zext ( %n )\n    \
                    %o : i64 = tile.global_atomic_add_scoped ( %c0, %w, %w ) { returns = b:true, scope = s:\"system\" }\n    \
                    %o32 : i32 = tile.global_atomic_add_scoped ( %tid, %n, %w ) { returns = b:true, scope = s:\"agent\" }\n    \
                    tile.global_atomic_add_scoped ( %tid, %n, %w ) { returns = b:false, scope = s:\"system\" }\n    \
                    %t : i32 = arith.trunc ( %o )\n    %r : i32 = arith.add ( %t, %o32 )\n";
        let l = listing(&kernel("    %n : i32\n", body));
        // GLOBAL word 0: sc0 bit 16, sc1 bit 25; llvm-mc `global_atomic_add_x2 v[0:1], v2, v[4:5], s[0:1] sc0 sc1` = 0xdf898000.
        let x2 = lines(&l, "global_atomic_add_x2");
        assert!(x2.len() == 1 && x2[0].1 == 0xdf89_8000, "{l}");
        let x1 = lines(&l, "global_atomic_add ");
        assert_eq!(x1.len(), 2, "{l}");
        let bits: Vec<(u32, u32)> = x1.iter().map(|(_, w, _)| (w >> 16 & 1, w >> 25 & 1)).collect();
        assert!(bits.contains(&(1, 0)) && bits.contains(&(0, 1)), "{l}");
    }

    #[test]
    fn l2_touch_loads_stay_in_flight_until_the_drain() {
        let body = "    %w : i64 = arith.zext ( %n )\n    \
                    %a : vec<i32x4> = tile.l2_touch_load ( %tid, %w )\n    %b : vec<i32x4> = tile.l2_touch_load ( %off, %w )\n    \
                    %c : vec<i32x4> = tile.l2_touch_load ( %c64, %w )\n    tile.l2_touch_drain ( %a, %b, %c )\n    %r : i32 = arith.add ( %tid, %n )\n";
        let l = listing(&kernel("    %n : i32\n", body));
        let loads = lines(&l, "global_load_dwordx4");
        assert_eq!(loads.len(), 3, "{l}");
        // Three different destinations (GLOBAL word 1, vdst bits 31:24).
        let mut dst: Vec<u32> = loads.iter().map(|(_, _, w1)| w1 >> 24).collect();
        dst.sort_unstable();
        dst.dedup();
        assert_eq!(dst.len(), 3, "{l}");
        // No wait between them, then vmcnt(0), and no KEEP_ALIVE left.
        let seq: Vec<&str> = l.lines().map(str::trim_start).filter(|x| x.starts_with("global_load_dwordx4") || x.starts_with("s_waitcnt")).collect();
        let first = seq.iter().position(|x| x.starts_with("global_load")).expect("a load");
        assert!(seq[first..first + 3].iter().all(|x| x.starts_with("global_load")), "{seq:?}");
        assert!(seq.get(first + 3).is_some_and(|x| x.ends_with("BF8C0F70")), "vmcnt(0): {seq:?}");
        assert!(!l.contains("KEEP_ALIVE"), "{l}");
    }

    #[test]
    fn the_preload_entry_is_comgrs() {
        // COMGR's first 256 bytes of a kernel preloading 14 dwords (the toy chain at kernarg_preload=8).
        let code = super::super::preload_entry(14).expect("an entry");
        let words: Vec<u32> = code.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(words.len(), 64);
        assert_eq!(
            words[..8],
            [0xc006_0080, 0x0000_0000, 0xc00e_0100, 0x0000_0008, 0xc00a_0300, 0x0000_0028, 0xbf8c_c07f, 0xbf82_0038]
        );
        assert!(words[8..].iter().all(|&w| w == 0xbf80_0000));
        // One dword: s_load_dword s2 then the wait and the branch.
        let one: Vec<u32> = super::super::preload_entry(1).unwrap().chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(one[..4], [0xc002_0080, 0x0000_0000, 0xbf8c_c07f, 0xbf82_003c]);
    }

    /// The 64-byte kernel descriptor of a single-kernel code object (the
    /// `.rodata` section).
    fn descriptor(hsaco: &[u8]) -> Vec<u8> {
        let u16_at = |o: usize| u16::from_le_bytes(hsaco[o..o + 2].try_into().unwrap()) as usize;
        let u64_at = |o: usize| u64::from_le_bytes(hsaco[o..o + 8].try_into().unwrap()) as usize;
        let (shoff, shentsize, shnum, shstrndx) = (u64_at(0x28), u16_at(0x3a), u16_at(0x3c), u16_at(0x3e));
        let sec = |i: usize| shoff + i * shentsize;
        let strtab = u64_at(sec(shstrndx) + 0x18);
        for i in 0..shnum {
            let name_off = u32::from_le_bytes(hsaco[sec(i)..sec(i) + 4].try_into().unwrap()) as usize;
            let name: Vec<u8> = hsaco[strtab + name_off..].iter().take_while(|&&b| b != 0).copied().collect();
            if name == b".rodata" {
                let off = u64_at(sec(i) + 0x18);
                return hsaco[off..off + 64].to_vec();
            }
        }
        panic!("no .rodata")
    }

    #[test]
    fn preloaded_arguments_live_in_user_sgprs() {
        // As the toy chain's first eight parameters: i64, i32, then i64s; the first seven fit in 14 dwords.
        let mut params = String::from("    %p0 : i64 { inreg = b:true }\n    %p1 : i32 { inreg = b:true }\n");
        for k in 2..8 {
            params += &format!("    %p{k} : i64 {{ inreg = b:true }}\n");
        }
        let mut body = String::from("    %a0 : i32 = arith.trunc ( %p0 )\n    %s1 : i32 = arith.add ( %a0, %p1 )\n");
        for k in 2..8 {
            body += &format!("    %a{k} : i32 = arith.trunc ( %p{k} )\n    %s{k} : i32 = arith.add ( %s{}, %a{k} )\n", k - 1);
        }
        body += "    %r : i32 = arith.add ( %s7, %tid )\n";
        let text = kernel(&params, &body).replace("    %Y : ptr<i32,global>\n", "").replace("  params {\n", "  params {\n    %Y : ptr<i32,global>\n");
        // %Y first: it is not inreg, so nothing is preloaded; with %Y last, seven are.
        let none = rir::compile_any_target(&text, Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(descriptor(&none.hsaco)[58..60], [0, 0]);
        let last = kernel(&params, &body).replace("    %Y : ptr<i32,global>\n", "").replace("  }\n  region @entry", "    %Y : ptr<i32,global>\n  }\n  region @entry");
        let k = rir::compile_any_target(&last, Target::Gfx950).unwrap_or_else(|e| panic!("{e}\n{last}"));
        let kd = descriptor(&k.hsaco);
        assert_eq!(u16::from_le_bytes([kd[58], kd[59]]), 14, "KERNARG_PRELOAD_SPEC_LENGTH");
        let rsrc2 = u32::from_le_bytes(kd[52..56].try_into().unwrap());
        assert_eq!(rsrc2 >> 1 & 0x1f, 16, "USER_SGPR_COUNT");
        let listing = k.listing();
        // p0 at s[2:3], p1 at s4, p2 at s[6:7] .. p6 at s[14:15]; p7 is loaded from the kernarg segment
        // (offset 56); the workgroup id follows the user SGPRs (s16).
        let offsets: Vec<i64> = listing
            .lines()
            .map(str::trim_start)
            .filter(|x| x.starts_with("s_load_dword"))
            .map(|x| x.split("imm:").nth(1).and_then(|r| r.split_whitespace().next()).and_then(|n| n.parse().ok()).unwrap_or(0))
            .collect();
        assert!(offsets.contains(&56) && offsets.iter().all(|&o| o >= 56), "kernarg loads at {offsets:?}: {listing}");
        let entry = &k.hsaco;
        let text_has = |w: u32| entry.windows(4).any(|x| x == w.to_le_bytes());
        assert!(text_has(0xc006_0080) && text_has(0xbf82_0038), "the preload entry");
    }

    /// `storewait`: a barrier waits on vmcnt for the youngest store in flight,
    /// not for the loads issued after it (and not at all with none in flight).
    #[test]
    fn storewait_barriers_wait_for_stores_only() {
        use crate::target::gfx9::waitcnt::insert_with;
        let v = |r: u16, n: u8| Opnd::P(Phys::V(r), n);
        let s = |r: u16, n: u8| Opnd::P(Phys::S(r), n);
        let load = |d: u16| MInst::new(Opc::GLoadB32, [v(d, 1)], [v(0, 2)]);
        let store = MInst::new(Opc::GStoreB32, [], [v(0, 2), v(9, 1)]);
        let run = |insts: Vec<MInst>, store_wait: bool| -> Vec<i64> {
            let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
            f.blocks = vec![MBlock { insts, loop_depth: 0 }];
            insert_with(&mut f, store_wait);
            let at = f.blocks[0].insts.iter().position(|i| i.op == Opc::SBarrier).expect("the barrier");
            f.blocks[0].insts[..at].iter().rev().take_while(|i| i.op == Opc::SWaitcnt).map(|i| i.imm).collect()
        };
        let _ = s;
        let body = vec![load(1), store.clone(), load(2), load(3), MInst::new(Opc::SBarrier, [], [])];
        let vm = |imm: i64| imm & 0xf | (imm >> 14 & 3) << 4;
        assert_eq!(run(body.clone(), false).iter().map(|&i| vm(i)).collect::<Vec<_>>(), vec![0]);
        // Two loads after the store: vmcnt(2) retires the store.
        assert_eq!(run(body, true).iter().map(|&i| vm(i)).collect::<Vec<_>>(), vec![2]);
        // Loads only: no vmcnt wait at all.
        let loads = vec![load(1), load(2), MInst::new(Opc::SBarrier, [], [])];
        assert!(run(loads.clone(), true).iter().all(|&i| vm(i) == 63), "{:?}", run(loads, true));
    }

    /// `wgfence`: a workgroup release waits for LDS and for the stores and LDS
    /// DMA loads in flight, never for plain loads (weights issued ahead of it).
    #[test]
    fn workgroup_releases_wait_for_stores_and_dma_not_loads() {
        use crate::target::gfx9::gen_isa::Gfx9Op;
        use crate::target::gfx9::waitcnt::insert_with;
        let v = |r: u16, n: u8| Opnd::P(Phys::V(r), n);
        let s = |r: u16, n: u8| Opnd::P(Phys::S(r), n);
        let load = |d: u16| MInst::new(Opc::GLoadB32, [v(d, 1)], [v(0, 2)]);
        let store = MInst::new(Opc::GStoreB32, [], [v(0, 2), v(9, 1)]);
        let dma = MInst::new(Opc::Gfx9(Gfx9Op::GLoadLdsB32), [], [v(0, 2)]);
        let lds = MInst::new(Opc::DsStoreB32, [], [v(1, 1), v(9, 1)]);
        let release = MInst::new(Opc::Gfx9(Gfx9Op::WaitRelease), [], []);
        let run = |insts: Vec<MInst>| -> Vec<i64> {
            let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
            f.blocks = vec![MBlock { insts, loop_depth: 0 }];
            insert_with(&mut f, false);
            let at = f.blocks[0].insts.iter().position(|i| i.op == Opc::Gfx9(Gfx9Op::WaitRelease)).expect("the release");
            f.blocks[0].insts[..at].iter().rev().take_while(|i| i.op == Opc::SWaitcnt).map(|i| i.imm).collect()
        };
        let _ = s;
        let vm = |imm: i64| imm & 0xf | (imm >> 14 & 3) << 4;
        let lgkm = |imm: i64| imm >> 8 & 0xf;
        // A store, then two loads: vmcnt(2) retires the store, not the loads; the LDS write: lgkmcnt(0).
        let w = run(vec![load(1), store.clone(), load(2), lds.clone(), load(3), release.clone()]);
        assert_eq!(w.iter().map(|&i| (vm(i), lgkm(i))).collect::<Vec<_>>(), vec![(2, 0)]);
        // An LDS DMA load then a plain load: vmcnt(1).
        let w = run(vec![dma, load(2), release.clone()]);
        assert_eq!(w.iter().map(|&i| vm(i)).collect::<Vec<_>>(), vec![1]);
        // Plain loads only: no wait at all.
        assert!(run(vec![load(1), load(2), release]).is_empty());
    }

    // ---- rust track (decision 6): words checked against `llvm-mc -mcpu=gfx950 -show-encoding`.

    fn count(listing: &str, mnemonic: &str) -> usize {
        lines(listing, mnemonic).len()
    }

    #[test]
    fn exp_is_llvms_expansion() {
        let body = "    %x : f32 = arith.bitcast ( %n )\n    %e : f32 = math.exp ( %x )\n    %r : i32 = arith.bitcast ( %e )\n";
        let l = listing(&kernel("    %n : i32\n", body));
        for m in ["v_exp_f32", "v_rndne_f32", "v_ldexp_f32", "v_cvt_i32_f32"] {
            assert_eq!(count(&l, m), 1, "{m}: {l}");
        }
        assert_eq!(count(&l, "v_fma_f32"), 2, "{l}");
        assert_eq!(count(&l, "v_cndmask_b32"), 2, "{l}");
        // log2(e), its low part and both range limits are in the code (literals or moved constants).
        for k in [super::exp_consts::LOG2E, super::exp_consts::LOG2E_LO, super::exp_consts::UNDERFLOW, super::exp_consts::OVERFLOW] {
            assert!(l.to_lowercase().contains(&format!("{k:08x}")) || l.contains(&format!("{k:#x}")) || l.contains(&format!("{}", k)), "{k:#x}: {l}");
        }
    }

    #[test]
    fn scaled_fp8_converts_keep_old_and_select_the_half() {
        let body = "    %f : f32 = arith.bitcast ( %n )\n    \
                    %a : i32 = arith.cvt_scalef32_pk_fp8_f32 ( %tid, %f, %f, %f ) { hi = b:false }\n    \
                    %r : i32 = arith.cvt_scalef32_pk_bf8_f32 ( %a, %f, %f, %f ) { hi = b:true }\n";
        let l = listing(&kernel("    %n : i32\n", body));
        // VOP3 0x235 / 0x236; op_sel[3] (bit 14) only on the high-half form.
        let lo = lines(&l, "v_cvt_scalef32_pk_fp8_f32");
        let hi = lines(&l, "v_cvt_scalef32_pk_bf8_f32");
        assert!(lo.len() == 1 && lo[0].1 >> 16 & 0x3ff == 0x235 && lo[0].1 >> 14 & 1 == 0, "{l}");
        assert!(hi.len() == 1 && hi[0].1 >> 16 & 0x3ff == 0x236 && hi[0].1 >> 14 & 1 == 1, "{l}");
    }

    #[test]
    fn mx_mfma_takes_formats_and_scale_selects() {
        let body = "    %a : vec<i32x8> = arith.constant_vec (  ) { elem = s:\"i32\", fill = i:1, vec = i:8 }\n    \
                    %b : vec<i32x4> = arith.constant_vec (  ) { elem = s:\"i32\", fill = i:2, vec = i:4 }\n    \
                    %c : vec<f32x4> = arith.constant_vec (  ) { elem = s:\"f32\", fill = f:0.0, vec = i:4 }\n    \
                    %d : vec<f32x4> = tile.mfma_scale_mx ( %a, %b, %c, %n, %tid ) { fmt_a = i:0, fmt_b = i:4, opsel_a = i:1, opsel_b = i:2, shape = s:\"16x16x128\" }\n    \
                    %e : f32 = vector.extract ( %d ) { index = i:1 }\n    %r : i32 = arith.bitcast ( %e )\n";
        let l = listing(&kernel("    %n : i32\n", body));
        // llvm-mc: v_mfma_scale_f32_16x16x128_f8f6f4 ... op_sel:[1,0,0] op_sel_hi:[0,1,0] blgp:4 =
        //   ld_scale word 0 0xd3ac0800 (op_sel[0], bit 11), word 1 bit 28 (op_sel_hi[1]); MFMA op 0x2d, blgp 4.
        let m = lines(&l, "v_mfma_scale_f32_16x16x128_f8f6f4");
        assert_eq!(m.len(), 1, "{l}");
        let words: Vec<u32> = m[0].0.rsplit(": ").next().unwrap().split_whitespace().map(|h| u32::from_str_radix(h, 16).unwrap()).collect();
        assert_eq!(words.len(), 4, "{l}");
        assert_eq!(words[0] & 0xffff_ff00, 0xd3ac_0800, "{words:x?}");
        assert_eq!(words[1] >> 27 & 3, 2, "op_sel_hi {words:x?}");
        assert_eq!(words[2] >> 16 & 0x7f, 0x2d, "{words:x?}");
        assert_eq!(words[3] >> 29, 4, "blgp {words:x?}");
    }

    #[test]
    fn maximum3_takes_abs_modifiers() {
        let body = "    %x : f32 = arith.bitcast ( %n )\n    %y : f32 = arith.bitcast ( %tid )\n    \
                    %m : f32 = arith.maximum3_f32 ( %x, %y, %y ) { abs = i:7 }\n    %r : i32 = arith.bitcast ( %m )\n";
        let l = listing(&kernel("    %n : i32\n", body));
        let m = lines(&l, "v_maximum3_f32");
        // llvm-mc: v_maximum3_f32 v1, |v2|, |v3|, |v4| = 0xd2a90701 ...
        assert!(m.len() == 1 && m[0].1 >> 16 & 0x3ff == 0x2a9 && m[0].1 >> 8 & 7 == 7, "{l}");
    }

    #[test]
    fn masked_dpp_keeps_old_lanes() {
        let body = "    %a : i32 = tile.dpp_op ( %tid, %tid ) { bank_mask = i:10, bound_ctrl = b:false, ctrl = i:276, op = s:\"mov\", row_mask = i:15 }\n    \
                    %b : i32 = tile.dpp_op ( %a, %tid ) { bank_mask = i:5, bound_ctrl = b:false, ctrl = i:260, op = s:\"mov\", row_mask = i:15 }\n    \
                    %m0 : i32 = tile.dpp_op ( %b, %b, %b ) { bound_ctrl = b:true, ctrl = i:273, op = s:\"max_u32\" }\n    \
                    %r : i32 = tile.dpp_op ( %m0, %m0, %m0 ) { bank_mask = i:15, ctrl = i:322, op = s:\"max_u32\", row_mask = i:10 }\n";
        let l = listing(&kernel("", body));
        // llvm-mc: v_mov_b32_dpp v1, v2 row_shr:4 row_mask:0xf bank_mask:0xa = 0x7e0202fa 0xfa011402;
        // v_max_u32_dpp v1, v1, v1 row_shr:1 bound_ctrl:1 = 0x1e0202fa 0xff091101; row_bcast:15 row_mask:0xa = .. 0xaf014201.
        let movs = lines(&l, "v_mov_b32");
        let dpp: Vec<u32> = movs.iter().filter(|(_, w, _)| w & 0xff == 0xfa).map(|(_, _, d)| *d & 0xffff_ff00).collect();
        assert!(dpp.contains(&0xfa01_1400) && dpp.contains(&0xf501_0400), "{dpp:x?}\n{l}");
        let maxes: Vec<u32> = lines(&l, "v_max_u32").iter().filter(|(_, w, _)| w & 0xff == 0xfa && w >> 25 == 0xf).map(|(_, _, d)| d & 0xffff_ff00).collect();
        assert!(maxes.contains(&0xff09_1100) && maxes.contains(&0xaf01_4200), "{maxes:x?}\n{l}");
    }

    #[test]
    fn constant_pointers_load_through_smem() {
        // A global write first: a plain load after it may not use the scalar cache; an invariant one may.
        let body = "    tile.buffer_store_cp ( %tid, %rs, %off, %c0 ) { aux = i:16, dwords = i:1 }\n    \
                    %w : i64 = arith.zext ( %n )\n    %p : ptr<i32,constant> = tile.int_to_ptr ( %w )\n    \
                    %c4 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:4 }\n    \
                    %v : vec<i32x4> = memref.global_load_vN ( %p, %c4 ) { align = i:4, elem_type = s:\"i32\", vec = i:4 }\n    \
                    %a : i32 = vector.extract ( %v ) { index = i:3 }\n    %r : i32 = arith.add ( %a, %tid )\n";
        let l = listing(&kernel("    %n : i32\n", body));
        assert_eq!(count(&l, "s_load_dwordx4"), 1, "{l}");
        assert_eq!(count(&l, "global_load_dword"), 0, "{l}");
    }

    #[test]
    fn global_accesses_carry_sc0_sc1_nt() {
        let body = "    %w : i64 = arith.zext ( %n )\n    \
                    tile.global_store_cp ( %tid, %off, %w ) { aux = i:19, offset = i:8 }\n    \
                    %v : vec<i32x2> = tile.global_load_cp ( %off, %w ) { aux = i:17, dwords = i:2, offset = i:0 }\n    \
                    %r : i32 = vector.extract ( %v ) { index = i:1 }\n";
        let l = listing(&kernel("    %n : i32\n", body));
        // GLOBAL word 0: sc0 bit 16, nt bit 17, sc1 bit 25 (llvm-mc: global_store_dwordx2 v1, v[2:3], s[4:5] offset:8 sc0 nt sc1 = 0xde778008).
        // (The kernel's own result store has no cache bits.)
        let st: Vec<_> = lines(&l, "global_store_dword ").into_iter().filter(|x| x.1 & 0xfff == 8).collect();
        assert!(st.len() == 1 && st[0].1 >> 16 & 3 == 3 && st[0].1 >> 25 & 1 == 1, "{l}");
        let ld = lines(&l, "global_load_dwordx2");
        assert!(ld.len() == 1 && ld[0].1 >> 16 & 3 == 1 && ld[0].1 >> 25 & 1 == 1, "{l}");
    }

    #[test]
    fn f64_sqrt_is_rsq_and_goldschmidt() {
        let body = "    %x : f32 = arith.bitcast ( %n )\n    %d : f64 = arith.cvt_f32_to_f64 ( %x )\n    \
                    %s : f64 = math.sqrt ( %d )\n    %f : f32 = arith.cvt_f64_to_f32 ( %s )\n    %r : i32 = arith.bitcast ( %f )\n";
        let l = listing(&kernel("    %n : i32\n", body));
        assert_eq!(count(&l, "v_rsq_f64"), 1, "{l}");
        assert_eq!(count(&l, "v_fma_f64"), 7, "{l}");
        assert_eq!(count(&l, "v_ldexp_f64"), 2, "{l}");
        assert_eq!(count(&l, "v_cvt_f64_f32"), 1, "{l}");
        assert_eq!(count(&l, "v_cvt_f32_f64"), 1, "{l}");
    }

    #[test]
    fn small_device_ops_encode() {
        let body = "    %b : i32 = tile.bcnt ( %tid )\n    %f : i32 = tile.s_ff1 ( %n )\n    %t : i64 = tile.memtime (  )\n    \
                    tile.s_nop (  ) { n = i:3 }\n    tile.compiler_fence (  )\n    \
                    tile.ds_write_b32 ( %tid, %b )\n    tile.ds_or_b32 ( %tid, %f )\n    %q : i32 = tile.ds_read_b32 ( %tid )\n    \
                    %d : i32 = tile.mov_dpp ( %q ) { bound_ctrl = b:false, row_shl = i:4 }\n    \
                    %tl : i32 = arith.trunc ( %t )\n    %s : i32 = arith.add ( %d, %tl )\n    %r : i32 = arith.add ( %s, %f )\n";
        let l = listing(&kernel("    %n : i32\n", body));
        // llvm-mc: v_bcnt_u32_b32 (VOP3 0x28b), s_ff1_i32_b32 s1, s2 = 0xbe811002, s_memtime = SMEM 0x24,
        // ds_or_b32 0xd8140000, ds_write_b32 0xd81a0000, ds_read_b32 0xd86c0000.
        assert!(lines(&l, "v_bcnt_u32_b32").iter().any(|(_, w, _)| w >> 16 & 0x3ff == 0x28b), "{l}");
        assert!(lines(&l, "s_ff1_i32_b32").iter().any(|(_, w, _)| w & 0xffff_ff00 == 0xbe80_1000 | (w & 0x7f_0000)), "{l}");
        assert!(lines(&l, "s_memtime").iter().any(|(_, w, _)| w >> 18 & 0xff == 0x24), "{l}");
        assert!(lines(&l, "ds_or_b32").iter().any(|(_, w, _)| w & 0xfffe_0000 == 0xd814_0000), "{l}");
        assert!(lines(&l, "ds_write_b32").iter().any(|(_, w, _)| w & 0xfffe_0000 == 0xd81a_0000), "{l}");
        assert!(lines(&l, "ds_read_b32").iter().any(|(_, w, _)| w & 0xfffe_0000 == 0xd86c_0000), "{l}");
        assert!(lines(&l, "s_nop").iter().any(|(_, w, _)| *w == 0xbf80_0003), "{l}");
        // row_shl:4 without bound_ctrl: the tied form (DPP word 0xff010400 | src).
        assert!(lines(&l, "v_mov_b32").iter().any(|(_, w, d)| w & 0xff == 0xfa && d & 0xffff_ff00 == 0xff01_0400), "{l}");
    }

    #[test]
    fn router_key_levers_save_the_xor_and_the_add() {
        // key = (select(x >= 0, x ^ INT_MIN, x ^ -1) & -512) | (511 - (8 tid + i)), two i
        let body = concat!(
                    "    %v : i32 = arith.xor ( %n, %tid )
",
                    "    %c16 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:16 }
",
                    "    %x : i32 = arith.shl ( %v, %c16 )
",
                    "    %z : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }
",
                    "    %pos : i1 = arith.cmp ( %x, %z ) { pred = s:\"ge\" }
",
                    "    %smin : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:-2147483648 }
",
                    "    %m1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:-1 }
",
                    "    %xa : i32 = arith.xor ( %x, %smin )
",
                    "    %xb : i32 = arith.xor ( %x, %m1 )
",
                    "    %mono : i32 = arith.select ( %pos, %xa, %xb )
",
                    "    %hi : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:-512 }
",
                    "    %k : i32 = arith.and ( %mono, %hi )
",
                    "    %c3 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:3 }
",
                    "    %c5 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:5 }
",
                    "    %c511 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:511 }
",
                    "    %t8 : i32 = arith.mul ( %tid, %c8 )
",
                    "    %i3 : i32 = arith.add ( %t8, %c3 )
",
                    "    %i5 : i32 = arith.add ( %t8, %c5 )
",
                    "    %lo3 : i32 = arith.sub ( %c511, %i3 )
",
                    "    %lo5 : i32 = arith.sub ( %c511, %i5 )
",
                    "    %k3 : i32 = arith.or ( %k, %lo3 )
",
                    "    %r : i32 = arith.xor ( %k3, %lo5 )
",
        );
        let text = kernel("    %n : i32
", body);
        let plain = listing(&text);
        let lev = rir::with_levers(&["selxor", "subadd", "bitop3"], || listing(&text));
        scopes_match_kernel_levers(&text, &["selxor", "subadd", "bitop3"]);
        let count = |l: &str, m: &str| lines(l, m).len();
        let valu = |l: &str| l.lines().filter(|x| x.trim_start().starts_with("v_")).count();
        assert!(count(&plain, "v_xor_b32") >= 2, "{plain}");
        assert!(valu(&lev) + 2 <= valu(&plain), "{plain}
{lev}");
        // `selxor` moves the -512 mask into the selected constants: 0xfffffe00 stays (as INT_MIN | 0 and -512).
        assert!(lev.contains("0xfffffe00"), "{lev}");
    }


    #[test]
    fn guarded_loads_issue_in_the_order_they_are_read() {
        // A 0/1-trip loop of loads read after it in order (the fused MoE's guarded weight loads): `loadorder`
        // keeps their program order (their in-block readers are the loop's yield copies).
        let body = concat!(
                    "    %c1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }
",
                    "    %ok : i1 = arith.cmp ( %n, %c0 ) { pred = s:\"gt\" }
",
                    "    %trip : i32 = arith.select ( %ok, %c1, %c0 )
",
                    "    %rs2 : vec<i32x4> = tile.buffer_rsrc ( %Y, %c8 )
",
                    "    %k0 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }
",
                    "    %o0 : i32 = arith.add ( %off, %k0 )
",
                    "    %k1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1024 }
",
                    "    %o1 : i32 = arith.add ( %off, %k1 )
",
                    "    %k2 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:2048 }
",
                    "    %o2 : i32 = arith.add ( %off, %k2 )
",
                    "    %k3 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:3072 }
",
                    "    %o3 : i32 = arith.add ( %off, %k3 )
",
                    "    %k4 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:4096 }
",
                    "    %o4 : i32 = arith.add ( %off, %k4 )
",
                    "    %k5 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:5120 }
",
                    "    %o5 : i32 = arith.add ( %off, %k5 )
",
                    "    %k6 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:6144 }
",
                    "    %o6 : i32 = arith.add ( %off, %k6 )
",
                    "    %k7 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:7168 }
",
                    "    %o7 : i32 = arith.add ( %off, %k7 )
",
                    "    %k8 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:8192 }
",
                    "    %o8 : i32 = arith.add ( %off, %k8 )
",
                    "    %k9 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:9216 }
",
                    "    %o9 : i32 = arith.add ( %off, %k9 )
",
                    "    %k10 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:10240 }
",
                    "    %o10 : i32 = arith.add ( %off, %k10 )
",
                    "    %k11 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:11264 }
",
                    "    %o11 : i32 = arith.add ( %off, %k11 )
",
                    "    %k12 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:12288 }
",
                    "    %o12 : i32 = arith.add ( %off, %k12 )
",
                    "    %k13 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:13312 }
",
                    "    %o13 : i32 = arith.add ( %off, %k13 )
",
                    "    %k14 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:14336 }
",
                    "    %o14 : i32 = arith.add ( %off, %k14 )
",
                    "    %k15 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:15360 }
",
                    "    %o15 : i32 = arith.add ( %off, %k15 )
",
                    "    %k16 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:16384 }
",
                    "    %o16 : i32 = arith.add ( %off, %k16 )
",
                    "    %k17 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:17408 }
",
                    "    %o17 : i32 = arith.add ( %off, %k17 )
",
                    "    %k18 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:18432 }
",
                    "    %o18 : i32 = arith.add ( %off, %k18 )
",
                    "    %k19 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:19456 }
",
                    "    %o19 : i32 = arith.add ( %off, %k19 )
",
                    "    %k20 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:20480 }
",
                    "    %o20 : i32 = arith.add ( %off, %k20 )
",
                    "    %k21 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:21504 }
",
                    "    %o21 : i32 = arith.add ( %off, %k21 )
",
                    "    %k22 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:22528 }
",
                    "    %o22 : i32 = arith.add ( %off, %k22 )
",
                    "    %k23 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:23552 }
",
                    "    %o23 : i32 = arith.add ( %off, %k23 )
",
                    "    %k24 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:24576 }
",
                    "    %o24 : i32 = arith.add ( %off, %k24 )
",
                    "    %k25 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:25600 }
",
                    "    %o25 : i32 = arith.add ( %off, %k25 )
",
                    "    %k26 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:26624 }
",
                    "    %o26 : i32 = arith.add ( %off, %k26 )
",
                    "    %k27 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:27648 }
",
                    "    %o27 : i32 = arith.add ( %off, %k27 )
",
                    "    %k28 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:28672 }
",
                    "    %o28 : i32 = arith.add ( %off, %k28 )
",
                    "    %k29 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:29696 }
",
                    "    %o29 : i32 = arith.add ( %off, %k29 )
",
                    "    %z0 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %z1 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %z2 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %z3 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %z6 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %z7 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %z8 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %z9 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %z12 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %z13 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %z14 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %z15 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %z18 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %z19 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %z20 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %z21 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %z24 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %z25 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %z26 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %z27 : vec<i32x4> = vector.pack ( %c0, %c0, %c0, %c0 ) { elem = s:\"i32\", vec = i:4 }
",
                    "    %l0 : vec<i32x4>, %l1 : vec<i32x4>, %l2 : vec<i32x4>, %l3 : vec<i32x4>, %l4 : i32, %l5 : i32, %l6 : vec<i32x4>, %l7 : vec<i32x4>, %l8 : vec<i32x4>, %l9 : vec<i32x4>, %l10 : i32, %l11 : i32, %l12 : vec<i32x4>, %l13 : vec<i32x4>, %l14 : vec<i32x4>, %l15 : vec<i32x4>, %l16 : i32, %l17 : i32, %l18 : vec<i32x4>, %l19 : vec<i32x4>, %l20 : vec<i32x4>, %l21 : vec<i32x4>, %l22 : i32, %l23 : i32, %l24 : vec<i32x4>, %l25 : vec<i32x4>, %l26 : vec<i32x4>, %l27 : vec<i32x4>, %l28 : i32, %l29 : i32 = scf.for ( %c0, %trip, %c1, %z0, %z1, %z2, %z3, %c0, %c0, %z6, %z7, %z8, %z9, %c0, %c0, %z12, %z13, %z14, %z15, %c0, %c0, %z18, %z19, %z20, %z21, %c0, %c0, %z24, %z25, %z26, %z27, %c0, %c0 ) { iter_args = l:[ { name = s:\"%a0\", type = s:\"vec<i32x4>\" }, { name = s:\"%a1\", type = s:\"vec<i32x4>\" }, { name = s:\"%a2\", type = s:\"vec<i32x4>\" }, { name = s:\"%a3\", type = s:\"vec<i32x4>\" }, { name = s:\"%a4\", type = s:\"i32\" }, { name = s:\"%a5\", type = s:\"i32\" }, { name = s:\"%a6\", type = s:\"vec<i32x4>\" }, { name = s:\"%a7\", type = s:\"vec<i32x4>\" }, { name = s:\"%a8\", type = s:\"vec<i32x4>\" }, { name = s:\"%a9\", type = s:\"vec<i32x4>\" }, { name = s:\"%a10\", type = s:\"i32\" }, { name = s:\"%a11\", type = s:\"i32\" }, { name = s:\"%a12\", type = s:\"vec<i32x4>\" }, { name = s:\"%a13\", type = s:\"vec<i32x4>\" }, { name = s:\"%a14\", type = s:\"vec<i32x4>\" }, { name = s:\"%a15\", type = s:\"vec<i32x4>\" }, { name = s:\"%a16\", type = s:\"i32\" }, { name = s:\"%a17\", type = s:\"i32\" }, { name = s:\"%a18\", type = s:\"vec<i32x4>\" }, { name = s:\"%a19\", type = s:\"vec<i32x4>\" }, { name = s:\"%a20\", type = s:\"vec<i32x4>\" }, { name = s:\"%a21\", type = s:\"vec<i32x4>\" }, { name = s:\"%a22\", type = s:\"i32\" }, { name = s:\"%a23\", type = s:\"i32\" }, { name = s:\"%a24\", type = s:\"vec<i32x4>\" }, { name = s:\"%a25\", type = s:\"vec<i32x4>\" }, { name = s:\"%a26\", type = s:\"vec<i32x4>\" }, { name = s:\"%a27\", type = s:\"vec<i32x4>\" }, { name = s:\"%a28\", type = s:\"i32\" }, { name = s:\"%a29\", type = s:\"i32\" } ], iv = s:\"%it\", iv_type = s:\"i32\", num_iter_args = i:30, unroll = b:false }
",
                    "      region @body {
",
                    "        %x0 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o0, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x1 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o1, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x2 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o2, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x3 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o3, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x4 : i32 = tile.buffer_load_cp ( %rs2, %o4, %c0 ) { aux = i:18, dwords = i:1 }
",
                    "        %x5 : i32 = tile.buffer_load_cp ( %rs2, %o5, %c0 ) { aux = i:18, dwords = i:1 }
",
                    "        %x6 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o6, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x7 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o7, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x8 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o8, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x9 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o9, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x10 : i32 = tile.buffer_load_cp ( %rs2, %o10, %c0 ) { aux = i:18, dwords = i:1 }
",
                    "        %x11 : i32 = tile.buffer_load_cp ( %rs2, %o11, %c0 ) { aux = i:18, dwords = i:1 }
",
                    "        %x12 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o12, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x13 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o13, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x14 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o14, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x15 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o15, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x16 : i32 = tile.buffer_load_cp ( %rs2, %o16, %c0 ) { aux = i:18, dwords = i:1 }
",
                    "        %x17 : i32 = tile.buffer_load_cp ( %rs2, %o17, %c0 ) { aux = i:18, dwords = i:1 }
",
                    "        %x18 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o18, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x19 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o19, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x20 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o20, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x21 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o21, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x22 : i32 = tile.buffer_load_cp ( %rs2, %o22, %c0 ) { aux = i:18, dwords = i:1 }
",
                    "        %x23 : i32 = tile.buffer_load_cp ( %rs2, %o23, %c0 ) { aux = i:18, dwords = i:1 }
",
                    "        %x24 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o24, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x25 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o25, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x26 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o26, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x27 : vec<i32x4> = tile.buffer_load_cp ( %rs, %o27, %c0 ) { aux = i:18, dwords = i:4 }
",
                    "        %x28 : i32 = tile.buffer_load_cp ( %rs2, %o28, %c0 ) { aux = i:18, dwords = i:1 }
",
                    "        %x29 : i32 = tile.buffer_load_cp ( %rs2, %o29, %c0 ) { aux = i:18, dwords = i:1 }
",
                    "        scf.yield ( %x0, %x1, %x2, %x3, %x4, %x5, %x6, %x7, %x8, %x9, %x10, %x11, %x12, %x13, %x14, %x15, %x16, %x17, %x18, %x19, %x20, %x21, %x22, %x23, %x24, %x25, %x26, %x27, %x28, %x29 ) { num = i:30 }
",
                    "      }
",
                    "    %v0 : i32 = vector.extract ( %l0 ) { index = i:1 }
",
                    "    %e0 : i32 = arith.add ( %v0, %c0 )
",
                    "    %v1 : i32 = vector.extract ( %l1 ) { index = i:1 }
",
                    "    %e1 : i32 = arith.add ( %v1, %e0 )
",
                    "    %v2 : i32 = vector.extract ( %l2 ) { index = i:1 }
",
                    "    %e2 : i32 = arith.add ( %v2, %e1 )
",
                    "    %v3 : i32 = vector.extract ( %l3 ) { index = i:1 }
",
                    "    %e3 : i32 = arith.add ( %v3, %e2 )
",
                    "    %e4 : i32 = arith.add ( %l4, %e3 )
",
                    "    %e5 : i32 = arith.add ( %l5, %e4 )
",
                    "    %v6 : i32 = vector.extract ( %l6 ) { index = i:1 }
",
                    "    %e6 : i32 = arith.add ( %v6, %e5 )
",
                    "    %v7 : i32 = vector.extract ( %l7 ) { index = i:1 }
",
                    "    %e7 : i32 = arith.add ( %v7, %e6 )
",
                    "    %v8 : i32 = vector.extract ( %l8 ) { index = i:1 }
",
                    "    %e8 : i32 = arith.add ( %v8, %e7 )
",
                    "    %v9 : i32 = vector.extract ( %l9 ) { index = i:1 }
",
                    "    %e9 : i32 = arith.add ( %v9, %e8 )
",
                    "    %e10 : i32 = arith.add ( %l10, %e9 )
",
                    "    %e11 : i32 = arith.add ( %l11, %e10 )
",
                    "    %v12 : i32 = vector.extract ( %l12 ) { index = i:1 }
",
                    "    %e12 : i32 = arith.add ( %v12, %e11 )
",
                    "    %v13 : i32 = vector.extract ( %l13 ) { index = i:1 }
",
                    "    %e13 : i32 = arith.add ( %v13, %e12 )
",
                    "    %v14 : i32 = vector.extract ( %l14 ) { index = i:1 }
",
                    "    %e14 : i32 = arith.add ( %v14, %e13 )
",
                    "    %v15 : i32 = vector.extract ( %l15 ) { index = i:1 }
",
                    "    %e15 : i32 = arith.add ( %v15, %e14 )
",
                    "    %e16 : i32 = arith.add ( %l16, %e15 )
",
                    "    %e17 : i32 = arith.add ( %l17, %e16 )
",
                    "    %v18 : i32 = vector.extract ( %l18 ) { index = i:1 }
",
                    "    %e18 : i32 = arith.add ( %v18, %e17 )
",
                    "    %v19 : i32 = vector.extract ( %l19 ) { index = i:1 }
",
                    "    %e19 : i32 = arith.add ( %v19, %e18 )
",
                    "    %v20 : i32 = vector.extract ( %l20 ) { index = i:1 }
",
                    "    %e20 : i32 = arith.add ( %v20, %e19 )
",
                    "    %v21 : i32 = vector.extract ( %l21 ) { index = i:1 }
",
                    "    %e21 : i32 = arith.add ( %v21, %e20 )
",
                    "    %e22 : i32 = arith.add ( %l22, %e21 )
",
                    "    %e23 : i32 = arith.add ( %l23, %e22 )
",
                    "    %v24 : i32 = vector.extract ( %l24 ) { index = i:1 }
",
                    "    %e24 : i32 = arith.add ( %v24, %e23 )
",
                    "    %v25 : i32 = vector.extract ( %l25 ) { index = i:1 }
",
                    "    %e25 : i32 = arith.add ( %v25, %e24 )
",
                    "    %v26 : i32 = vector.extract ( %l26 ) { index = i:1 }
",
                    "    %e26 : i32 = arith.add ( %v26, %e25 )
",
                    "    %v27 : i32 = vector.extract ( %l27 ) { index = i:1 }
",
                    "    %e27 : i32 = arith.add ( %v27, %e26 )
",
                    "    %e28 : i32 = arith.add ( %l28, %e27 )
",
                    "    %e29 : i32 = arith.add ( %l29, %e28 )
",
                    "    %r : i32 = arith.add ( %e29, %c1 )
",
        );
        let text = kernel("    %n : i32
", body);
        // Each load's address register, in issue order: v1 (no offset) and the 29 offset sums are
        // allocated in program order, so the issue order shows as the order of the address operands.
        let order = |l: &str| -> Vec<String> {
            l.lines()
                .map(str::trim_start)
                .filter(|x| x.starts_with("buffer_load"))
                .map(|x| x.split_whitespace().nth(2).unwrap_or("").trim_end_matches(',').to_string())
                .collect()
        };
        let plain = order(&listing(&text));
        let lev = order(&rir::with_levers(&["loadorder"], || listing(&text)));
        scopes_match_kernel_levers(&text, &["loadorder"]);
        assert_eq!(plain.len(), 30, "{plain:?}");
        assert_eq!(lev, plain, "loadorder reordered the guarded loads");
    }


    #[test]
    fn dsrebase_shares_one_base_past_the_offset_field() {
        // Two LDS loads at 68000 and 104000 bytes past 4 tid: past the 16-bit offset field (the second's
        // field part has bit 15 set).
        let body = concat!(
                    "    %lds : smem<f32, [30000]> = tile.smem_alloc (  )
",
                    "    %k1 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:17000 }
",
                    "    %k2 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:26000 }
",
                    "    %i1 : i32 = arith.add ( %tid, %k1 )
",
                    "    %i2 : i32 = arith.add ( %tid, %k2 )
",
                    "    %y1 : vec<f32x1> = tile.smem_load_vN_f32 ( %lds, %i1 ) { elem_type = s:\"f32\", rank = i:1, vec = i:1 }
",
                    "    %y2 : vec<f32x1> = tile.smem_load_vN_f32 ( %lds, %i2 ) { elem_type = s:\"f32\", rank = i:1, vec = i:1 }
",
                    "    %e1 : f32 = vector.extract ( %y1 ) { index = i:0 }
",
                    "    %e2 : f32 = vector.extract ( %y2 ) { index = i:0 }
",
                    "    %s : f32 = arith.fadd ( %e1, %e2 )
",
                    "    %r : i32 = arith.bitcast ( %s )
",
        );
        let text = kernel("", body);
        let plain = listing(&text);
        let lev = rir::with_levers(&["dsrebase"], || listing(&text));
        scopes_match_kernel_levers(&text, &["dsrebase"]);
        let adds = |l: &str, k: &str| l.lines().filter(|x| x.trim_start().starts_with("v_add") && x.contains(k)).count();
        // Plain: each load adds its whole offset (0x109a0 = 68000, 0x19640 = 104000) and reads at 0.
        assert_eq!((adds(&plain, "0x109a0"), adds(&plain, "0x19640")), (1, 1), "{plain}");
        // dsrebase: one add of 0x10000, then the offset fields 2464 and 38464 (0x10000 + each = the address).
        assert_eq!((adds(&lev, "0x10000"), adds(&lev, "0x109a0"), adds(&lev, "0x19640")), (1, 0, 0), "{lev}");
        let offs: Vec<u32> = lines(&lev, "ds_read_b32").iter().map(|(_, w, _)| w & 0xffff).collect();
        assert_eq!(offs, [2464, 38464], "{lev}");
    }

    #[test]
    fn scalesep_reads_the_scales_where_they_are() {
        // Two MX MFMAs with their scales in VGPRs: no copy into a scale pair, and each ld_scale word names them.
        let body = "    %a : vec<i32x8> = arith.constant_vec (  ) { elem = s:\"i32\", fill = i:1, vec = i:8 }
                        %b : vec<i32x4> = arith.constant_vec (  ) { elem = s:\"i32\", fill = i:2, vec = i:4 }
                        %c : vec<f32x4> = arith.constant_vec (  ) { elem = s:\"f32\", fill = f:0.0, vec = i:4 }
                        %sb : i32 = arith.xor ( %tid, %n )
                        %d : vec<f32x4> = tile.mfma_scale_mx ( %a, %b, %c, %tid, %sb ) { fmt_a = i:0, fmt_b = i:4, opsel_a = i:0, opsel_b = i:0, shape = s:\"16x16x128\" }
                        %d2 : vec<f32x4> = tile.mfma_scale_mx ( %a, %b, %d, %sb, %tid ) { fmt_a = i:0, fmt_b = i:4, opsel_a = i:0, opsel_b = i:0, shape = s:\"16x16x128\" }
                        %e : f32 = vector.extract ( %d2 ) { index = i:1 }
    %r : i32 = arith.bitcast ( %e )
";
        let text = kernel("    %n : i32
", body);
        let plain = listing(&text);
        let lev = rir::with_levers(&["scalesep"], || listing(&text));
        scopes_match_kernel_levers(&text, &["scalesep"]);
        let m = lines(&lev, "v_mfma_scale_f32_16x16x128_f8f6f4");
        assert_eq!(m.len(), 2, "{lev}");
        // ld_scale word 1: src0 bits 8:0, src1 bits 17:9 (VGPRs 256 + n); the two MFMAs swap the scales.
        let srcs: Vec<(u32, u32)> = m
            .iter()
            .map(|(l, _, _)| {
                let w: Vec<u32> = l.rsplit(": ").next().unwrap().split_whitespace().map(|h| u32::from_str_radix(h, 16).unwrap()).collect();
                (w[1] & 0x1ff, w[1] >> 9 & 0x1ff)
            })
            .collect();
        assert_eq!((srcs[0].0, srcs[0].1), (srcs[1].1, srcs[1].0), "{srcs:?}
{lev}");
        assert!(srcs[0].0 >= 256 && srcs[0].1 >= 256 && srcs[0].0 != srcs[0].1, "{srcs:?}");
        // The plain build copies them into a fresh pair per MFMA: more moves.
        let moves = |l: &str| lines(l, "v_mov_b32").len();
        assert!(moves(&lev) < moves(&plain), "{plain}
{lev}");
    }


    #[test]
    fn a_bounded_poll_template_assembles_as_written() {
        // P4's counter poll (`p4_sync.poll_counter`): the template's instructions in order, its loop kept.
        let body = concat!(
                    "    %cap : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1000 }
",
                    "    %want : i32 = arith.add ( %n, %tid )
",
                    "    %base : i64 = tile.global_ptr_to_i64 ( %Y )
",
                    "    %seen : i32, %mask : i64, %it : i32 = tile.inline_asm ( %off, %base, %want, %cap ) { constraints = s:\"=&v,=&s,=&s,v,s,v,s,~{vcc},~{scc}\", convergent = b:true, sideeffect = b:true, template = s:\"s_mov_b32 $2, 0\\n${:private}p4c${:uid}_top:\\nglobal_load_dword $0, $3, $4 sc1\\ns_waitcnt vmcnt(0)\\nv_cmp_lt_u32_e64 $1, $0, $5\\ns_and_b64 $1, $1, exec\\ns_cbranch_scc0 ${:private}p4c${:uid}_done\\ns_add_u32 $2, $2, 1\\ns_cmp_ge_u32 $2, $6\\ns_cbranch_scc1 ${:private}p4c${:uid}_done\\ns_nop 0\\ns_branch ${:private}p4c${:uid}_top\\n${:private}p4c${:uid}_done:\" }\n",
                    "    %r : i32 = arith.add ( %seen, %it )
",
        );
        let l = listing(&kernel("    %n : i32
", body));
        let seq: Vec<&str> = l.lines().map(str::trim_start).filter(|x| !x.starts_with("bb")).filter_map(|x| x.split_whitespace().next()).collect();
        let at = seq.iter().position(|m| *m == "global_load_dword").unwrap_or_else(|| panic!("{l}"));
        assert_eq!(
            &seq[at..at + 10],
            ["global_load_dword", "s_waitcnt", "v_cmp_lt_u32", "s_and_b64", "s_cbranch_scc0", "s_add_u32", "s_cmp_ge_u32", "s_cbranch_scc1", "s_nop", "s_branch"],
            "{l}"
        );
        // The counter's zeroing is outside the loop (scheduled freely in its block).
        assert!(seq[..at].contains(&"s_mov_b32"), "{l}");
        // sc1 on the load; the loop branch goes back to the load; the exits to the instruction after the loop.
        let words = lines(&l, "global_load_dword");
        // llvm-objdump of COMGR's object: global_load_dword v4, v1, s[20:21] sc1 = 0xde508000 (sc1 bit 25, op 0x14),
        // and the loop's s_branch back over its 12 words = 0xbf82fff4.
        assert!(words[0].1 >> 25 & 1 == 1 && words[0].1 >> 18 & 0x7f == 0x14, "{l}");
        assert!(lines(&l, "s_branch").iter().any(|(_, w, _)| *w == 0xbf82_fff4), "{l}");
        // The exits (s_cbranch_scc0 +5, s_cbranch_scc1 +2) and the template's vmcnt(0), word for word as COMGR's.
        assert!(lines(&l, "s_cbranch_scc0").iter().any(|(_, w, _)| *w == 0xbf84_0005), "{l}");
        assert!(lines(&l, "s_cbranch_scc1").iter().any(|(_, w, _)| *w == 0xbf85_0002), "{l}");
        assert!(lines(&l, "s_waitcnt").iter().any(|(_, w, _)| *w == 0xbf8c_0f70), "{l}");
    }

    #[test]
    fn i64_shifts_right_by_constants() {
        // x >> 8 of a lane's 64-bit value (unknown bits), stored whole: low = (lo >> 8) | (hi << 24), high = hi >> 8.
        let body = "    %w : i64 = arith.zext ( %tid )
    %x : i64 = arith.xor ( %w, %p )
                        %c8 : i64 = arith.constant (  ) { ity = s:\"i64\", value = i:8 }
                        %s : i64 = arith.lshr ( %x, %c8 )
                        memref.global_store_typed ( %Z, %tid, %s ) { align = i:8, elem_type = s:\"i64\" }
                        %r : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:0 }
";
        let l = listing(&kernel("    %p : i64
    %Z : ptr<i64,global>
", body));
        let has = |m: &str, amount: &str| l.lines().any(|x| x.trim_start().starts_with(m) && x.contains(&format!(" {amount}, ")));
        // The carry is ORed into the low word (fused or not); dropping it leaves the shift dead.
        let ors = lines(&l, "v_or_b32").len() + lines(&l, "v_or3_b32").len();
        assert!(has("v_lshrrev_b32", "8") && (has("v_lshl_or_b32", "24") || has("v_lshlrev_b32", "24") && ors > 0), "{l}");
    }


    #[test]
    fn bufoffany_moves_the_constant_into_the_offset_field() {
        // voffset = select(tid < n, n - tid, 2^30) + 64: the rest's sign is unknown (n - tid may be negative).
        let body = concat!(
                    "    %k30 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1073741824 }
",
                    "    %lt : i1 = arith.cmp ( %tid, %n ) { pred = s:\"lt\" }
",
                    "    %x : i32 = arith.sub ( %n, %tid )
",
                    "    %sel : i32 = arith.select ( %lt, %x, %k30 )
",
                    "    %k64 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:64 }
",
                    "    %a : i32 = arith.add ( %sel, %k64 )
",
                    "    %v : vec<i32x4> = tile.buffer_load_cp ( %rs, %a, %c0 ) { aux = i:0, dwords = i:4 }
",
                    "    %r : i32 = vector.extract ( %v ) { index = i:2 }
",
        );
        let text = kernel("    %n : i32
", body);
        let off = |l: &str| -> Vec<u32> { lines(l, "buffer_load_dwordx4").iter().map(|(_, w0, _)| w0 & 0xfff).collect() };
        let plain = listing(&text);
        let lev = rir::with_levers(&["bufoffany"], || listing(&text));
        scopes_match_kernel_levers(&text, &["bufoffany"]);
        assert_eq!(off(&plain), [0], "{plain}");
        assert_eq!(off(&lev), [64], "{lev}");
        assert!(!lev.contains("0x40, ") && !lev.lines().any(|x| x.trim_start().starts_with("v_add") && x.contains(" 64, ")), "{lev}");
    }


    #[test]
    fn inline_asm_cache_atomics_scalar_loads_and_fixed_vgprs() {
        // P5's scalar re-read (s_dcache_inv, buffer_inv sc1, s_load_dword glc), P4's ticket atomic, and CCL's
        // fixed-VGPR outputs (`{v[96:99]}`, dwords named v97 / v98 in the template).
        let body = concat!(
                    "    %base : i64 = tile.global_ptr_to_i64 ( %Y )\n",
                    "    %one : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n",
                    "    %w : i32 = tile.inline_asm ( %base ) { constraints = s:\"=s,s,~{memory}\", sideeffect = b:true, template = s:\"s_dcache_inv\\ns_waitcnt lgkmcnt(0)\\nbuffer_inv sc1\\ns_load_dword $0, $1, 0x0 glc\\ns_waitcnt lgkmcnt(0)\" }\n",
                    "    %old : i32 = tile.inline_asm ( %off, %one, %base ) { constraints = s:\"=&v,v,v,s,~{memory}\", sideeffect = b:true, template = s:\"global_atomic_add $0, $1, $2, $3 sc0\\ns_waitcnt vmcnt(0)\" }\n",
                    "    %vec : vec<i32x4>, %sum : i32 = tile.inline_asm ( %off, %base ) { constraints = s:\"={v[96:99]},=&v,v,s,~{memory}\", sideeffect = b:true, template = s:\"global_load_dwordx4 $0, $2, $3 sc1\\ns_waitcnt vmcnt(0)\\nv_add_u32_e32 $1, v97, v98\" }\n",
                    "    %e : i32 = vector.extract ( %vec ) { index = i:0 }\n",
                    "    %t1 : i32 = arith.add ( %w, %old )\n",
                    "    %t2 : i32 = arith.add ( %t1, %sum )\n",
                    "    %r : i32 = arith.add ( %t2, %e )\n",
        );
        let l = listing(&kernel("", body));
        // llvm-mc -mcpu=gfx950: s_dcache_inv = 0xc0800000; buffer_inv sc1 = 0xe0a48000; s_load_dword glc sets bit 16
        // (op 0); global_atomic_add .. sc0 = op 0x42 with glc (bit 16); the fixed load writes v96..v99.
        assert!(lines(&l, "s_dcache_inv").iter().any(|(_, w, _)| *w == 0xc080_0000), "{l}");
        assert!(lines(&l, "buffer_inv").iter().any(|(_, w, _)| *w == 0xe0a4_8000), "{l}");
        assert!(lines(&l, "s_load_dword ").iter().any(|(_, w, _)| w >> 16 & 1 == 1 && w >> 18 & 0xff == 0), "{l}");
        assert!(lines(&l, "global_atomic_add").iter().any(|(_, w, _)| w >> 18 & 0x7f == 0x42 && w >> 16 & 1 == 1), "{l}");
        assert!(lines(&l, "global_load_dwordx4").iter().any(|(_, _, w1)| w1 >> 24 == 96), "{l}");
        assert!(l.contains("v97") && l.contains("v98") || lines(&l, "v_add_u32").iter().any(|(_, w, _)| w >> 9 & 0xff == 98 && w & 0x1ff == 256 + 97), "{l}");
    }


    /// COMPUTE_PGM_RSRC1 of a code object's (only) kernel descriptor (`.rodata`).
    fn rsrc1_of(hsaco: &[u8]) -> u32 {
        let u16_at = |o: usize| u16::from_le_bytes([hsaco[o], hsaco[o + 1]]) as usize;
        let u64_at = |o: usize| u64::from_le_bytes(hsaco[o..o + 8].try_into().unwrap()) as usize;
        let (shoff, shentsize, shnum, shstrndx) = (u64_at(0x28), u16_at(0x3a), u16_at(0x3c), u16_at(0x3e));
        let sh = |i: usize| shoff + i * shentsize;
        let names = u64_at(sh(shstrndx) + 0x18);
        for i in 0..shnum {
            let name_off = u32::from_le_bytes(hsaco[sh(i)..sh(i) + 4].try_into().unwrap()) as usize;
            let name: Vec<u8> = hsaco[names + name_off..].iter().take_while(|&&b| b != 0).copied().collect();
            if name == b".rodata" {
                let at = u64_at(sh(i) + 0x18) + 48;
                return u32::from_le_bytes(hsaco[at..at + 4].try_into().unwrap());
            }
        }
        panic!("no .rodata");
    }

    #[test]
    fn wpecap_caps_the_occupancy_in_the_descriptor() {
        // waves_per_eu = 1: the VGPR granules cover 264 registers (COMGR's descriptor of the fused head), the
        // allocation's own count without the lever.
        let text = kernel("", "    %r : i32 = arith.add ( %tid, %c8 )
").replace("max_workgroup_size = i:256 }", "max_workgroup_size = i:256, waves_per_eu = i:1 }");
        let plain = rir::compile_any_target(&text, Target::Gfx950).unwrap();
        let lev = rir::with_levers(&["wpecap"], || rir::compile_any_target(&text, Target::Gfx950).unwrap());
        assert!(rsrc1_of(&plain.hsaco) & 0x3f < 8, "{:#x}", rsrc1_of(&plain.hsaco));
        assert_eq!(rsrc1_of(&lev.hsaco) & 0x3f, 32, "{:#x}", rsrc1_of(&lev.hsaco));
    }

    /// A lever scope (`dlevers`) applies to its op only: of two buffer loads with an offset of unknown sign,
    /// the scoped one takes `bufoffany`; a `-bufoffany` scope keeps a kernel lever off for its op.
    #[test]
    fn scoped_levers_apply_to_their_ops_only() {
        let load = |name: &str, scope: &str| {
            format!(
                "    %a{name} : i32 = arith.add ( %sel, %k64 )
    %v{name} : vec<i32x4> = tile.buffer_load_cp ( %rs, %a{name}, %c0 ) {{ {scope}aux = i:0, dwords = i:4 }}
    %e{name} : i32 = vector.extract ( %v{name} ) {{ index = i:2 }}
"
            )
        };
        let head = concat!(
            "    %k30 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1073741824 }
",
            "    %lt : i1 = arith.cmp ( %tid, %n ) { pred = s:\"lt\" }
",
            "    %x : i32 = arith.sub ( %n, %tid )
",
            "    %sel : i32 = arith.select ( %lt, %x, %k30 )
",
            "    %k64 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:64 }
",
        );
        let off = |l: &str| -> Vec<u32> { lines(l, "buffer_load_dwordx4").iter().map(|(_, w0, _)| w0 & 0xfff).collect() };
        let text = |s1: &str, s2: &str| {
            let body = format!("{head}{}{}    %r : i32 = arith.xor ( %e1, %e2 )
", load("1", s1), load("2", s2));
            kernel("    %n : i32
", &body)
        };
        let mut both = off(&listing(&text("dlevers = s:\"bufoffany\", ", "")));
        both.sort();
        assert_eq!(both, [0, 64]);
        let mut both = off(&rir::with_levers(&["bufoffany"], || listing(&text("", "dlevers = s:\"-bufoffany\", "))));
        both.sort();
        assert_eq!(both, [0, 64]);
        assert_eq!(off(&listing(&text("", ""))), [0, 0]);
        assert_eq!(off(&rir::with_levers(&["bufoffany"], || listing(&text("", "")))), [64, 64]);
    }

    /// A `tile.sched_barrier` in a `nofence` scope is no scheduling boundary; one outside stays one.
    #[test]
    fn nofence_scopes_one_barrier() {
        let body = |s1: &str, s2: &str| {
            format!(
                "    %p : i32 = arith.add ( %tid, %n )
    tile.sched_barrier (  ) {{ {s1}mask = i:0 }}
    %q : i32 = arith.mul ( %p, %n )
    tile.sched_barrier (  ) {{ {s2}mask = i:0 }}
    %r : i32 = arith.xor ( %q, %tid )
"
            )
        };
        let k = |s1: &str, s2: &str| rir::compile_any_target(&kernel("    %n : i32
", &body(s1, s2)), Target::Gfx950).unwrap().hsaco;
        let none = k("", "");
        let all = rir::with_levers(&["nofence"], || k("", ""));
        let scoped_all = k("dlevers = s:\"nofence\", ", "dlevers = s:\"nofence\", ");
        assert!(all == scoped_all, "both barriers scoped = the kernel lever");
        let off_all = rir::with_levers(&["nofence"], || k("dlevers = s:\"-nofence\", ", "dlevers = s:\"-nofence\", "));
        assert!(off_all == none, "scoped off = no lever");
    }

    /// The ops passes make (the unrolled loop's iteration constants, the constants hoisted to the kernel head)
    /// have no scope of their own: their instructions take their neighbours' (`MFunc::fill_scopes`), so a scope
    /// over every op of an unrolled GEMM (the census kernel that showed it) still builds what the kernel lever
    /// builds, `litreuse` and `litshare` included.
    #[test]
    fn scopes_reach_what_passes_make() {
        let text = include_str!("testdata/unrolled_gemm.ir");
        for lever in ["litreuse", "litshare"] {
            assert!(rir::with_levers(&[lever], || listing(text)) != listing(text), "{lever} changes this kernel");
            scopes_match_kernel_levers(text, &[lever]);
        }
    }

    /// An op of the source outside every scope keeps the kernel's levers even next to scoped ones: a barrier
    /// after ops in a `storewait` scope waits for every load unless it is in the scope itself.
    #[test]
    fn unscoped_source_ops_keep_the_kernels_levers() {
        let body = |sync: &str| {
            let s = "dlevers = s:\"storewait\", ";
            format!(
                "    %v1 : i32 = memref.global_load_typed ( %Y, %tid ) {{ {s}align = i:4, elem_type = s:\"i32\" }}\n    \
                 memref.global_store_typed ( %Y, %off, %v1 ) {{ {s}align = i:4, elem_type = s:\"i32\" }}\n    \
                 %v2 : i32 = memref.global_load_typed ( %Y, %off ) {{ {s}align = i:4, elem_type = s:\"i32\" }}\n    \
                 tile.sync (  ) {sync}\n    %r : i32 = arith.add ( %v2, %tid )\n"
            )
        };
        let waits = |l: &str| -> Vec<String> {
            let lines: Vec<&str> = l.lines().map(str::trim_start).collect();
            let at = lines.iter().position(|x| x.starts_with("s_barrier")).expect("the barrier");
            lines[..at].iter().rev().take_while(|x| x.starts_with("s_waitcnt")).map(|x| x.split("//").next().unwrap().trim().to_string()).collect()
        };
        let plain_text = kernel("", &body("").replace("dlevers = s:\"storewait\", ", ""));
        let plain = waits(&listing(&plain_text));
        let lever = waits(&rir::with_levers(&["storewait"], || listing(&plain_text)));
        assert_ne!(plain, lever, "storewait changes the barrier's wait");
        assert_eq!(waits(&listing(&kernel("", &body("")))), plain, "an unscoped barrier: the kernel's levers");
        assert_eq!(waits(&listing(&kernel("", &body("{ dlevers = s:\"storewait\" }")))), lever, "a scoped barrier");
    }

    /// `loadfirst`: a load whose address waits on a multiply chain issues as soon as the address is ready,
    /// ahead of independent work the scheduler could start sooner (LLVM issues the head GEMV's next-tile loads
    /// before the tile's MFMAs); without it the independent work goes first.
    #[test]
    fn loadfirst_issues_loads_once_their_address_is_ready() {
        let mut body = String::from(
            "    %a1 : i32 = arith.mul ( %tid, %n )\n    %a2 : i32 = arith.mul ( %a1, %n )\n    %a3 : i32 = arith.and ( %a2, %c64 )\n    \
             %ld : vec<i32x4> = tile.buffer_load_cp ( %rs, %a3, %c0 ) { aux = i:0, dwords = i:4 }\n    %l0 : i32 = vector.extract ( %ld ) { index = i:0 }\n",
        );
        let mut acc = "%tid".to_string();
        for i in 0..24 {
            body += &format!("    %w{i} : i32 = arith.mul ( {acc}, %n )\n    %x{i} : i32 = arith.xor ( %w{i}, %tid )\n");
            acc = format!("%x{i}");
        }
        // independent MFMAs the free matrix pipe could start at once: without the lever they go ahead of the load
        body += "    %ha : vec<f16x4> = arith.constant_vec (  ) { elem = s:\"f16\", fill = f:1.0, vec = i:4 }\n    \
                 %m0 : vec<f32x4> = arith.constant_vec (  ) { elem = s:\"f32\", fill = f:0.0, vec = i:4 }\n";
        for i in 0..8 {
            body += &format!("    %m{} : vec<f32x4> = tile.mma ( %ha, %ha, %m{i} ) {{ op_id = s:\"mfma_f32_16x16x16_f16\" }}\n", i + 1);
        }
        body += "    %me : f32 = vector.extract ( %m8 ) { index = i:0 }\n    %mi : i32 = arith.bitcast ( %me )\n";
        body += &format!("    %r0 : i32 = arith.add ( {acc}, %l0 )\n    %r : i32 = arith.xor ( %r0, %mi )\n");
        let text = kernel("    %n : i32\n", &body);
        let at = |l: &str| l.lines().map(str::trim_start).filter(|x| x.starts_with("v_") || x.starts_with("buffer_")).position(|x| x.starts_with("buffer_load")).expect("the load");
        let plain = at(&rir::with_levers(&["vmemfirst"], || listing(&text)));
        let first = at(&rir::with_levers(&["loadfirst"], || listing(&text)));
        assert!(first < plain, "the load at {first} with loadfirst, {plain} with vmemfirst");
        let ahead = |l: &str| {
            let ops: Vec<&str> = l.lines().map(str::trim_start).collect();
            ops.iter().position(|x| x.starts_with("buffer_load")) < ops.iter().position(|x| x.starts_with("v_mfma"))
        };
        assert!(!ahead(&rir::with_levers(&["vmemfirst"], || listing(&text))), "without the lever an MFMA goes first");
        assert!(ahead(&rir::with_levers(&["loadfirst"], || listing(&text))), "the load ahead of the MFMAs");
        scopes_match_kernel_levers(&text, &["loadfirst"]);
    }
}
