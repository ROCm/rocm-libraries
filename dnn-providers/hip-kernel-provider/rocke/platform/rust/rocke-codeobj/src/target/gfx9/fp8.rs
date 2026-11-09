// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! The fp8 and bf8 conversions of gfx950 (OCP e4m3 and e5m2), selected as
//! rocKE's LLVM lowering calls their intrinsics:
//!
//! * `arith.cvt_fp8_to_f32` / `cvt_bf8_to_f32`: `v_cvt_f32_fp8` / `_bf8` of
//!   byte 0 (`llvm.amdgcn.cvt.f32.fp8` with byte select 0).
//! * `arith.cvt_pk_f32_fp8x4` / `_bf8x4`: two `v_cvt_pk_f32_fp8` / `_bf8`,
//!   bytes 0 and 1 into the low f32 pair and, through SDWA src0_sel:WORD_1,
//!   bytes 2 and 3 into the high pair (`llvm.amdgcn.cvt.pk.f32.fp8` with
//!   word select false, then true).
//! * `arith.cvt_f32_to_fp8` / `_bf8`: `v_cvt_pk_fp8_f32 d, x, 0`, the result
//!   in byte 0 (`llvm.amdgcn.cvt.pk.fp8.f32 (x, 0.0, old, false)` then a
//!   trunc to i8).
//! * `arith.cvt_pk_fp8_f32x4` / `cvt_pk_bf8_f32x4`: `v_cvt_pk_fp8_f32` of
//!   elements 0 and 1 into the low word, then of 2 and 3 into the high word
//!   with op_sel[3], which keeps the low word: its old value is a tied use.
//!
//! Each rounds to nearest even and does not saturate.

use super::gen_isa::{Gfx9Op, Role, Sig};
use crate::error::CompileError;
use crate::rir::mir::*;

/// Which conversion an op is: its element format, direction and width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cvt {
    /// One fp8/bf8 byte to f32.
    ToF32,
    /// Four bytes to four f32.
    PkToF32,
    /// One f32 to an fp8/bf8 byte.
    FromF32,
    /// Four f32 to four bytes.
    PkFromF32,
}

fn reg(f: &mut MFunc, n: u8) -> VRef {
    let reg = f.vregs.len() as u32;
    f.vregs.push(VRegInfo { class: Class::V, size: n, align: 1, fixed: None });
    VRef { reg, off: 0, n }
}

fn sub(v: VRef, off: u8, n: u8) -> VRef {
    VRef { reg: v.reg, off: v.off + off, n }
}

/// Selects one conversion into `block` of `f`: `src` is the source word
/// (`ToF32`, `PkToF32`, `FromF32`) or the four f32 registers (`PkFromF32`).
/// Returns the result: an f32 word, a four-dword tuple, or the word whose
/// byte 0 (`FromF32`) or all four bytes (`PkFromF32`) hold the result.
pub fn select(f: &mut MFunc, block: usize, cvt: Cvt, bf8: bool, src: &[Opnd]) -> Result<VRef, CompileError> {
    use Gfx9Op::*;
    let pick = |fp8: Gfx9Op, bf: Gfx9Op| Opc::Gfx9(if bf8 { bf } else { fp8 });
    let push = |f: &mut MFunc, inst: MInst| f.blocks[block].insts.push(inst);
    let need = if cvt == Cvt::PkFromF32 { 4 } else { 1 };
    if src.len() != need {
        return Err(CompileError::new("ROCKE_DIRECT_INTERNAL", format!("fp8 conversion {cvt:?} of {} sources", src.len())));
    }
    Ok(match cvt {
        Cvt::ToF32 => {
            let d = reg(f, 1);
            push(f, MInst::new(pick(VCvtF32Fp8, VCvtF32Bf8), [Opnd::R(d)], [src[0]]));
            d
        }
        Cvt::PkToF32 => {
            let d = reg(f, 4);
            let op = pick(VCvtPkF32Fp8, VCvtPkF32Bf8);
            push(f, MInst::new(op, [Opnd::R(sub(d, 0, 2))], [src[0]]));
            push(f, MInst::new(op, [Opnd::R(sub(d, 2, 2))], [src[0]]).with_mods(M_OPSEL_SRC0));
            d
        }
        Cvt::FromF32 => {
            let d = reg(f, 1);
            push(f, MInst::new(pick(VCvtPkFp8F32, VCvtPkBf8F32), [Opnd::R(d)], [src[0], Opnd::Imm(0)]));
            d
        }
        Cvt::PkFromF32 => {
            let d = reg(f, 1);
            let op = pick(VCvtPkFp8F32, VCvtPkBf8F32);
            push(f, MInst::new(op, [Opnd::R(d)], [src[0], src[1]]));
            push(f, MInst::new(op, [Opnd::R(d)], [src[2], src[3], Opnd::R(d)]).with_mods(M_OPSEL_DST));
            d
        }
    })
}

/// How many of `uses` a VOP3 encoding holds as its sources: at most three,
/// and none from the destination's old value on (a tied use the row names
/// `old`, which the hardware reads through the destination field).
pub fn encoded_sources(sig: &Sig, uses: &[Opnd]) -> usize {
    let old = sig.uses.iter().position(|o| o.role == Role::Old).unwrap_or(usize::MAX);
    uses.len().min(3).min(old)
}

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    fn kernel(params: &str, body: &str) -> String {
        format!(
            "ckdsl.ir v1\nkernel @fp8 {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n{params}  }}\n  \
             region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n{body}    cf.return (  )\n  }}\n}}\n"
        )
    }

    /// Four bytes per lane in, four f32 out, and back: both packed forms.
    fn packed(fmt: &str) -> String {
        kernel(
            &format!("    %X : ptr<{fmt},global>\n    %Y : ptr<f32,global>\n    %Z : ptr<{fmt},global>\n"),
            &format!(
                "    %c4 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:4 }}\n    %o : i32 = arith.mul ( %tid, %c4 )\n    \
                 %x : vec<{fmt}x4> = memref.global_load_vN ( %X, %o ) {{ align = i:4, elem_type = s:\"{fmt}\", vec = i:4 }}\n    \
                 %y : vec<f32x4> = arith.cvt_pk_f32_{short}x4 ( %x )\n    \
                 memref.global_store_vN ( %Y, %o, %y ) {{ align = i:16, elem_type = s:\"f32\", vec = i:4 }}\n    \
                 %z : vec<{fmt}x4> = arith.cvt_pk_{short}_f32x4 ( %y )\n    \
                 memref.global_store_vN ( %Z, %o, %z ) {{ align = i:4, elem_type = s:\"{fmt}\", vec = i:4 }}\n",
                short = &fmt[..3]
            ),
        )
    }

    /// One byte per lane in, one f32 out, and back: both scalar forms.
    fn scalar(fmt: &str) -> String {
        kernel(
            &format!("    %X : ptr<{fmt},global>\n    %Y : ptr<f32,global>\n    %Z : ptr<{fmt},global>\n"),
            &format!(
                "    %x : {fmt} = memref.global_load_typed ( %X, %tid ) {{ align = i:1, elem_type = s:\"{fmt}\" }}\n    \
                 %y : f32 = arith.cvt_{short}_to_f32 ( %x )\n    \
                 memref.global_store_typed ( %Y, %tid, %y ) {{ align = i:4, elem_type = s:\"f32\" }}\n    \
                 %z : {fmt} = arith.cvt_f32_to_{short} ( %y )\n    \
                 memref.global_store_typed ( %Z, %tid, %z ) {{ align = i:1, elem_type = s:\"{fmt}\" }}\n",
                short = &fmt[..3]
            ),
        )
    }

    /// The encoded words of each listing line that starts with `mnemonic`.
    fn words(listing: &str, mnemonic: &str) -> Vec<Vec<u32>> {
        listing
            .lines()
            .filter(|l| l.trim_start().starts_with(mnemonic))
            .map(|l| l.rsplit(": ").next().unwrap_or("").split_whitespace().map(|w| u32::from_str_radix(w, 16).expect("a hex word")).collect())
            .collect()
    }

    #[test]
    fn packed_conversions_encode_as_comgr_does() {
        for (fmt, unpack, pack, vop1, vop3) in [
            ("fp8e4m3", "v_cvt_pk_f32_fp8", "v_cvt_pk_fp8_f32", 0x56, 0x2a2),
            ("bf8e5m2", "v_cvt_pk_f32_bf8", "v_cvt_pk_bf8_f32", 0x57, 0x2a3),
        ] {
            let listing = rir::compile_any_target(&packed(fmt), Target::Gfx950).unwrap_or_else(|e| panic!("{fmt}: {e}")).listing();
            // The low word as VOP1 (7e0cad04: v_cvt_pk_f32_fp8_e32 v[6:7], v4), the
            // high word as SDWA src0_sel:WORD_1 (7e18acf9 00050604).
            let unpacks = words(&listing, unpack);
            assert_eq!(unpacks.len(), 2, "{listing}");
            let (plain, sdwa) = if unpacks[0].len() == 1 { (&unpacks[0], &unpacks[1]) } else { (&unpacks[1], &unpacks[0]) };
            assert_eq!((plain.len(), plain[0] >> 25, plain[0] >> 9 & 0xff), (1, 0x3f, vop1), "{listing}");
            assert_eq!((sdwa.len(), sdwa[0] & 0x1ff, sdwa[0] >> 9 & 0xff), (2, 0xf9, vop1), "{listing}");
            assert_eq!((sdwa[1] >> 8 & 7, sdwa[1] >> 16 & 7), (6, 5), "dst_sel DWORD, src0_sel WORD_1: {listing}");
            assert_eq!(sdwa[1] & 0xff, plain[0] & 0xff, "one source word: {listing}");
            // The low word, then the high word with op_sel[3] (d2a20000 00020b04,
            // then d2a24000 00020702) into the same register.
            let packs = words(&listing, pack);
            assert_eq!(packs.len(), 2, "{listing}");
            for p in &packs {
                assert_eq!((p.len(), p[0] >> 16 & 0x3ff), (2, vop3), "{listing}");
                assert_eq!(p[1] >> 18 & 0x1ff, 0, "no third source: {listing}");
            }
            assert_eq!((packs[0][0] >> 11 & 0xf, packs[1][0] >> 11 & 0xf), (0, 8), "{listing}");
            assert_eq!(packs[0][0] & 0xff, packs[1][0] & 0xff, "one destination: {listing}");
        }
    }

    #[test]
    fn scalar_conversions_encode_as_comgr_does() {
        for (fmt, to_f32, from_f32, vop1, vop3) in [
            ("fp8e4m3", "v_cvt_f32_fp8", "v_cvt_pk_fp8_f32", 0x54, 0x2a2),
            ("bf8e5m2", "v_cvt_f32_bf8", "v_cvt_pk_bf8_f32", 0x55, 0x2a3),
        ] {
            let listing = rir::compile_any_target(&scalar(fmt), Target::Gfx950).unwrap_or_else(|e| panic!("{fmt}: {e}")).listing();
            // 7e3ea91f: v_cvt_f32_fp8_e32 v31, v31 (byte 0).
            let up = words(&listing, to_f32);
            assert!(up.len() == 1 && up[0].len() == 1 && up[0][0] >> 9 & 0xff == vop1, "{listing}");
            // d2a2002a 00010124: v_cvt_pk_fp8_f32 v42, v36, 0 (src1 the inline 0).
            let down = words(&listing, from_f32);
            assert!(down.len() == 1 && down[0][0] >> 16 & 0x3ff == vop3 && down[0][1] >> 9 & 0x1ff == 0x80, "{listing}");
        }
    }

    #[test]
    fn gfx1151_refuses_the_conversions() {
        for text in [packed("fp8e4m3"), scalar("bf8e5m2")] {
            let e = rir::compile_any_target(&text, Target::Gfx1151).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(e.contains("on gfx1151"), "{e}");
        }
    }
}
