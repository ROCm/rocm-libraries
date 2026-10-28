// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! `tile.mma` on gfx950. The op id names one MFMA (VOP3P-MAI), and the IR's
//! per-lane vectors are its operands as they are: rocKE writes the IR for
//! the instruction's lane layout, as its LLVM lowering passes them to the
//! intrinsic (with `cbsz`, `abid` and `blgp` 0).
//!
//! D takes C's registers. C is copied to a fresh tuple that the MFMA then
//! updates in place; where C dies at the MFMA (a loop's accumulator) the
//! copy coalesces away. So D never overlaps C partially, nor A or B at all,
//! which the MFMAs with more than four result dwords require. D and C stay
//! in architected VGPRs (`acc_cd` 0); A and B are VGPRs too.
//!
//! The emulated TF32 atoms (`mfma_f32_16x16x32_xf32`, `mfma_f32_32x32x16_xf32`)
//! expand as rocKE's LLVM lowering expands them (BF16x3): each f32 element
//! of A and B splits into `big`, its bf16 rounding (to nearest even, as
//! `v_cvt_pk_bf16_f32` rounds) with a magnitude past the largest finite
//! bf16 capped there, sign kept; and `small`, the bf16 rounding of what
//! `big` left out. Three bf16 MFMAs then accumulate small*big, big*small
//! and big*big in that order: f32 accumulation is not associative, so the
//! order fixes the result bits.

use super::gen_isa::Gfx9Op;
use crate::error::CompileError;
use crate::rir::mir::*;

fn unsupported(message: String) -> CompileError {
    CompileError::new("ROCKE_DIRECT_UNSUPPORTED", message)
}

/// One MFMA: the instruction, the dwords per lane of A and B and of C and
/// D, and its controls (cbsz in bits 2:0, abid in 6:3, blgp in 9:7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub op: Gfx9Op,
    pub ab: u8,
    pub cd: u8,
    pub ctl: i64,
}

/// The MFMA a plain `tile.mma` op id names on gfx950.
pub fn shape(op_id: &str) -> Option<Shape> {
    use Gfx9Op::*;
    let (op, ab, cd) = match op_id {
        "mfma_f32_16x16x32_bf16" => (VMfma16x16x32Bf16, 4, 4),
        "mfma_f32_16x16x32_f16" => (VMfma16x16x32F16, 4, 4),
        "mfma_f32_16x16x16_f16" => (VMfma16x16x16F16, 2, 4),
        "mfma_f32_16x16x16_bf16" => (VMfma16x16x16Bf16, 2, 4),
        "mfma_f32_32x32x16_f16" => (VMfma32x32x16F16, 4, 16),
        "mfma_f32_32x32x16_bf16" => (VMfma32x32x16Bf16, 4, 16),
        "mfma_f32_32x32x8_f16" => (VMfma32x32x8F16, 2, 16),
        "mfma_f32_32x32x8_bf16" => (VMfma32x32x8Bf16, 2, 16),
        "mfma_f32_4x4x4_f16" => (VMfma4x4x4F16, 2, 4),
        "mfma_f32_4x4x4_bf16" => (VMfma4x4x4Bf16, 2, 4),
        "mfma_f32_16x16x32_fp8" => (VMfma16x16x32Fp8Fp8, 2, 4),
        "mfma_f32_16x16x32_bf8" => (VMfma16x16x32Bf8Bf8, 2, 4),
        // The dense fp8 16x16x128 atom: the f8f6f4 instruction with both
        // formats fp8 (cbsz and blgp 0), as rocKE's lowering calls it.
        "mfma_f32_16x16x128_fp8" => (VMfma16x16x128F8f6f4, 8, 4),
        "mfma_f32_16x16x4_f32" => (VMfma16x16x4F32, 1, 4),
        "mfma_f32_32x32x2_f32" => (VMfma32x32x2F32, 1, 16),
        "mfma_f64_16x16x4_f64" => (VMfma16x16x4F64, 2, 8),
        "mfma_f64_4x4x4_f64" => (VMfma4x4x4F64, 2, 2),
        _ => return None,
    };
    Some(Shape { op, ab, cd, ctl: 0 })
}

/// The block-scaled 32x32x64 atom `mfma_gfx950_f32_32x32x64_<a>_<b>_scale_e8m0_e8m0_k32`
/// (A and B formats fp8, bf8, fp6, bf6 or fp4, one E8M0 scale per 32 K
/// elements): the MFMA with cbsz and blgp naming the formats, and the
/// dwords of A and B the instruction reads (the IR pads both to 8).
pub fn scaled(op_id: &str) -> Option<(Shape, u8, u8)> {
    let formats = op_id.strip_prefix("mfma_gfx950_f32_32x32x64_")?.strip_suffix("_scale_e8m0_e8m0_k32")?;
    let (a, b) = formats.split_once('_')?;
    let format = |f: &str| match f {
        "fp8" => Some((0, 8)),
        "bf8" => Some((1, 8)),
        "fp6" => Some((2, 6)),
        "bf6" => Some((3, 6)),
        "fp4" => Some((4, 4)),
        _ => None,
    };
    let ((ca, na), (cb, nb)) = (format(a)?, format(b)?);
    Some((Shape { op: Gfx9Op::VMfmaScale32x32x64F8f6f4, ab: 8, cd: 16, ctl: ca | cb << 7 }, na, nb))
}

pub fn is_scaled(op_id: &str) -> bool {
    scaled(op_id).is_some()
}

/// The bf16 atom a BF16x3-emulated TF32 op id runs on.
pub fn emulated(op_id: &str) -> Option<Shape> {
    match op_id {
        "mfma_f32_16x16x32_xf32" => shape("mfma_f32_16x16x32_bf16"),
        "mfma_f32_32x32x16_xf32" => shape("mfma_f32_32x32x16_bf16"),
        _ => None,
    }
}

/// The sign of an f32, and the largest finite bf16 as the high half of one.
const SIGN: u32 = 0x8000_0000;
const BF16_MAX_HIGH: u32 = 0x7f7f_0000;
/// `v_perm_b32` selector taking the high halves of src1 (low) and src0 (high).
const PERM_HIGH_HALVES: u32 = 0x0706_0302;

/// Instructions appended to one block of a function under selection.
struct Emit<'a> {
    f: &'a mut MFunc,
    block: usize,
}

impl Emit<'_> {
    fn reg(&mut self, class: Class, n: u8) -> VRef {
        let reg = self.f.vregs.len() as u32;
        self.f.vregs.push(VRegInfo { class, size: n, align: 1, fixed: None });
        VRef { reg, off: 0, n }
    }

    fn push(&mut self, inst: MInst) {
        self.f.blocks[self.block].insts.push(inst);
    }

    fn copy(&mut self, dst: VRef, src: Opnd) {
        self.push(MInst::new(Opc::Copy, [Opnd::R(dst)], [src]));
    }

    fn valu(&mut self, op: Opc, uses: &[Opnd]) -> VRef {
        let dst = self.reg(Class::V, 1);
        self.push(MInst::new(op, [Opnd::R(dst)], uses));
        dst
    }

    /// `v` in VGPRs: MFMAs read A and B from no other file.
    fn in_vgprs(&mut self, v: VRef) -> VRef {
        if self.f.vregs[v.reg as usize].class == Class::V {
            return v;
        }
        let dst = self.reg(Class::V, v.n);
        self.copy(dst, Opnd::R(v));
        dst
    }

    /// D = A * B + C as `shape`, D a copy of C updated in place.
    fn mfma(&mut self, shape: Shape, a: VRef, b: VRef, c: VRef) -> VRef {
        let (a, b) = (self.in_vgprs(a), self.in_vgprs(b));
        let d = self.reg(Class::V, shape.cd);
        self.copy(d, Opnd::R(c));
        self.accumulate(shape, a, b, d);
        d
    }

    fn accumulate(&mut self, shape: Shape, a: VRef, b: VRef, d: VRef) {
        let inst = MInst::new(Opc::Gfx9(shape.op), [Opnd::R(d)], [Opnd::R(a), Opnd::R(b), Opnd::R(d)]);
        self.push(inst.with_imm(shape.ctl));
    }

    /// The block-scaled MFMA: A and B are the first `na` and `nb` dwords of
    /// their 8, and the two scales go to one VGPR pair (A's low), which the
    /// instruction reads after C.
    fn scaled_mfma(&mut self, shape: Shape, (na, nb): (u8, u8), a: VRef, b: VRef, c: VRef, (sa, sb): (Opnd, Opnd)) -> VRef {
        let (a, b) = (self.in_vgprs(a), self.in_vgprs(b));
        let pair = self.reg(Class::V, 2);
        self.copy(VRef { n: 1, ..pair }, sa);
        self.copy(VRef { off: 1, n: 1, ..pair }, sb);
        let d = self.reg(Class::V, shape.cd);
        self.copy(d, Opnd::R(c));
        let (a, b) = (VRef { n: na, ..a }, VRef { n: nb, ..b });
        let uses = [Opnd::R(a), Opnd::R(b), Opnd::R(d), Opnd::R(pair)];
        self.push(MInst::new(Opc::Gfx9(shape.op), [Opnd::R(d)], uses).with_imm(shape.ctl));
        d
    }

    /// The BF16x3 split of the f32 vector `x`: (big, small) as packed bf16
    /// vectors of half its dwords.
    fn split(&mut self, x: VRef, consts: (VRef, VRef, VRef)) -> (VRef, VRef) {
        let (sign, cap, sel) = consts;
        let n = x.n;
        let (mut sat, mut diff) = (Vec::new(), Vec::new());
        for i in 0..n {
            let xi = Opnd::R(VRef { off: x.off + i, n: 1, ..x });
            // The bf16 rounding of x as the high half of an f32 (low half 0).
            let high = self.valu(Opc::Gfx9(Gfx9Op::VCvtPkBf16F32), &[Opnd::Imm(0), xi]);
            // Past the largest finite bf16 in magnitude (infinities and NaNs
            // included): sign | 0x7f7f. Same-signed patterns order as their
            // magnitudes, so that is the unsigned minimum with that cap.
            let limit = self.valu(Opc::VAndOrB32, &[Opnd::R(high), Opnd::R(sign), Opnd::R(cap)]);
            let s = self.valu(Opc::VMinU32, &[Opnd::R(high), Opnd::R(limit)]);
            diff.push(self.valu(Opc::VSubF32, &[xi, Opnd::R(s)]));
            sat.push(s);
        }
        let (big, small) = (self.reg(Class::V, n / 2), self.reg(Class::V, n / 2));
        for k in 0..(n / 2) as usize {
            let (lo, hi) = (VRef { off: big.off + k as u8, n: 1, ..big }, VRef { off: small.off + k as u8, n: 1, ..small });
            let (s0, s1, d0, d1) = (sat[2 * k], sat[2 * k + 1], diff[2 * k], diff[2 * k + 1]);
            self.push(MInst::new(Opc::VPermB32, [Opnd::R(lo)], [Opnd::R(s1), Opnd::R(s0), Opnd::R(sel)]));
            self.push(MInst::new(Opc::Gfx9(Gfx9Op::VCvtPkBf16F32), [Opnd::R(hi)], [Opnd::R(d0), Opnd::R(d1)]));
        }
        (big, small)
    }

    /// The BF16x3 expansion of an emulated TF32 atom running on `shape`.
    fn bf16x3(&mut self, shape: Shape, a: VRef, b: VRef, c: VRef) -> VRef {
        let sign = self.reg(Class::V, 1);
        self.copy(sign, Opnd::Imm(SIGN));
        let cap = self.reg(Class::V, 1);
        self.copy(cap, Opnd::Imm(BF16_MAX_HIGH));
        let sel = self.reg(Class::S, 1);
        self.copy(sel, Opnd::Imm(PERM_HIGH_HALVES));
        let (a, b) = (self.in_vgprs(a), self.in_vgprs(b));
        let (a_big, a_small) = self.split(a, (sign, cap, sel));
        let (b_big, b_small) = self.split(b, (sign, cap, sel));
        let d = self.reg(Class::V, shape.cd);
        self.copy(d, Opnd::R(c));
        for (x, y) in [(a_small, b_big), (a_big, b_small), (a_big, b_big)] {
            self.accumulate(shape, x, y, d);
        }
        d
    }
}

/// Selects `tile.mma` `op_id` over the tuples A, B and C (and a block-scaled
/// atom's A and B scales) into block `block` of `f`, and returns D.
pub fn select(f: &mut MFunc, block: usize, op_id: &str, a: VRef, b: VRef, c: VRef, scales: Option<(Opnd, Opnd)>) -> Result<VRef, CompileError> {
    let mut emit = Emit { f, block };
    if let Some((s, na, nb)) = scaled(op_id) {
        let scales = scales.ok_or_else(|| unsupported(format!("tile.mma {op_id} without its A and B scales")))?;
        if (a.n, b.n, c.n) != (s.ab, s.ab, s.cd) {
            return Err(unsupported(format!("tile.mma {op_id}: operands of {}, {} and {} dwords where gfx950 takes 8, 8 and 16", a.n, b.n, c.n)));
        }
        return Ok(emit.scaled_mfma(s, (na, nb), a, b, c, scales));
    }
    if scales.is_some() {
        return Err(unsupported(format!("tile.mma {op_id} with scales on gfx950")));
    }
    let sizes = |s: Shape, ab: u8| {
        if (a.n, b.n, c.n) == (ab, ab, s.cd) {
            Ok(())
        } else {
            Err(unsupported(format!(
                "tile.mma {op_id}: operands of {}, {} and {} dwords where gfx950's {} takes {ab}, {ab} and {}",
                a.n,
                b.n,
                c.n,
                Opc::Gfx9(s.op).name(),
                s.cd
            )))
        }
    };
    if let Some(s) = emulated(op_id) {
        sizes(s, 2 * s.ab)?;
        return Ok(emit.bf16x3(s, a, b, c));
    }
    let s = shape(op_id).ok_or_else(|| unsupported(format!("tile.mma {op_id} on gfx950")))?;
    sizes(s, s.ab)?;
    Ok(emit.mfma(s, a, b, c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rir;
    use crate::target::Target;

    /// One wave: an f32 vector of `dwords` per lane from global memory, its
    /// bits as A and B of type `ab`, `tile.mma` over C (`c` defines `%c`),
    /// and D's last element stored.
    fn kernel(op_id: &str, ab: &str, dwords: u32, cd: u32, c: &str) -> String {
        format!(
            "ckdsl.ir v1\nkernel @mma {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %X : ptr<f32,global>\n    \
             %Y : ptr<f32,global>\n  }}\n  region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %cn : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{dwords} }}\n    %o : i32 = arith.mul ( %tid, %cn )\n    \
             %x : vec<f32x{dwords}> = memref.global_load_vN ( %X, %o ) {{ align = i:{}, elem_type = s:\"f32\", vec = i:{dwords} }}\n    \
             %a : {ab} = arith.bitcast ( %x ) {{ target = s:\"{ab}\" }}\n    {c}\n    \
             %d : vec<f32x{cd}> = tile.mma ( %a, %a, %c ) {{ op_id = s:\"{op_id}\" }}\n    \
             %e : f32 = vector.extract ( %d ) {{ index = i:{} }}\n    \
             memref.global_store_typed ( %Y, %tid, %e ) {{ align = i:4, elem_type = s:\"f32\" }}\n    cf.return (  )\n  }}\n}}\n",
            4 * dwords,
            cd - 1
        )
    }

    fn zero(cd: u32) -> String {
        format!("%c : vec<f32x{cd}> = arith.constant_vec (  ) {{ elem = s:\"f32\", fill = f:0.0, vec = i:{cd} }}")
    }

    /// The MFMA lines of a listing, without their address and encoding.
    fn mfma_lines(listing: &str) -> Vec<String> {
        listing.lines().filter(|l| l.contains("v_mfma")).map(|l| l.split("//").next().unwrap_or(l).trim().to_string()).collect()
    }

    /// The operands of a listed `v_mfma_<...> d, a, b, c`.
    fn operands(line: &str) -> Vec<String> {
        let (_, rest) = line.split_once(' ').expect("operands");
        rest.split(',').map(|s| s.trim().to_string()).collect()
    }

    #[test]
    fn plain_atoms_name_one_mfma_with_d_on_c() {
        for (op_id, ab, dwords, cd, mnemonic) in [
            ("mfma_f32_16x16x32_f16", "vec<f16x8>", 4, 4, "v_mfma_f32_16x16x32_f16"),
            ("mfma_f32_16x16x16_bf16", "vec<bf16x4>", 2, 4, "v_mfma_f32_16x16x16_bf16"),
            ("mfma_f32_32x32x16_bf16", "vec<bf16x8>", 4, 16, "v_mfma_f32_32x32x16_bf16"),
            ("mfma_f32_32x32x8_f16", "vec<f16x4>", 2, 16, "v_mfma_f32_32x32x8_f16"),
        ] {
            let text = kernel(op_id, ab, dwords, cd, &zero(cd));
            let compiled = rir::compile_any_target(&text, Target::Gfx950).unwrap_or_else(|e| panic!("{op_id}: {e}"));
            let listing = compiled.listing();
            let lines = mfma_lines(&listing);
            assert_eq!(lines.len(), 1, "{listing}");
            assert!(lines[0].starts_with(mnemonic), "{}", lines[0]);
            // D on C, in VGPRs; no AGPR anywhere.
            let ops = operands(&lines[0]);
            assert_eq!(ops[0], ops[3], "{}", lines[0]);
            assert!(ops.iter().all(|o| o.starts_with("v[")), "{}", lines[0]);
            assert!(!listing.contains(" a[") && !listing.contains("accvgpr"), "{listing}");
            assert_eq!(compiled.stats.spill.agpr_dwords, 0);
            // gfx1151 has no MFMA.
            assert!(rir::compile(&text).is_err());
        }
    }

    #[test]
    fn d_is_a_copy_of_c_updated_in_place() {
        let mut f = MFunc { target: Target::Gfx950, ..MFunc::default() };
        f.blocks.push(MBlock::default());
        let mut reg = |class: Class, n: u8| {
            f.vregs.push(VRegInfo { class, size: n, align: 1, fixed: None });
            VRef { reg: f.vregs.len() as u32 - 1, off: 0, n }
        };
        let (a, c, s) = (reg(Class::V, 4), reg(Class::V, 4), reg(Class::S, 4));
        let d = select(&mut f, 0, "mfma_f32_16x16x32_f16", a, s, c, None).unwrap_or_else(|e| panic!("{e}"));
        let insts = &f.blocks[0].insts;
        // B from SGPRs is copied to VGPRs first; then D = C, then D = A * B + D.
        assert_eq!(insts.len(), 3);
        let Opnd::R(b) = insts[0].defs[0] else { panic!("{:?}", insts[0]) };
        assert_eq!((insts[0].op, insts[0].uses[0], f.vregs[b.reg as usize].class), (Opc::Copy, Opnd::R(s), Class::V));
        assert_eq!((insts[1].op, insts[1].defs[0], insts[1].uses[0]), (Opc::Copy, Opnd::R(d), Opnd::R(c)));
        assert_eq!(insts[2].op, Opc::Gfx9(Gfx9Op::VMfma16x16x32F16));
        assert_eq!((insts[2].defs[0], &insts[2].uses[..]), (Opnd::R(d), &[Opnd::R(a), Opnd::R(b), Opnd::R(d)][..]));
        assert_eq!(insts[2].imm, 0);
    }

    #[test]
    fn operands_of_another_size_are_refused() {
        let text = kernel("mfma_f32_16x16x32_f16", "vec<f16x4>", 2, 4, &zero(4));
        let error = rir::compile_any_target(&text, Target::Gfx950).err().expect("refused");
        assert!(error.to_string().contains("operands of 2, 2 and 4 dwords"), "{error}");
        let unknown = kernel("mfma_f32_8x8x8_f16", "vec<f16x8>", 4, 4, &zero(4));
        let error = rir::compile_any_target(&unknown, Target::Gfx950).err().expect("refused");
        assert!(error.to_string().contains("tile.mma mfma_f32_8x8x8_f16 on gfx950"), "{error}");
    }

    #[test]
    fn emulated_tf32_atoms_issue_three_bf16_mfmas_in_order() {
        for (op_id, cd, mnemonic) in [("mfma_f32_16x16x32_xf32", 4, "v_mfma_f32_16x16x32_bf16"), ("mfma_f32_32x32x16_xf32", 16, "v_mfma_f32_32x32x16_bf16")] {
            let splat = format!(
                "%z : f32 = arith.constant (  ) {{ ity = s:\"f32\", value = f:0.0 }}\n    %c : vec<f32x{cd}> = vector.splat ( %z ) {{ vec = i:{cd} }}"
            );
            let text = kernel(op_id, "vec<tf32x8>", 8, cd, &splat);
            let compiled = rir::compile_any_target(&text, Target::Gfx950).unwrap_or_else(|e| panic!("{op_id}: {e}"));
            let listing = compiled.listing();
            let lines = mfma_lines(&listing);
            assert_eq!(lines.len(), 3, "{listing}");
            let ops: Vec<Vec<String>> = lines.iter().map(|l| operands(l)).collect();
            for (line, o) in lines.iter().zip(&ops) {
                assert!(line.starts_with(mnemonic), "{line}");
                assert_eq!((&o[0], &o[3]), (&ops[0][0], &ops[0][0]), "one accumulator: {line}");
            }
            // small*big, big*small, big*big: the last reads the second's A
            // and the first's B.
            let (a_small, b_big, a_big, b_small) = (&ops[0][1], &ops[0][2], &ops[1][1], &ops[1][2]);
            assert!(a_small != a_big && b_small != b_big, "{listing}");
            assert_eq!((&ops[2][1], &ops[2][2]), (a_big, b_big));
            // For A and for B, a rounding per element and one per pair of residuals.
            assert_eq!(listing.matches("v_cvt_pk_bf16_f32").count(), 2 * (8 + 4), "{listing}");
        }
    }

    /// The split's cap as the two instructions compute it on the high half
    /// of an f32, against the lowering's `icmp ugt` and `select` on the
    /// bf16 bits.
    #[test]
    fn the_cap_is_the_unsigned_select_of_the_lowering() {
        let lowering = |bits: u16| {
            let cap = (bits & 0x8000) | 0x7f7f;
            if bits > cap {
                cap
            } else {
                bits
            }
        };
        let ours = |bits: u16| {
            let high = (bits as u32) << 16;
            let limit = (high & SIGN) | BF16_MAX_HIGH;
            (high.min(limit) >> 16) as u16
        };
        for bits in 0..=u16::MAX {
            assert_eq!(ours(bits), lowering(bits), "{bits:#06x}");
        }
    }

    /// The registers a listed tuple `v[a:b]` spans.
    fn span(o: &str) -> usize {
        let inner = o.trim_start_matches(['v', 'a']).trim_matches(['[', ']']);
        inner.split_once(':').map_or(1, |(a, b)| b.parse::<usize>().unwrap() - a.parse::<usize>().unwrap() + 1)
    }

    #[test]
    fn block_scaled_atoms_read_their_formats_and_a_scale_pair() {
        let text = "ckdsl.ir v1\nkernel @scaled {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %X : ptr<i32,global>\n    \
                    %S : ptr<i32,global>\n    %Y : ptr<f32,global>\n  }\n  region @entry {\n    \
                    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n    \
                    %c8 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:8 }\n    %o : i32 = arith.mul ( %tid, %c8 )\n    \
                    %a : vec<i32x8> = memref.global_load_vN ( %X, %o ) { align = i:32, elem_type = s:\"i32\", vec = i:8 }\n    \
                    %sa : i32 = memref.global_load_typed ( %S, %tid ) { align = i:4, elem_type = s:\"i32\" }\n    \
                    %c : vec<f32x16> = arith.constant_vec (  ) { elem = s:\"f32\", fill = f:0.0, vec = i:16 }\n    \
                    %d : vec<f32x16> = tile.mma ( %a, %a, %c, %sa, %sa ) { op_id = s:\"mfma_gfx950_f32_32x32x64_fp6_fp4_scale_e8m0_e8m0_k32\" }\n    \
                    %e : f32 = vector.extract ( %d ) { index = i:15 }\n    \
                    memref.global_store_typed ( %Y, %tid, %e ) { align = i:4, elem_type = s:\"f32\" }\n    cf.return (  )\n  }\n}\n";
        let compiled = rir::compile_any_target(text, Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        let listing = compiled.listing();
        let lines = mfma_lines(&listing);
        assert_eq!(lines.len(), 1, "{listing}");
        assert!(lines[0].starts_with("v_mfma_scale_f32_32x32x64_f8f6f4"), "{}", lines[0]);
        // D (16) on C, A as fp6 (6 dwords), B as fp4 (4), the scale pair; cbsz 2, blgp 4.
        let ops = operands(lines[0].split(" imm:").next().unwrap());
        assert_eq!(ops.iter().map(|o| span(o)).collect::<Vec<_>>(), [16, 6, 4, 16, 2], "{}", lines[0]);
        assert_eq!(ops[0], ops[3]);
        assert!(lines[0].ends_with(&format!("imm:{}", 2 | 4 << 7)), "{}", lines[0]);
        assert_eq!(scaled("mfma_gfx950_f32_32x32x64_bf8_bf6_scale_e8m0_e8m0_k32").map(|(s, a, b)| (s.ctl, a, b)), Some((1 | 3 << 7, 8, 6)));
        assert!(!is_scaled("mfma_gfx950_f32_32x32x64_fp9_fp8_scale_e8m0_e8m0_k32") && !is_scaled("mfma_f32_32x32x16_f16"));
        // The scales are only taken with a scaled atom.
        assert!(rir::compile_any_target(&text.replace("fp6_fp4_scale_e8m0_e8m0_k32", "fp6_fp4"), Target::Gfx950).is_err());
    }

    #[test]
    fn op_ids_name_rows_of_their_sizes() {
        for (id, ab, cd) in [
            ("mfma_f32_16x16x16_f16", 2, 4),
            ("mfma_f32_32x32x16_f16", 4, 16),
            ("mfma_f64_16x16x4_f64", 2, 8),
            ("mfma_f32_32x32x2_f32", 1, 16),
            ("mfma_f32_16x16x128_fp8", 8, 4),
        ] {
            let s = shape(id).unwrap_or_else(|| panic!("{id}"));
            assert_eq!((s.ab, s.cd, s.ctl), (ab, cd, 0), "{id}");
            let sig = super::super::gen_isa::signature(Opc::Gfx9(s.op)).expect("a row");
            assert_eq!((sig.uses[0].dwords, sig.uses[1].dwords, sig.uses[2].dwords, sig.defs[0].dwords), (ab, ab, cd, cd), "{id}");
        }
        assert_eq!(emulated("mfma_f32_32x32x16_xf32").map(|s| s.op), Some(Gfx9Op::VMfma32x32x16Bf16));
        assert!(shape("mfma_f32_16x16x8_xf32").is_none() && emulated("mfma_f32_16x16x8_xf32").is_none());
    }
}
