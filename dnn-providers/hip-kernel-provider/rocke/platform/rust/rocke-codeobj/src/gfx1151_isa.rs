// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Minimal gfx1151 encoder for the qualified wave32 WMMA GEMM.
//!
//! This is deliberately an instruction encoder, not an embedded machine-code
//! image.  The schedule and register assignment are fixed for the one
//! supported rocKE IR pattern, while operand fields and branch displacements
//! are encoded here.

use std::collections::HashMap;

const INLINE_ZERO: u16 = 128;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Label {
    Loop,
    ZeroAccumulator,
    Epilogue,
}

#[derive(Clone, Copy, Debug)]
struct BranchFixup {
    word: usize,
    label: Label,
}

#[derive(Default)]
struct Encoder {
    words: Vec<u32>,
    labels: HashMap<Label, usize>,
    branches: Vec<BranchFixup>,
}

impl Encoder {
    fn word(&mut self, word: u32) {
        self.words.push(word);
    }

    fn dword(&mut self, first: u32, second: u32) {
        self.words.extend([first, second]);
    }

    fn mark(&mut self, label: Label) {
        assert!(self.labels.insert(label, self.words.len()).is_none());
    }

    fn branch(&mut self, opcode: u32, label: Label) {
        let word = self.words.len();
        self.word(opcode);
        self.branches.push(BranchFixup { word, label });
    }

    fn finish(mut self) -> Vec<u8> {
        for fixup in self.branches {
            let target = *self
                .labels
                .get(&fixup.label)
                .expect("all gfx1151 labels must be bound");
            let displacement = target as isize - (fixup.word + 1) as isize;
            let displacement =
                i16::try_from(displacement).expect("gfx1151 branch displacement must fit i16");
            self.words[fixup.word] = (self.words[fixup.word] & 0xffff_0000)
                | u16::from_ne_bytes(displacement.to_ne_bytes()) as u32;
        }
        let mut bytes = Vec::with_capacity(self.words.len() * 4);
        for word in self.words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    fn s_clause(&mut self, clauses_minus_one: u16) {
        self.word(0xbf85_0000 | u32::from(clauses_minus_one));
    }

    fn s_load_b64(&mut self, destination: u8, base_pair: u8, byte_offset: u32) {
        assert_eq!(base_pair % 2, 0);
        self.dword(
            0xf404_0000 | (u32::from(destination) << 6) | u32::from(base_pair / 2),
            0xf800_0000 | byte_offset,
        );
    }

    fn s_load_b128(&mut self, destination: u8, base_pair: u8, byte_offset: u32) {
        assert_eq!(base_pair % 2, 0);
        self.dword(
            0xf408_0000 | (u32::from(destination) << 6) | u32::from(base_pair / 2),
            0xf800_0000 | byte_offset,
        );
    }

    fn s_lshl_b32(&mut self, destination: u8, source: u8, shift: u8) {
        self.word(
            0x8400_0000
                | (u32::from(destination) << 16)
                | (u32::from(INLINE_ZERO + u16::from(shift)) << 8)
                | u32::from(source),
        );
    }

    fn s_add_i32(&mut self, destination: u8, source: u8, value: u8) {
        self.word(
            0x8100_0000
                | (u32::from(destination) << 16)
                | (u32::from(INLINE_ZERO + u16::from(value)) << 8)
                | u32::from(source),
        );
    }

    fn s_mov_b32_zero(&mut self, destination: u8) {
        self.word(0xbe80_0000 | (u32::from(destination) << 16) | u32::from(INLINE_ZERO));
    }

    fn s_cmp_lt_i32(&mut self, left: u8, right: u16) {
        self.word(0xbf04_0000 | (u32::from(right) << 8) | u32::from(left));
    }

    fn s_wait_lgkm_zero(&mut self) {
        self.word(0xbf89_fc07);
    }

    fn s_wait_vm_zero(&mut self) {
        self.word(0xbf89_03f7);
    }

    fn s_delay_alu(&mut self, immediate: u16) {
        self.word(0xbf87_0000 | u32::from(immediate));
    }

    fn s_nop(&mut self) {
        self.word(0xbf80_0000);
    }

    fn s_cbranch_scc1(&mut self, label: Label) {
        self.branch(0xbfa2_0000, label);
    }

    fn s_branch(&mut self, label: Label) {
        self.branch(0xbfa0_0000, label);
    }

    fn s_endpgm(&mut self) {
        self.word(0xbfb0_0000);
    }

    fn vop2(&mut self, base: u32, destination: u8, source0: u16, source1_vgpr: u8) {
        assert!(source0 < 512);
        self.word(
            base | (u32::from(destination) << 17)
                | (u32::from(source1_vgpr) << 9)
                | u32::from(source0),
        );
    }

    fn v_and_b32(&mut self, destination: u8, inline_value: u8, source_vgpr: u8) {
        self.vop2(
            0x3600_0000,
            destination,
            INLINE_ZERO + u16::from(inline_value),
            source_vgpr,
        );
    }

    fn v_add_nc_u32_s_v(&mut self, destination: u8, scalar: u8, vector: u8) {
        self.vop2(0x4a00_0000, destination, u16::from(scalar), vector);
    }

    fn v_or_b32_s_v(&mut self, destination: u8, scalar: u8, vector: u8) {
        self.vop2(0x3800_0000, destination, u16::from(scalar), vector);
    }

    fn v_shift_i32(&mut self, base: u32, destination: u8, shift: u8, source: u8) {
        self.vop2(base, destination, INLINE_ZERO + u16::from(shift), source);
    }

    fn v_ashrrev_i32(&mut self, destination: u8, shift: u8, source: u8) {
        self.v_shift_i32(0x3400_0000, destination, shift, source);
    }

    fn v_lshrrev_b32(&mut self, destination: u8, shift: u8, source: u8) {
        self.v_shift_i32(0x3200_0000, destination, shift, source);
    }

    fn v_mov_b32_zero(&mut self, destination: u8) {
        self.word(0x7e00_0200 | (u32::from(destination) << 17) | u32::from(INLINE_ZERO));
    }

    fn v_mov_b32(&mut self, destination: u8, source: u8) {
        self.word(0x7e00_0200 | (u32::from(destination) << 17) | u32::from(vgpr(source)));
    }

    fn v_cvt_f16_f32(&mut self, destination: u8, high_half: bool, source: u8) {
        let base = if high_half { 0x7f00_1400 } else { 0x7e00_1400 };
        self.word(base | (u32::from(destination) << 17) | u32::from(vgpr(source)));
    }

    fn vop3_binary(&mut self, first: u32, destination: u8, source0: u16, source1: u16) {
        assert!(source0 < 512 && source1 < 512);
        self.dword(
            first | u32::from(destination),
            u32::from(source0) | (u32::from(source1) << 9),
        );
    }

    fn v_mul_lo_u32_s_v(&mut self, destination: u8, scalar: u8, vector: u8) {
        self.vop3_binary(0xd72c_0000, destination, u16::from(scalar), vgpr(vector));
    }

    fn v_lshlrev_b64(&mut self, destination_pair: u8, shift: u8, source_pair: u8) {
        self.vop3_binary(
            0xd73c_0000,
            destination_pair,
            INLINE_ZERO + u16::from(shift),
            vgpr(source_pair),
        );
    }

    fn v_add_co_u32(&mut self, destination: u8, source0: u16, source1: u16) {
        self.vop3_binary(0xd700_6a00, destination, source0, source1);
    }

    fn v_add_co_ci_u32(&mut self, destination: u8, source0: u16, source1: u16) {
        assert!(source0 < 512 && source1 < 512);
        self.dword(
            0xd520_7c00 | u32::from(destination),
            0x01a8_0000 | u32::from(source0) | (u32::from(source1) << 9),
        );
    }

    fn v_mad_u64_u32(
        &mut self,
        destination_pair: u8,
        multiplier_scalar: u8,
        multiplier_vector: u8,
        addend_pair: u8,
    ) {
        self.dword(
            0xd6fe_7c00 | u32::from(destination_pair),
            u32::from(multiplier_scalar)
                | (u32::from(vgpr(multiplier_vector)) << 9)
                | (u32::from(vgpr(addend_pair)) << 18),
        );
    }

    fn v_wmma_f32_16x16x16_f16(&mut self, destination: u8, a: u8, b: u8, accumulator: u8) {
        self.dword(
            0xcc40_4000 | u32::from(destination),
            0x1800_0000
                | u32::from(vgpr(a))
                | (u32::from(vgpr(b)) << 9)
                | (u32::from(vgpr(accumulator)) << 18),
        );
    }

    fn global_load_b128(&mut self, destination: u8, address_pair: u8, byte_offset: u16) {
        self.dword(
            0xdc5e_0000 | u32::from(byte_offset),
            (u32::from(destination) << 24) | 0x007c_0000 | u32::from(address_pair),
        );
    }

    fn global_store_b16(&mut self, address_pair: u8, data: u8, high_half: bool) {
        self.dword(
            if high_half { 0xdc96_0000 } else { 0xdc66_0000 },
            0x007c_0000 | (u32::from(data) << 8) | u32::from(address_pair),
        );
    }
}

const fn vgpr(register: u8) -> u16 {
    256 + register as u16
}

/// Encode the fixed-register instruction schedule selected for the rocKE
/// gfx1151 f16 RCR WMMA kernel.
pub fn encode_wmma_gemm(block_x_is_m: bool) -> Vec<u8> {
    let mut e = Encoder::default();
    let (m_block_sgpr, n_block_sgpr) = if block_x_is_m { (2, 3) } else { (3, 2) };

    e.s_clause(1);
    e.s_load_b64(6, 0, 0x1c);
    e.s_load_b64(4, 0, 0x10);
    e.v_and_b32(13, 15, 0);
    e.s_lshl_b32(2, 2, 4);
    e.s_lshl_b32(3, 3, 4);
    e.s_wait_lgkm_zero();
    e.s_cmp_lt_i32(7, INLINE_ZERO + 1);
    e.s_cbranch_scc1(Label::ZeroAccumulator);
    e.v_add_nc_u32_s_v(1, n_block_sgpr, 13);
    e.v_add_nc_u32_s_v(2, m_block_sgpr, 13);
    e.s_load_b128(8, 0, 0);
    e.s_mov_b32_zero(0);
    e.s_delay_alu(0x132);
    e.v_mul_lo_u32_s_v(5, 7, 1);
    e.v_mov_b32_zero(1);
    e.v_mul_lo_u32_s_v(8, 7, 2);
    e.v_mov_b32(2, 1);
    e.v_mov_b32(3, 1);
    e.v_ashrrev_i32(6, 31, 5);
    e.v_mov_b32(4, 1);
    e.v_ashrrev_i32(9, 31, 8);
    e.v_mov_b32(7, 1);
    e.s_delay_alu(0x224);
    e.v_lshlrev_b64(10, 1, 5);
    e.v_mov_b32(5, 1);
    e.v_lshlrev_b64(14, 1, 8);
    e.v_mov_b32(6, 1);
    e.v_mov_b32(8, 1);
    e.s_wait_lgkm_zero();
    e.v_add_co_u32(9, 10, vgpr(10));
    e.s_delay_alu(0xa1);
    e.v_add_co_ci_u32(10, 11, vgpr(11));
    e.v_add_co_u32(11, 8, vgpr(14));
    e.v_add_co_ci_u32(12, 9, vgpr(15));
    e.s_nop();
    e.s_nop();

    e.mark(Label::Loop);
    e.s_clause(1);
    e.global_load_b128(18, 11, 16);
    e.global_load_b128(14, 11, 0);
    e.s_clause(1);
    e.global_load_b128(26, 9, 16);
    e.global_load_b128(22, 9, 0);
    e.v_add_co_u32(9, vgpr(9), INLINE_ZERO + 32);
    e.s_delay_alu(0xa1);
    e.v_add_co_ci_u32(10, INLINE_ZERO, vgpr(10));
    e.v_add_co_u32(11, vgpr(11), INLINE_ZERO + 32);
    e.v_add_co_ci_u32(12, INLINE_ZERO, vgpr(12));
    e.s_add_i32(0, 0, 16);
    e.s_delay_alu(0x9);
    e.s_cmp_lt_i32(0, 7);
    e.s_wait_vm_zero();
    e.v_wmma_f32_16x16x16_f16(1, 14, 22, 1);
    e.s_cbranch_scc1(Label::Loop);
    e.s_branch(Label::Epilogue);

    e.mark(Label::ZeroAccumulator);
    e.v_mov_b32_zero(8);
    e.s_delay_alu(0x1);
    for destination in (1..=7).rev() {
        e.v_mov_b32(destination, 8);
    }

    e.mark(Label::Epilogue);
    e.v_lshrrev_b32(9, 4, 0);
    e.v_or_b32_s_v(0, n_block_sgpr, 13);
    e.s_lshl_b32(0, 6, 1);
    e.s_delay_alu(0x92);
    e.v_or_b32_s_v(9, m_block_sgpr, 9);
    e.v_mad_u64_u32(9, 6, 9, 0);
    e.v_cvt_f16_f32(0, false, 1);
    e.v_cvt_f16_f32(1, false, 3);
    e.v_cvt_f16_f32(1, true, 4);
    e.v_cvt_f16_f32(0, true, 2);
    e.v_cvt_f16_f32(2, false, 5);
    e.v_add_nc_u32_s_v(11, 0, 9);
    e.v_ashrrev_i32(10, 31, 9);
    e.s_delay_alu(0x112);
    e.v_add_nc_u32_s_v(3, 0, 11);
    e.v_lshlrev_b64(9, 1, 9);
    e.v_ashrrev_i32(12, 31, 11);
    e.s_delay_alu(0x223);
    e.v_add_nc_u32_s_v(13, 0, 3);
    e.v_ashrrev_i32(4, 31, 3);
    e.v_add_co_u32(9, 4, vgpr(9));
    e.s_delay_alu(0x4);
    e.v_lshlrev_b64(11, 1, 11);
    e.v_add_co_ci_u32(10, 5, vgpr(10));
    e.v_ashrrev_i32(14, 31, 13);
    e.v_add_nc_u32_s_v(15, 0, 13);
    e.v_lshlrev_b64(3, 1, 3);
    e.global_store_b16(9, 0, false);
    e.v_add_co_u32(9, 4, vgpr(11));
    e.s_delay_alu(0x1);
    e.v_add_co_ci_u32(10, 5, vgpr(12));
    e.v_lshlrev_b64(11, 1, 13);
    e.v_ashrrev_i32(16, 31, 15);
    e.v_add_nc_u32_s_v(13, 0, 15);
    e.global_store_b16(9, 0, true);
    e.v_add_co_u32(3, 4, vgpr(3));
    e.v_lshlrev_b64(9, 1, 15);
    e.v_add_nc_u32_s_v(15, 0, 13);
    e.v_add_co_ci_u32(4, 5, vgpr(4));
    e.v_ashrrev_i32(14, 31, 13);
    e.v_cvt_f16_f32(0, false, 6);
    e.s_delay_alu(0x4);
    e.v_add_nc_u32_s_v(5, 0, 15);
    e.v_ashrrev_i32(16, 31, 15);
    e.global_store_b16(3, 1, false);
    e.v_add_co_u32(3, 4, vgpr(11));
    e.v_ashrrev_i32(6, 31, 5);
    e.v_add_co_ci_u32(4, 5, vgpr(12));
    e.v_lshlrev_b64(11, 1, 13);
    e.v_lshlrev_b64(13, 1, 15);
    e.v_add_co_u32(9, 4, vgpr(9));
    e.v_lshlrev_b64(5, 1, 5);
    e.v_add_co_ci_u32(10, 5, vgpr(10));
    e.v_add_co_u32(11, 4, vgpr(11));
    e.s_delay_alu(0xa1);
    e.v_add_co_ci_u32(12, 5, vgpr(12));
    e.v_add_co_u32(13, 4, vgpr(13));
    e.v_add_co_ci_u32(14, 5, vgpr(14));
    e.v_add_co_u32(5, 4, vgpr(5));
    e.v_cvt_f16_f32(0, true, 7);
    e.v_cvt_f16_f32(1, false, 8);
    e.v_add_co_ci_u32(6, 5, vgpr(6));
    e.s_clause(4);
    e.global_store_b16(3, 1, true);
    e.global_store_b16(9, 2, false);
    e.global_store_b16(11, 0, false);
    e.global_store_b16(13, 0, true);
    e.global_store_b16(5, 1, false);
    e.s_endpgm();

    let code = e.finish();
    assert_eq!(code.len(), 0x2e8);
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORACLE_HEX: &str = include_str!("gfx1151_wmma_gemm.hex");

    fn oracle() -> Vec<u8> {
        let digits: String = ORACLE_HEX.chars().filter(|c| !c.is_whitespace()).collect();
        (0..digits.len())
            .step_by(2)
            .map(|offset| u8::from_str_radix(&digits[offset..offset + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn xm_encoding_matches_qualified_machine_code() {
        let oracle = oracle();
        assert_eq!(encode_wmma_gemm(true), oracle[0x700..0x9e8]);
    }

    #[test]
    fn xn_encoding_changes_only_grid_mapping_operands() {
        let xm = encode_wmma_gemm(true);
        let xn = encode_wmma_gemm(false);
        let differences: Vec<_> = xm
            .iter()
            .zip(&xn)
            .enumerate()
            .filter_map(|(index, (left, right))| (left != right).then_some(index))
            .collect();
        assert_eq!(differences, [0x2c, 0x30, 0x154, 0x160]);
    }
}
