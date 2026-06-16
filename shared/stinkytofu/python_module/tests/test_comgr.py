# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for comgr-based toolchain capability probing."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/lib"))

import stinkytofu


class TestComgrSupport:
    def test_has_comgr_support(self):
        assert stinkytofu.hasComgrSupport() is True


class TestProbeToolchainCaps:
    def test_gfx1250_returns_dict(self):
        caps = stinkytofu.probeToolchainCaps([12, 5, 0])
        assert isinstance(caps, dict)
        assert "VgprMsbMode" in caps

    def test_gfx1250_vgpr_msb_mode_non_zero(self):
        caps = stinkytofu.probeToolchainCaps([12, 5, 0])
        assert caps["VgprMsbMode"] != stinkytofu.VgprMsbMode.NONE.value

    def test_gfx1250_vgpr_msb_mode_is_msb8_or_msb16(self):
        caps = stinkytofu.probeToolchainCaps([12, 5, 0])
        assert caps["VgprMsbMode"] in (
            stinkytofu.VgprMsbMode.MSB8.value,
            stinkytofu.VgprMsbMode.MSB16.value,
        )


class TestVgprMsbModeEnum:
    def test_enum_values(self):
        assert stinkytofu.VgprMsbMode.NONE.value == 0
        assert stinkytofu.VgprMsbMode.MSB8.value == 1
        assert stinkytofu.VgprMsbMode.MSB16.value == 2


class TestTryAssemble:
    """Tests for the tryAssemble Python binding."""

    def test_valid_instruction(self):
        assert stinkytofu.tryAssemble("v_mov_b32 v0, v1", [12, 5, 0]) is True

    def test_invalid_instruction(self):
        assert stinkytofu.tryAssemble("v_nonexistent_op v0, v1", [12, 5, 0]) is False


class TestGetHardwareCaps:
    def test_gfx1250_returns_all_dicts(self):
        caps = stinkytofu.getHardwareCaps([12, 5, 0])
        assert isinstance(caps, dict)
        for key in ("asmCaps", "archCaps", "regCaps", "asmBugs"):
            assert key in caps
            assert isinstance(caps[key], dict)

    def test_gfx1250_asm_bugs(self):
        bugs = stinkytofu.getHardwareCaps([12, 5, 0])["asmBugs"]
        assert isinstance(bugs["ExplicitCO"], bool)
        assert isinstance(bugs["ExplicitNC"], bool)

    def test_idempotent(self):
        caps1 = stinkytofu.getHardwareCaps([12, 5, 0])
        caps2 = stinkytofu.getHardwareCaps([12, 5, 0])
        assert caps1["asmCaps"] == caps2["asmCaps"]
        assert caps1["archCaps"] == caps2["archCaps"]


class TestAsmCapsVsCompiler:
    """Verify asmCaps values against actual compiler assembly probes.

    For mnemonic-based caps, we check that the cap is 1 iff tryAssemble
    succeeds for the corresponding instruction.  For modifier/encoding
    caps we use the exact assembly strings that HardwareCaps.cpp uses.
    """

    ARCH = [12, 5, 0]

    @staticmethod
    def _asm(s):
        return stinkytofu.tryAssemble(s, TestAsmCapsVsCompiler.ARCH)

    @staticmethod
    def _asm_any(*strs):
        return any(stinkytofu.tryAssemble(s, TestAsmCapsVsCompiler.ARCH) for s in strs)

    def _caps(self):
        return stinkytofu.getHardwareCaps(self.ARCH)["asmCaps"]

    # -- Mnemonic-only caps --

    def test_has_explicit_co(self):
        asm = self._caps()
        assert asm["HasExplicitCO"] == int(self._asm("v_add_co_u32 v0, vcc_lo, v1, v2"))

    def test_has_explicit_nc(self):
        asm = self._caps()
        assert asm["HasExplicitNC"] == int(self._asm("v_add_nc_u32 v0, v1, v2"))

    def test_has_mfma(self):
        asm = self._caps()
        expected = int(
            self._asm_any(
                "v_mfma_f32_32x32x2bf16 a[0:31], v0, v1, a[0:31]",
                "v_mfma_f32_32x32x1_2b_f32 a[0:31], v0, v1, a[0:31]",
            )
        )
        assert asm["HasMFMA"] == expected

    def test_has_wmma(self):
        asm = self._caps()
        expected = int(
            self._asm_any(
                "v_wmma_f32_16x16x16_f16 v[0:7], v[8:15], v[16:23], v[0:7]",
                "v_wmma_f32_16x16x32_bf16 v[0:7], v[8:15], v[16:23], v[0:7]",
                "v_wmma_f32_16x16x4_f32 v[0:7], v[8:15], v[16:23], v[0:7]",
            )
        )
        assert asm["HasWMMA"] == expected

    def test_has_new_barrier(self):
        asm = self._caps()
        assert asm["HasNewBarrier"] == int(self._asm("s_barrier_wait -1"))

    def test_has_tdm(self):
        asm = self._caps()
        # HasTDM is a mnemonic-lookup cap; just verify it's a valid int
        assert asm["HasTDM"] in (0, 1)

    def test_s_delay_alu(self):
        asm = self._caps()
        assert asm["s_delay_alu"] == int(self._asm("s_delay_alu 0"))

    # -- Modifier/encoding caps (comgr-probed) --

    def test_has_direct_to_lds(self):
        asm = self._caps()
        expected = int(
            self._asm_any(
                "buffer_load_dword v36, s[24:27], s28 offen offset:0 lds",
                "buffer_load_b32 v36, s[24:27], s28 offen offset:0 lds",
            )
        )
        assert asm["HasDirectToLds"] == expected

    def test_has_wmma_v1(self):
        asm = self._caps()
        expected = int(
            self._asm("v_wmma_f32_16x16x16_f16 v[0:3], v[8:15], v[16:23], v[0:3]")
        )
        assert asm["HasWMMA_V1"] == expected

    def test_has_wmma_v2(self):
        asm = self._caps()
        # V2 uses 2-register source, but is overridden to 0 if V3 is present
        v2_asm = int(
            self._asm("v_wmma_f32_16x16x16_f16 v[0:3], v[8:9], v[16:17], v[0:3]")
        )
        v3 = asm["HasWMMA_V3"]
        expected = 0 if v3 else v2_asm
        assert asm["HasWMMA_V2"] == expected

    def test_has_glc_modifier(self):
        asm = self._caps()
        expected = int(
            self._asm_any(
                "buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, glc",
                "buffer_load_dwordx4 v[10:13], v[0], s[0:3], null, offen offset:0, glc",
            )
        )
        assert asm["HasGLCModifier"] == expected

    def test_has_scope_modifier(self):
        asm = self._caps()
        expected = int(
            self._asm_any(
                "buffer_load_dwordx4 v[10:13], v[0], s[0:3], 0, offen offset:0, scope:SCOPE_DEV",
                "buffer_load_dwordx4 v[10:13], v[0], s[0:3], null offen offset:0, scope:SCOPE_DEV",
            )
        )
        assert asm["HasSCOPEModifier"] == expected

    def test_v_fmac_f16(self):
        asm = self._caps()
        assert asm["v_fmac_f16"] == int(self._asm("v_fma_f16 v47, v36, v34"))

    def test_vop3_v_dot4_i32_i8(self):
        asm = self._caps()
        assert asm["VOP3v_dot4_i32_i8"] == int(
            self._asm("v_dot4_i32_i8 v47, v36, v34, v47")
        )

    def test_has_cvt_fp8_to_f16(self):
        asm = self._caps()
        assert asm["HasCvtFP8toF16"] == int(
            self._asm("v_cvt_f16_fp8 v[0], v[1] byte_sel:2")
        )

    # -- Wait count limits (derived from instruction support) --

    def test_separate_vmcnt_and_limits(self):
        asm = self._caps()
        has_loadcnt = int(self._asm("s_wait_loadcnt 0"))
        has_storecnt = int(self._asm("s_wait_storecnt 0"))
        assert asm["SeparateVMcnt"] == (has_loadcnt and has_storecnt)
        if asm["SeparateVMcnt"]:
            assert asm["MaxLoadcnt"] == 63
            assert asm["MaxStorecnt"] == 63

    def test_has_xcnt(self):
        asm = self._caps()
        assert asm["HasXcnt"] == int(self._asm("s_wait_xcnt 0"))
        if asm["HasXcnt"]:
            assert asm["MaxXcnt"] == 63


class TestArchCaps:
    """Verify archCaps values for gfx1250 are correct based on ISA version checks."""

    def test_gfx1250_known_values(self):
        arch = stinkytofu.getHardwareCaps([12, 5, 0])["archCaps"]
        assert arch["HasWave32"] == 1
        assert arch["HasSchedMode"] == 1
        assert arch["NoSDWA"] == 1
        assert arch["HasEccHalf"] == 1
        assert arch["DeviceLDS"] == 327680
        assert arch["CMPXWritesSGPR"] == 0
        assert arch["HasInvWbDevFences"] == 1
        assert arch["MaxSgprPreload"] == 32
        assert arch["LDSBankCount"] == 64


class TestRegCaps:
    """Verify regCaps values for gfx1250 are correct."""

    def test_gfx1250_known_values(self):
        reg = stinkytofu.getHardwareCaps([12, 5, 0])["regCaps"]
        assert reg["MaxVgpr"] == 1024
        assert reg["MaxSgpr"] == 102
        assert reg["PhysicalMaxVgpr"] == 1024
        assert reg["PhysicalMaxVgprCU"] == 4096 * 32
