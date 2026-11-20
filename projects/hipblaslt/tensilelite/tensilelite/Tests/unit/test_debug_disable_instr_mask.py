# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DebugDisableInstrMask: assembly filter, parameter registration and kernel naming."""

import pytest

from tensilelite.Common.DebugInstrMask import (DEBUG_INSTR_MASK_ALL,
                                               applyDebugDisableInstrMask)
from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.Common.ValidParameters import validParameters

pytestmark = pytest.mark.unit

ASM = """\
label_Loop:
  buffer_load_dwordx4 v[0:3], v4, s[8:11], 0 offen offset:0 // read A
  global_load_b128 v[0:3], v[4:5], off
  buffer_load_dword v4, s[8:11], 0 offen lds
  ds_store_b128 v5, v[0:3] offset:0
  ds_write_b64 v5, v[0:1]
  s_waitcnt lgkmcnt(0) vmcnt(0)
  ds_load_b128 v[8:11], v6 offset:256
  ds_read_b32 v8, v6
  ds_bpermute_b32 v8, v6, v7
  v_wmma_f32_16x16x16_bf16 v[16:23], v[8:15], v[0:7], v[16:23]
  v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[2:3], a[0:15]
  v_add_f32 v1, v2, v3 // v_wmma in a comment stays
  buffer_store_dwordx4 v[16:19], v20, s[12:15], 0 offen
  global_store_b32 v[4:5], v1, off
  buffer_atomic_add_f32 v1, v2, s[12:15], 0 offen
  s_cbranch_scc0 label_Loop
"""

CLASSES = {
  1: ["v_wmma_f32", "v_mfma_f32"],
  2: ["buffer_load_dwordx4", "global_load_b128", "buffer_load_dword v4"],
  4: ["buffer_store_dwordx4", "global_store_b32"],
  8: ["ds_load_b128", "ds_read_b32"],
  16: ["ds_store_b128", "ds_write_b64"],
}
UNTOUCHED = ["label_Loop:", "s_waitcnt", "ds_bpermute_b32", "v_add_f32", "buffer_atomic_add_f32", "s_cbranch_scc0"]


def _disabled(out, mnemonic):
  line = next(l for l in out.splitlines() if mnemonic in l)
  return line.lstrip().startswith("// DebugDisableInstrMask: ")


def test_mask_zero_is_identity():
  assert applyDebugDisableInstrMask(ASM, 0) is ASM


@pytest.mark.parametrize("mask", list(range(1, DEBUG_INSTR_MASK_ALL + 1)))
def test_each_combination_disables_exactly_its_classes(mask):
  out = applyDebugDisableInstrMask(ASM, mask)
  assert len(out.splitlines()) == len(ASM.splitlines())
  for bit, mnemonics in CLASSES.items():
    for m in mnemonics:
      assert _disabled(out, m) == bool(mask & bit), (mask, m)
  for m in UNTOUCHED:
    assert not _disabled(out, m), (mask, m)


def test_unknown_bits_rejected():
  with pytest.raises(ValueError):
    applyDebugDisableInstrMask(ASM, DEBUG_INSTR_MASK_ALL + 1)


def test_parameter_registered_with_off_default():
  assert defaultSolution["DebugDisableInstrMask"] == 0
  assert validParameters["DebugDisableInstrMask"] == list(range(DEBUG_INSTR_MASK_ALL + 1))
