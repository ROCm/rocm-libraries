# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DebugDisableInstrMask: assembly filter, parameter registration and kernel naming."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from config_harness import assert_assembles, emit_kernels_from_config
from tensilelite.Common.DebugInstrMask import (DEBUG_INSTR_CLASSES, DEBUG_INSTR_KEEP_TAG,
                                               DEBUG_INSTR_MASK_ALL,
                                               applyDebugDisableInstrMask, keepInstrComment)
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


@pytest.mark.parametrize("mask", [DEBUG_INSTR_MASK_ALL, 2, 4, 8, 16])
def test_keep_tag_survives(mask):
  asm = (f"  buffer_load_dword v4, v5, s[8:11], 0 offen // get flag {DEBUG_INSTR_KEEP_TAG}\n"
         f"  buffer_store_dword v4, v5, s[8:11], 0 offen // set flag {DEBUG_INSTR_KEEP_TAG}\n"
         f"  ds_store_b32 v1, v2 // {DEBUG_INSTR_KEEP_TAG}\n"
         f"  ds_load_b32 v2, v1 // {DEBUG_INSTR_KEEP_TAG}\n")
  assert applyDebugDisableInstrMask(asm, mask) == asm


@pytest.mark.parametrize("mask,expected", [(0, "c"), (1, f"c {DEBUG_INSTR_KEEP_TAG}"),
                                           (DEBUG_INSTR_MASK_ALL, f"c {DEBUG_INSTR_KEEP_TAG}")])
def test_keep_comment_only_tags_with_nonzero_mask(mask, expected):
  writer = SimpleNamespace(states=SimpleNamespace(kernel={"DebugDisableInstrMask": mask}))
  assert keepInstrComment(writer, "c") == expected
  assert keepInstrComment(SimpleNamespace(states=SimpleNamespace()), "c") == "c"


# --- Codegen: Stream-K control data survives every mask (gfx950: VMEM flags). ---

_SK_CONFIG = Path(__file__).parent / "characterization/_codegen/data/test_data/_designed/gfx942/streamk.yaml"
_ALL_PREFIXES = tuple(p for ps in DEBUG_INSTR_CLASSES.values() for p in ps)
_LIVE_MEM = re.compile(r"^\s*((?:%s)\w*)\b(.*)$" % "|".join(_ALL_PREFIXES), re.M)


@pytest.fixture(scope="module")
def _sk_kernels(tmp_path_factory):
  config = yaml.safe_load(_SK_CONFIG.read_text())
  forks = config["BenchmarkProblems"][0][1]["ForkParameters"]
  for p in forks:
    if "WorkAssignment" in p:
      p["WorkAssignment"] = ["StaticGrid", "DynamicWorkQueue"]
    elif "StreamKAtomic" in p:
      p["StreamKAtomic"] = [0]
  forks.append({"DebugDisableInstrMask": [0, DEBUG_INSTR_MASK_ALL]})
  path = tmp_path_factory.mktemp("ddim") / "sk_ddim_gfx950.yaml"
  path.write_text(yaml.safe_dump(config))
  results = emit_kernels_from_config(path, limit=8, arch="gfx950", canonical=False)
  assert len(results) == 4 and all(err == 0 for _, _, err in results)
  return {(("DebugDisableInstrMask:" in src), ("mailbox publish" in src)): (name, src)
          for name, src, _ in results}


# Static grid: fixup ready flag (VMEM on gfx950). Dynamic queue: arrival fixup
# (scalar-atomic tickets, no flag) plus the LDS work-item mailbox.
@pytest.mark.parametrize("dynamic,expected", [
  (False, {"buffer_load_dword", "buffer_store_dword"}),
  (True, {"ds_write_b32", "ds_read_b32"}),
])
def test_streamk_full_mask_keeps_only_control_data(_sk_kernels, dynamic, expected):
  name, src = _sk_kernels[(True, dynamic)]
  live = _LIVE_MEM.findall(src)
  untagged = [m for m, rest in live if DEBUG_INSTR_KEEP_TAG not in rest]
  assert not untagged, untagged
  assert {m for m, _ in live} == expected
  assert_assembles(src, name)


@pytest.mark.parametrize("dynamic", [False, True])
def test_streamk_mask_zero_has_no_keep_tag(_sk_kernels, dynamic):
  _, src = _sk_kernels[(False, dynamic)]
  assert DEBUG_INSTR_KEEP_TAG not in src
