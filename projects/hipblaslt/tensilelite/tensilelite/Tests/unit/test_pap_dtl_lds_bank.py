# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check that a DTL PrefetchAcrossPersistent prefetch survives the tail loop and epilogue staging."""

from pathlib import Path
import re

import pytest
import yaml

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_PAP_BANK = re.compile(
    r"^s_and_b32 \S+, s\[sgpr(?:LocalWriteAddrA|LDSBufferWriteInc)\], (0x[0-9a-f]+|\d+)\s+// PAP DTL bank",
    re.MULTILINE,
)
# The register the next-tile prefetch records its primed flag (bit 0) and DTL
# bank in, taken from the kernel's own save code.
_PAP_STATE = re.compile(r"^s_or_b32 s\[sgpr(\w+)\], s\[sgpr\1\], s\d+\s+// encode PAP DTL LDS bank", re.MULTILINE)
# Only emitted when 3+ LDS buffers rotate through LDSBufferWriteInc/ReadInc.
_LDS_BLOCK = re.compile(r"^\.set LdsOneBlockSize, (\d+)", re.MULTILINE)
_LDS_BLOCK_END = re.compile(r"^\.set LdsBlockEndSize, (\d+)", re.MULTILINE)
_INSN = re.compile(r"^([sv]_\w+)\s+([^/]*?)\s*(?://.*)?$")
_LWA = "s[sgprLocalWriteAddrA]"
_LWB = "s[sgprLocalWriteAddrB]"
_LRA = "v[vgprLocalReadAddrA]"
_LRB = "v[vgprLocalReadAddrB]"
_LRA_ORIG = "v[vgprLocalReadAddrOrigA]"
_LRB_ORIG = "v[vgprLocalReadAddrOrigB]"
_WRITE_INC = "s[sgprLDSBufferWriteInc]"
_READ_INC = "s[sgprLDSBufferReadInc]"
_TRACKED = {_LWA, _LWB, _LRA, _LRB}
# Not primed: cleared, or reserved for the next work item with no data issued.
_NOT_PRIMED = (0, 0x80000000)


def _value(state, token):
    if token in state:
        return state[token]
    try:
        return int(token, 0)
    except ValueError:
        return None


def _run(lines, state):
    """Evaluate the straight-line scalar/vector address arithmetic in lines."""
    tracked = _TRACKED | set(state)
    scc = None
    for line in lines:
        insn = _INSN.match(line)
        if not insn:
            continue
        op = insn[1]
        args = [a.strip().replace("+0]", "]") for a in insn[2].split(",")]
        src = [_value(state, a) for a in args[1:]]
        if op == "s_cmp_eq_u32":
            vals = [_value(state, a) for a in args]
            scc = None if None in vals else vals[0] == vals[1]
            continue
        if op == "s_bitcmp1_b32":
            vals = [_value(state, a) for a in args]
            scc = None if None in vals else bool(vals[0] >> vals[1] & 1)
            continue
        if op.startswith("s_cmp") or op.startswith("s_bitcmp"):
            scc = None
            continue
        dst = args[0]
        if op == "s_cselect_b32":
            val = None if scc is None else src[0 if scc else 1]
        elif op == "s_cmov_b32":
            val = None if scc is None else (src[0] if scc else state.get(dst))
        elif op in ("s_mov_b32", "v_mov_b32"):
            val = src[0]
        elif op in ("s_and_b32", "v_and_b32"):
            val = None if None in src else src[0] & src[1]
        elif op == "s_or_b32":
            val = None if None in src else src[0] | src[1]
        elif op in ("s_add_u32", "v_add_u32"):
            val = None if None in src else src[0] + src[1]
        else:
            assert dst not in tracked, f"unexpected update of {dst}: {line}"
            state.pop(dst, None)
            continue
        if op.startswith("s_") and op != "s_mov_b32" and not op.startswith("s_c"):
            scc = None if val is None else (val != 0 if "and" in op or "or" in op else val >= 1 << 32)
        assert val is not None or dst not in tracked, f"lost track of {dst}: {line}"
        if val is None:
            state.pop(dst, None)
        else:
            state[dst] = val
    return state


def _tail_regions(assembly):
    lines = assembly.splitlines()
    read_begin = lines.index("/* Tail: local read reset offsets a */")
    read_end = lines.index("/* tail loop: macs */", read_begin)
    write_begin = max(i for i, line in enumerate(lines[:read_begin])
                      if line == "/* local write reset offsets a */")
    write_end = next(i for i in range(write_begin, read_begin)
                     if re.match(r"^buffer_load_.*\blds\b", lines[i]))
    return lines[write_begin:write_end], lines[read_begin:read_end]


def _pap_state(assembly):
    match = _PAP_STATE.search(assembly)
    assert match, "kernel does not record the DTL PAP bank"
    assert re.search(rf"^\.set sgpr{match[1]}, \d+", assembly, re.MULTILINE)
    return f"s[sgpr{match[1]}]"


def _assert_rotating_bank_round_trip(assembly, blk, num_blk):
    """With 3+ buffers the saved bank is a full buffer offset, restored into both inc registers."""
    pap_state = _pap_state(assembly)
    lines = assembly.splitlines()
    save_end = next(i for i, line in enumerate(lines) if "// encode PAP DTL LDS bank" in line)
    save = lines[save_end - 1:save_end + 1]
    restore_ends = [i for i, line in enumerate(lines) if line.startswith("label_SkipDtlPapBankRestore")]
    assert restore_ends
    for bank in range(0, num_blk * blk, blk):
        state = _run(save, {pap_state: 1, _WRITE_INC: bank})
        assert state[pap_state] == 1 | bank, f"saved PAP bank {state[pap_state]:#x} for buffer {bank:#x}"
        primed = state[pap_state]
        for end in restore_ends:
            start = max(i for i in range(end) if lines[i] == "/* local write addresses: reset inc */")
            state = _run(lines[start:end], {pap_state: primed, _WRITE_INC: 0x40, _READ_INC: 0x40,
                                            _LRA: 0x80, _LRB: 0x2080})
            assert (state[_WRITE_INC], state[_READ_INC]) == (bank, bank)
            assert (state[_LRA], state[_LRB]) == (0x80 + bank, 0x2080 + bank)
    for end in restore_ends:
        swap = next(i for i in range(end, len(lines))
                    if lines[i].startswith(f"s_add_u32 {_READ_INC}, LdsOneBlockSize"))
        assert not any(line.startswith(f"s_mov_b32 {_READ_INC}") for line in lines[end:swap]), (
            "LDSBufferReadInc is reset after the PAP bank restore"
        )


def _assert_rotating_tail_avoids_pap_bank(assembly, blk, num_blk):
    pap_state = _pap_state(assembly)
    writes, reads = _tail_regions(assembly)
    banks = range(0, num_blk * blk, blk)
    states = [(None, s) for s in _NOT_PRIMED] + [(bank, 1 | bank) for bank in banks]
    for current in banks:
        for pap_bank, primed in states:
            expected = blk if pap_bank == 0 else 0
            assert expected != pap_bank
            state = _run(writes, {pap_state: primed, _LWA: 0x100, _LWB: 0x2040, _WRITE_INC: current})
            assert (state[_LWA] + state[_WRITE_INC], state[_LWB] + state[_WRITE_INC]) == (
                0x100 + expected, 0x2040 + expected), f"tail writes buffer {state[_WRITE_INC]:#x} with PAP bank {pap_bank}"
            state = _run(reads, {pap_state: primed, _READ_INC: current, _LRA: 0x80 + current, _LRB: 0x2080 + current,
                                 _LRA_ORIG: 0x80, _LRB_ORIG: 0x2080})
            assert (state[_LRA], state[_LRB]) == (0x80 + expected, 0x2080 + expected), (
                f"tail reads {state[_LRA] - 0x80:#x} after writing {expected:#x}"
            )


def _assert_tail_avoids_pap_bank(assembly, blk):
    pap_state = _pap_state(assembly)
    writes, reads = _tail_regions(assembly)
    states = [(None, s) for s in _NOT_PRIMED] + [(bank, 1 | bank) for bank in (0, blk)]
    for current in (0, blk):
        for pap_bank, primed in states:
            # The tail must not overwrite a primed next-tile group, and must read
            # the bank it loaded into.
            expected = 0 if pap_bank is None else blk ^ pap_bank
            state = _run(writes, {pap_state: primed, _LWA: 0x100 | current, _LWB: 0x2040 | current})
            assert (state[_LWA], state[_LWB]) == (0x100 | expected, 0x2040 | expected), (
                f"tail writes bank {state[_LWA] & blk:#x} with PAP bank {pap_bank}"
            )
            state = _run(reads, {pap_state: primed, _LRA: 0x80 | current, _LRB: 0x2080 | current})
            assert (state[_LRA], state[_LRB]) == (0x80 | expected, 0x2080 | expected), (
                f"tail reads bank {state[_LRA] & blk:#x} after writing {expected:#x}"
            )


def _assert_epilogue_staging_past_banks(assembly, banks_end):
    lines = assembly.splitlines()
    starts = [i for i, line in enumerate(lines) if "// Local address scaled by BPE" in line]
    assert starts
    for start in starts:
        end = next(i for i in range(start, len(lines)) if lines[i].startswith("ds_write"))
        staging = lines[start:end]
        offsets = [int(m[1], 0) for m in
                   (re.match(r"^v_add_u32 v\d+, (\S+), v\d+\s+// add lds offset", line) for line in staging) if m]
        assert offsets and min(offsets) >= banks_end, f"epilogue staging overlaps the A/B banks: {offsets}"
        assert any(line.startswith("s_barrier") for line in staging)


@pytest.mark.parametrize("pgr, epilogue, extra, num_blk", [
    (2, {"UseScaleAlphaVec": 1}, {}, 2),
    (1, {"UseBias": 1}, {}, 2),
    (2, {"UseBias": 1}, {"DtlPlusLdsBuf": 1}, 3),
], ids=["pgr2_scale_alpha_vec", "pgr1_bias", "pgr2_bias_dtl_plus_lds_buf"])
def test_pap_dtl_lds_bank(pgr, epilogue, extra, num_blk, tmp_path):
    config = yaml.safe_load((Path(__file__).parent / "test_data" / "pap_dtl_lds_bank.yaml").read_text())
    problem, params = config["BenchmarkProblems"][0]
    problem.update(epilogue)
    params["ForkParameters"].append({"PrefetchGlobalRead": [pgr]})
    params["ForkParameters"].extend({key: [value]} for key, value in extra.items())
    if epilogue.get("UseBias"):
        params["BenchmarkFinalParameters"].append({"BiasTypeArgs": ["s"]})
    path = tmp_path / "pap_dtl_lds_bank.yaml"
    path.write_text(yaml.safe_dump(config))
    results = emit_kernels_from_config(path, limit=1, arch="gfx950")
    assert len(results) == 1
    _, assembly, error = results[0]
    assert error == 0
    bank = _PAP_BANK.search(assembly)
    assert bank, "kernel does not keep a DTL PAP prefetch in LDS"
    rotating = _LDS_BLOCK.search(assembly)
    assert bool(rotating) == (num_blk > 2)
    if rotating:
        blk = int(rotating[1])
        assert int(_LDS_BLOCK_END.search(assembly)[1]) == num_blk * blk
        _assert_rotating_bank_round_trip(assembly, blk, num_blk)
        _assert_rotating_tail_avoids_pap_bank(assembly, blk, num_blk)
    else:
        blk = int(bank[1], 0)
        _assert_tail_avoids_pap_bank(assembly, blk)
    _assert_epilogue_staging_past_banks(assembly, num_blk * blk)
