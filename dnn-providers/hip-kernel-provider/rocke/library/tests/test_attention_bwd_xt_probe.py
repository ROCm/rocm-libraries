# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Device probe of the XT transposed LDS layout: bank conflicts vs the predictor.

For each XT image of the gfx942 start points (Q^T / dO^T: ``D x kM0``; K^T:
``D x kN0``) and each swizzle under test (``xor``, the planner's choice, and
``none``, the control), two probe kernels are built:

* writer kernel: the XT writer of ``_attention_bwd_lds`` (global vector loads,
  transposed ``k_per_thread``-element LDS stores) repeated ``reps`` times, then
  a linear 16-byte read-back of the LDS image to global memory;
* reader kernel: a linear 16-byte fill of the LDS image from a host-built
  physical image, then every 16x16x16 B-operand fragment read
  (``ds_read_b64``) of the image repeated ``reps`` times, stored to global.

The linear fill and read-back are conflict-free by construction, so the
conflict counter of the writer kernel measures the writer stores and the one of
the reader kernel measures the B reads. Every repeat's LDS address carries a
runtime ``+ 0`` (``iv * zero``) so the compiler keeps all repeats.

Checks:

* CPU (needs comgr and ``llvm-objdump``): the probe kernels compile for gfx942
  and use exactly the intended LDS opcodes (``ds_read_b64`` reads,
  ``ds_write_b32`` / ``ds_write_b64`` writer stores, ``ds_*_b128`` fill and
  read-back).
* Device (gfx942 or gfx950): the read-back image equals the layout map and the
  reader returns the MFMA B fragments (the layout is correct end to end).
* Device + ``rocprofv3``: the hardware counter ``SQ_LDS_BANK_CONFLICT`` of each
  probe kernel is zero exactly when the predictor reports the instruction
  pattern conflict-free, and non-zero otherwise. Method: rocprofv3 PMC
  collection per dispatch (``--pmc SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE``) of
  a child process that launches the probes. Only zero / non-zero verdicts are
  printed; raw counter values go to the JSON file named by the
  ``ROCKE_BWD_PROBE_RECORD`` environment variable when it is set.

Run ``python test_attention_bwd_xt_probe.py --run-probe`` to launch the probes
alone (the profiler child does this).
"""

from __future__ import annotations

import csv
import ctypes
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

if __name__ == "__main__":  # child process: make the library importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "platform" / "python"))

import pytest
from rocke.core.ir import F16, I32, IRBuilder, PtrType

from kernels.common._attention_bwd_lds import (
    LDS_LINE_BYTES,
    XtLayout,
    XtWriter,
    default_xt_writer,
    xt_conflicts,
)
from ._attention_bwd_test_utils import device_arch

THREADS = 256
REPS = 64
GRID = 1024
# (tag, rows = D, cols = sequence extent)
SHAPES = [
    ("qt_d128_m16", 128, 16),
    ("qt_d64_m32", 64, 32),
    ("qt_d32_m32", 32, 32),
    ("kt_d128_n128", 128, 128),
    ("kt_d64_n128", 64, 128),
    ("kt_d32_n128", 32, 128),
]
SWIZZLES = ("xor", "none")


def _kernel_name(mode: str, tag: str, swizzle: str) -> str:
    return f"bwd_xt_probe_{mode}_{tag}_{swizzle}"


class _Ir:
    def __init__(self, b: IRBuilder):
        self.b = b

    def c(self, v: int):
        return self.b.const_i32(int(v))

    def offset(self, lay: XtLayout, row, col):
        """IR twin of :meth:`XtLayout.offset` (bytes)."""
        b, c = self.b, self.c
        eb = lay.elem_bytes
        if lay.swizzle != "xor":
            return b.add(b.mul(row, c(lay.row_pitch_bytes)), b.mul(col, c(eb)))
        p = lay.cols * eb
        nch = LDS_LINE_BYTES // lay.chunk_bytes
        rpl = max(1, LDS_LINE_BYTES // p)
        lin = b.add(b.mul(row, c(p)), b.mul(col, c(eb)))
        line = b.div(lin, c(LDS_LINE_BYTES))
        within = b.mod(lin, c(LDS_LINE_BYTES))
        chunk = b.div(within, c(lay.chunk_bytes))
        inner = b.mod(within, c(lay.chunk_bytes))
        g = b.div(row, c(rpl))
        h = b.mod(b.xor(g, b.div(g, c(nch))), c(nch))
        return b.add(
            b.add(
                b.mul(line, c(LDS_LINE_BYTES)),
                b.mul(b.xor(chunk, h), c(lay.chunk_bytes)),
            ),
            inner,
        )


def _emit_writer(ir: _Ir, lay: XtLayout, w: XtWriter, src, lds, tid, jitter) -> None:
    b, c = ir.b, ir.c
    if w.units % w.threads:
        raise ValueError("probe writer needs whole iterations")
    dl = w.d_lanes_eff
    for it in range(w.iterations):
        u = b.add(tid, c(it * w.threads))
        dg_lo = b.mod(u, c(dl))
        rest = b.div(u, c(dl))
        sg = b.mod(rest, c(w.n_sg))
        dg = b.add(b.mul(b.div(rest, c(w.n_sg)), c(dl)), dg_lo)
        s0 = b.mul(sg, c(w.k_per_thread))
        d0 = b.mul(dg, c(w.vec_d))
        loads = []
        for r in range(w.k_per_thread):
            gidx = b.add(b.mul(b.add(s0, c(r)), c(w.depth)), d0)
            if w.vec_d == 1:
                loads.append(b.global_load_f16(src, gidx))
            else:
                loads.append(b.global_load_vN(src, gidx, F16, w.vec_d))
        for j in range(w.vec_d):
            elems = [ld if w.vec_d == 1 else b.vec_extract(ld, j) for ld in loads]
            val = b.vec_pack(elems, F16)
            off = ir.offset(lay, b.add(d0, c(j)), s0)
            if jitter is not None:
                off = b.xor(off, jitter)
            b.smem_store_vN(lds, [b.div(off, c(2))], val, w.k_per_thread)


def _for_slots(b: IRBuilder, tid, n_slots: int, body) -> None:
    """``body(slot)`` for ``slot = tid + k * THREADS < n_slots`` (one per thread per k)."""
    for k in range(-(-n_slots // THREADS)):
        slot = b.add(tid, b.const_i32(k * THREADS))
        if (k + 1) * THREADS <= n_slots:
            body(slot)
        else:
            with b.scf_if(b.cmp_lt(slot, b.const_i32(n_slots))):
                body(slot)


def make_probe_kernel(mode: str, tag: str, lay: XtLayout, w: XtWriter):
    """Writer (``mode = "w"``) or reader (``"r"``) probe kernel for one image."""
    b = IRBuilder(_kernel_name(mode, tag, lay.swizzle))
    b.kernel.attrs["max_workgroup_size"] = THREADS
    ir = _Ir(b)
    c = ir.c
    src = b.param("src", PtrType(F16, "global"), readonly=True)
    out = b.param("out", PtrType(F16, "global"))
    reps = b.param("reps", I32)
    zero = b.param("zero", I32)
    halfs = lay.size_bytes // 2
    lds = b.smem_alloc(F16, [halfs])
    tid = b.thread_id_x()
    lane = b.mod(tid, c(64))
    n_lin = lay.size_bytes // 16  # 16-byte slots of the image
    if mode == "w":
        with b.scf_for(c(0), reps, c(1), iv_name="rep") as iv:
            _emit_writer(ir, lay, w, src, lds, tid, b.mul(iv, zero))
        b.sync()

        def readback(slot):
            v = b.smem_load_vN(lds, b.mul(slot, c(8)), dtype=F16, n=8)
            b.global_store_vN(out, b.mul(slot, c(8)), v, 8)

        _for_slots(b, tid, n_lin, readback)
        return b.kernel

    # Reader: linear fill from the physical image in ``src``, then B reads.
    def fill(slot):
        v = b.global_load_vN(src, b.mul(slot, c(8)), F16, 8)
        b.smem_store_vN(lds, [b.mul(slot, c(8))], v, 8)

    _for_slots(b, tid, n_lin, fill)
    b.sync()
    frags = [(n0, k0) for n0 in range(0, lay.rows, 16) for k0 in range(0, lay.cols, 16)]
    with b.scf_for(c(0), reps, c(1), iv_name="rep") as iv:
        jit = b.mul(iv, zero)
        row_l = b.mod(lane, c(16))
        col_l = b.mul(b.div(lane, c(16)), c(4))
        for f, (n0, k0) in enumerate(frags):
            off = ir.offset(lay, b.add(row_l, c(n0)), b.add(col_l, c(k0)))
            x = b.smem_load_vN(lds, b.div(b.xor(off, jit), c(2)), dtype=F16, n=4)
            oidx = b.xor(b.mul(b.add(lane, c(f * 64)), c(4)), jit)
            b.global_store_vN(out, oidx, x, 4)
    return b.kernel


def _probe_cases() -> list[tuple[str, str, XtLayout, XtWriter]]:
    cases = []
    for tag, rows, cols in SHAPES:
        w = default_xt_writer(cols, rows, THREADS)
        for sw in SWIZZLES:
            cases.append((tag, sw, XtLayout(rows, cols, 2, sw), w))
    return cases


def _expected(lay: XtLayout, w: XtWriter, src):
    """Host twins: physical image bytes (as f16 bits) and reader fragments."""
    import numpy as np

    img = np.zeros(lay.size_bytes // 2, dtype=np.uint16)
    for r in range(lay.rows):  # rows = d, cols = sequence
        for col in range(lay.cols):
            img[lay.offset(r, col) // 2] = src[col * lay.rows + r]
    frags = []
    for n0 in range(0, lay.rows, 16):
        for k0 in range(0, lay.cols, 16):
            for lane in range(64):
                row, col = n0 + lane % 16, k0 + (lane // 16) * 4
                frags.extend(src[(col + i) * lay.rows + row] for i in range(4))
    return img, np.array(frags, dtype=np.uint16)


def _src_bits(rows: int, cols: int):
    import numpy as np

    # Distinct finite f16 bit patterns (exponent field never all ones).
    n = rows * cols
    idx = np.arange(n, dtype=np.uint32)
    return ((idx % 0x7000) + 0x0400 + (idx // 0x7000) * 0x8000).astype(np.uint16)


# ---------------------------------------------------------------------------
# Child entry: launch every probe (run under the profiler)
# ---------------------------------------------------------------------------


def run_probes(arch: str) -> list[str]:
    """Compile and launch all probes on device 0; return failure messages."""
    import numpy as np
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.packing import pack_args

    rt = Runtime()
    sig = [
        {"name": "src", "type": "ptr<f16, global>"},
        {"name": "out", "type": "ptr<f16, global>"},
        {"name": "reps", "type": "i32"},
        {"name": "zero", "type": "i32"},
    ]
    failures = []
    for tag, sw, lay, w in _probe_cases():
        src = _src_bits(lay.rows, lay.cols)
        img, frags = _expected(lay, w, src)
        for mode in ("w", "r"):
            kern = make_probe_kernel(mode, tag, lay, w)
            art = compile_kernel(kern, arch=arch, backend="python")
            mod = rt.load_module(art.hsaco)
            fn = mod.get_function(kern.name)
            host_in = src if mode == "w" else img
            want = img if mode == "w" else frags
            d_in = rt.alloc(host_in.nbytes)
            d_out = rt.alloc(want.nbytes)
            rt.memcpy_h2d(
                d_in,
                (ctypes.c_ubyte * host_in.nbytes).from_buffer_copy(host_in.tobytes()),
                host_in.nbytes,
            )
            rt.memset(d_out, 0, want.nbytes)
            args = pack_args(sig, {"src": d_in, "out": d_out, "reps": REPS, "zero": 0})
            rt.launch_blocking(fn, (GRID, 1, 1), (THREADS, 1, 1), args)
            buf = (ctypes.c_ubyte * want.nbytes)()
            rt.memcpy_d2h(buf, d_out, want.nbytes)
            got = np.frombuffer(bytes(buf), dtype=np.uint16)
            if not np.array_equal(got, want):
                bad = int(np.count_nonzero(got != want))
                failures.append(f"{kern.name}: {bad} of {want.size} elements differ")
            rt.free(d_in)
            rt.free(d_out)
    return failures


# ---------------------------------------------------------------------------
# Helpers for the tests
# ---------------------------------------------------------------------------


def _disassemble(hsaco: bytes, arch: str) -> str:
    """Disassembly via the stage-table test's tool discovery (skips if none works)."""
    from tests.test_attention_bwd_sched import _disassemble as disassemble

    return disassemble(hsaco, arch)


def _find_rocprofv3() -> str | None:
    found = shutil.which("rocprofv3")
    if found:
        return found
    hits = sorted(glob.glob("/opt/rocm*/bin/rocprofv3"))
    return hits[-1] if hits else None


def _predictions(arch: str) -> dict[str, int]:
    """Kernel name -> predicted worst multiplicity of the probed instruction."""
    out = {}
    for tag, sw, lay, w in _probe_cases():
        pred = xt_conflicts(lay, w, arch)
        out[_kernel_name("w", tag, sw)] = pred["write"]
        out[_kernel_name("r", tag, sw)] = pred["read"]
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_probe_kernels_use_the_intended_lds_opcodes():
    try:
        from rocke.helpers.compile import compile_kernel
    except ImportError as exc:
        pytest.skip(f"comgr unavailable: {exc}")
    pat = re.compile(r"^\s*(ds_(?:read|write|load|store)\S*)", re.MULTILINE)
    for tag, sw, lay, w in _probe_cases():
        for mode in ("w", "r"):
            kern = make_probe_kernel(mode, tag, lay, w)
            art = compile_kernel(kern, arch="gfx942", backend="python")
            txt = _disassemble(art.hsaco, "gfx942")
            ops = set(pat.findall(txt))
            if mode == "w":
                store = {4: "ds_write_b32", 8: "ds_write_b64"}[w.write_bytes]
                assert ops == {store, "ds_read_b128"}, (kern.name, ops)
            else:
                assert ops == {"ds_write_b128", "ds_read_b64"}, (kern.name, ops)


def test_probe_layout_correct_on_device():
    arch = device_arch()
    if arch not in ("gfx942", "gfx950"):
        pytest.skip(f"needs a gfx942 or gfx950 device (found {arch})")
    pytest.importorskip("numpy")
    failures = run_probes(arch)
    assert not failures, failures


def _parse_counter_csv(root: Path) -> dict[str, dict[str, float]]:
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    files = list(root.rglob("*counter_collection.csv"))
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                kname = row.get("Kernel_Name") or row.get("Kernel-Name") or ""
                cname = row.get("Counter_Name") or row.get("Counter-Name") or ""
                val = row.get("Counter_Value") or row.get("Counter-Value") or "0"
                m = re.search(r"(bwd_xt_probe_\w+)", kname)
                if m:
                    sums[m.group(1)][cname] += float(val)
    return sums


def test_conflict_counter_matches_predictor_on_device():
    arch = device_arch()
    if arch not in ("gfx942", "gfx950"):
        pytest.skip(f"needs a gfx942 or gfx950 device (found {arch})")
    prof = _find_rocprofv3()
    if prof is None:
        pytest.skip("rocprofv3 not found")
    pytest.importorskip("numpy")
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            prof,
            "--pmc",
            "SQ_LDS_BANK_CONFLICT",
            "SQ_LDS_IDX_ACTIVE",
            "--output-format",
            "csv",
            "-d",
            tmp,
            "-o",
            "xtprobe",
            "--",
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-probe",
            arch,
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=900, check=False
        )
        assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
        sums = _parse_counter_csv(Path(tmp))
    pred = _predictions(arch)
    record_path = os.environ.get("ROCKE_BWD_PROBE_RECORD")
    if record_path:
        rec = {
            "arch": arch,
            "method": "rocprofv3 --pmc SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE, per dispatch",
            "reps": REPS,
            "grid": GRID,
            "threads": THREADS,
            "counters": {k: dict(v) for k, v in sums.items()},
            "predicted_multiplicity": pred,
        }
        Path(record_path).write_text(
            json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8"
        )
    mismatches = []
    for kname, mult in sorted(pred.items()):
        assert kname in sums, f"no counters for {kname} (got {sorted(sums)})"
        conflict = sums[kname].get("SQ_LDS_BANK_CONFLICT", 0.0)
        active = sums[kname].get("SQ_LDS_IDX_ACTIVE", 0.0)
        assert active > 0, f"{kname}: LDS never active"
        verdict = "zero" if conflict == 0 else "nonzero"
        print(
            f"{kname}: predicted multiplicity {mult}, device conflict counter {verdict}"
        )
        if (mult == 1) != (conflict == 0):
            mismatches.append(f"{kname}: predicted {mult}, counter {verdict}")
    assert not mismatches, mismatches


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--run-probe":
        target = sys.argv[2] if len(sys.argv) > 2 else (device_arch() or "gfx942")
        errs = run_probes(target)
        for e in errs:
            print("PROBE_MISMATCH", e)
        sys.exit(3 if errs else 0)
    sys.exit(pytest.main([__file__, "-q"]))
