# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""End-to-end tests for example/ck_tile/dsl/<N>_*.

Each example carries a gen.py (builds the HSACO + manifest.json) and
an expected.json that pins the correctness + TFLOPS lower-bound gate.
This module walks the directory at test discovery time, runs gen.py
in a subprocess, and (when a HIP runtime is available) launches the
kernel through the shared `ck_dsl_launcher` binary to verify it.

Skipping rules:

  - If hipcc / a working HIP runtime is not present, we still run the
    gen.py + IR + LLVM lowering stages (those are the static parts) and
    leave runtime verification as a skip.
  - If we cannot acquire GPU access, the runtime stage skips with a
    descriptive reason; the IR-build stage still runs and gates on
    HSACO output, which is the only thing the build system needs.

These tests are slow-ish (each example pays a cold libamd_comgr load
in its subprocess). We mark them with the `examples` attribute so a
caller can run only the fast unit tests via:

    python -m unittest discover -s python/test -t . -p 'test_ck_dsl.py'

while still running the examples on demand via:

    python -m unittest python.test.test_ck_dsl_examples
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
PY_PKG = ROOT / "python"
DSL_EXAMPLES = ROOT / "example" / "ck_tile" / "dsl"
LAUNCHER_CACHE = Path(os.environ.get("CK_DSL_LAUNCHER", "/tmp/ck_dsl_launcher"))


def _have_hipcc() -> bool:
    return shutil.which("hipcc") is not None


def _build_launcher() -> Optional[Path]:
    """Legacy C++ launcher builder.

    Tests use the Python-native `ck_dsl.run_manifest` runner now. This
    helper remains for ad-hoc compatibility but is no longer called by
    the example harness.
    """
    if LAUNCHER_CACHE.exists():
        return LAUNCHER_CACHE
    if not _have_hipcc():
        return None
    src = DSL_EXAMPLES / "common" / "launcher.cpp"
    cmd = [
        "hipcc",
        "-std=c++20",
        "-O3",
        "--offload-arch=gfx950",
        str(src),
        "-o",
        str(LAUNCHER_CACHE),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return None
    return LAUNCHER_CACHE


def _gen_example(example_dir: Path, out_dir: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PY_PKG) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(example_dir / "gen.py"), "--output-dir", str(out_dir)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"gen.py failed for {example_dir.name}: {r.stderr or r.stdout}"
        )
    return json.loads((out_dir / "manifest.json").read_text())


def _run_launcher(
    launcher: Optional[Path], hsaco: Path, manifest: Path, shape: str, verify: bool
) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PY_PKG) + os.pathsep + env.get("PYTHONPATH", "")
    args = [
        sys.executable,
        "-m",
        "ck_dsl.run_manifest",
        str(hsaco),
        str(manifest),
        "--shape",
        shape,
    ]
    if verify:
        args.append("--verify")
    sudo = shutil.which("sudo")
    if sudo:
        args = [sudo, "-n", "env", f"PYTHONPATH={env['PYTHONPATH']}"] + args
    r = subprocess.run(args, env=env, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(
            f"launcher failed: rc={r.returncode}\nstdout:{r.stdout}\nstderr:{r.stderr}"
        )
    # The verify summary goes to stderr; the Perf line goes to stdout.
    return r.stdout + "\n" + r.stderr


def _parse_tflops(stdout: str) -> Optional[float]:
    for line in stdout.splitlines():
        if line.startswith("Perf:"):
            for tok in line.split(","):
                tok = tok.strip()
                if tok.endswith(" TFlops"):
                    return float(tok.removesuffix(" TFlops").strip())
    return None


def _parse_gbps(stdout: str) -> Optional[float]:
    for line in stdout.splitlines():
        if line.startswith("Perf:"):
            for tok in line.split(","):
                tok = tok.strip()
                if tok.endswith(" GB/s"):
                    return float(tok.removesuffix(" GB/s").strip())
    return None


def _all_examples() -> list[Path]:
    if not DSL_EXAMPLES.exists():
        return []
    return sorted(
        p
        for p in DSL_EXAMPLES.iterdir()
        if p.is_dir() and (p / "gen.py").exists() and (p / "expected.json").exists()
    )


class TestCkDslExamples(unittest.TestCase):
    """One subtest per DSL example, gated by expected.json."""

    def test_examples(self):
        examples = _all_examples()
        if not examples:
            self.skipTest("no DSL examples found")

        launcher = None
        runtime_ok = True

        for ex in examples:
            with self.subTest(example=ex.name):
                expected = json.loads((ex / "expected.json").read_text())
                with tempfile.TemporaryDirectory() as tmp:
                    out = Path(tmp)
                    manifest = _gen_example(ex, out)
                    hsacos = list(out.glob("*.hsaco"))
                    self.assertEqual(len(hsacos), 1, f"{ex.name}: missing hsaco")
                    hs = hsacos[0]
                    self.assertGreater(
                        manifest["hsaco_bytes"],
                        0,
                        f"{ex.name}: empty hsaco",
                    )
                    if not runtime_ok:
                        self.skipTest(
                            f"{ex.name}: HIP launcher unavailable, "
                            "ran static stages only"
                        )
                    # `kind`-aware verification. GEMM examples are
                    # bit-exact on rounded inputs at small shapes; conv
                    # examples have inherent fp16 sum noise and rely on
                    # the launcher's `bad=0/N` exit (1e-2 tolerance vs
                    # fp32 CPU ref).
                    kind = expected.get("kind", manifest.get("kind", "gemm_fp16"))
                    if kind in ("gemm_fp16", "batched_gemm_fp16"):
                        # Correctness on small shape.
                        verify_shape = "256,256,256" if kind == "gemm_fp16" else "0,0,0"
                        out_verify = _run_launcher(
                            launcher,
                            hs,
                            out / "manifest.json",
                            shape=verify_shape,
                            verify=True,
                        )
                        self.assertIn(
                            "max_abs_diff=0",
                            out_verify,
                            f"{ex.name}: not bit-exact",
                        )
                        # Performance gate on production shape.
                        if kind == "gemm_fp16":
                            shape = expected["shapes"][0]
                            perf_shape = f"{shape['M']},{shape['N']},{shape['K']}"
                        else:
                            shape = None
                            perf_shape = "0,0,0"
                        perf = _run_launcher(
                            launcher,
                            hs,
                            out / "manifest.json",
                            shape=perf_shape,
                            verify=False,
                        )
                        tflops = _parse_tflops(perf)
                        self.assertIsNotNone(tflops, f"{ex.name}: no TFLOPS line")
                        lb = (
                            float(shape["tflops_lower_bound"])
                            if shape is not None
                            else float(expected.get("min_tflops", 0.0))
                        )
                        self.assertGreaterEqual(
                            tflops,
                            lb,
                            f"{ex.name}: {tflops:.2f} TFLOPS < lower bound {lb}",
                        )
                    elif kind == "conv_fp16":
                        # Correctness: launcher verifies vs CPU ref and
                        # exits non-zero on `bad > 0` (the 1e-2 absolute
                        # threshold). _run_launcher raises on non-zero,
                        # so reaching here means verify passed.
                        _run_launcher(
                            launcher,
                            hs,
                            out / "manifest.json",
                            shape="0,0,0",
                            verify=True,
                        )
                        tflops_runs = []
                        for _ in range(3):
                            perf = _run_launcher(
                                launcher,
                                hs,
                                out / "manifest.json",
                                shape="0,0,0",
                                verify=False,
                            )
                            t = _parse_tflops(perf)
                            self.assertIsNotNone(t, f"{ex.name}: no TFLOPS line")
                            tflops_runs.append(t)
                        tflops = statistics.median(tflops_runs)
                        if "min_tflops" in expected:
                            lb = float(expected["min_tflops"])
                            self.assertGreaterEqual(
                                tflops,
                                lb,
                                f"{ex.name}: median {tflops:.2f} TFLOPS < lower bound {lb}; "
                                f"runs={tflops_runs}",
                            )
                    elif kind in {
                        "elementwise_fp16",
                        "reduce_fp16",
                        "layernorm_fp16",
                        "rmsnorm_fp16",
                        "transpose_fp16",
                    }:
                        # Generic simple-op kernels use their manifest's
                        # default shape and the Python-native runner's
                        # numpy reference. The verify gate is authoritative:
                        # reaching here means max_abs/bad is within that
                        # op family's tolerance.
                        _run_launcher(
                            launcher,
                            hs,
                            out / "manifest.json",
                            shape="0,0,0",
                            verify=True,
                        )
                        gbps_runs = []
                        for _ in range(3):
                            perf = _run_launcher(
                                launcher,
                                hs,
                                out / "manifest.json",
                                shape="0,0,0",
                                verify=False,
                            )
                            gbps = _parse_gbps(perf)
                            self.assertIsNotNone(gbps, f"{ex.name}: no GB/s line")
                            gbps_runs.append(gbps)
                        if "min_gbps" in expected:
                            lb = float(expected["min_gbps"])
                            med = statistics.median(gbps_runs)
                            self.assertGreaterEqual(
                                med,
                                lb,
                                f"{ex.name}: median {med:.2f} GB/s < lower bound {lb}; "
                                f"runs={gbps_runs}",
                            )
                    else:
                        self.fail(f"{ex.name}: unknown kind {kind!r}")


if __name__ == "__main__":
    unittest.main()
