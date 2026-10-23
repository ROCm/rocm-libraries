# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The on-device BF16x3 numeric gate, run by the suite (XF32 plan, step 9).

``rocke/examples/gfx950/xf32/bf16x3_parity.py`` is the only check that runs the
expansion on silicon: bitwise against a hand-written BF16x3 kernel, and for
accuracy against ``K * alpha * beta``. As an examples-only ``__main__`` script
neither pytest nor ``tests/run_all.py`` would ever invoke it, so the device
assertions would not run even on a gfx950 host. This test runs it there.

**Every visible device is examined**, not just device 0, and the first gfx950
among them is handed to the harness through the visibility env var. A node
whose gfx950 sits at index 3 behind two other cards would otherwise skip the
only gate that exercises the emulation.

It skips only on a *confirmed* absence, of which there are three shapes:

- the HIP library will not load, so there is no ROCm on this host;
- ``hipGetDeviceCount`` answers ``hipErrorNoDevice``, which is a real answer:
  the runtime looked and there is no GPU (what a host with ROCm installed but
  an unusable ``/dev/kfd`` reports);
- it answers success, and none of the devices it counted is a gfx950 -- the
  emulated rows being gfx950-only.

Anything else fails, because it establishes nothing about what hardware is
present: any other error code, an import error, or -- when no gfx950 was found
-- a device whose architecture HIP would not name, since that device could be
the one this gate exists for.

Which is why the probe cannot use ``get_device_count()`` / ``get_device_arch()``
as they stand. Those are deliberately forgiving: both flatten *every* failure
into the answer a healthy query gives for an empty machine, ``0`` and ``None``,
collapsing two of the cases above together with all of the ones that must fail.
``get_device_count`` also raises ``HipError`` when the library cannot be loaded
-- the one case that should be a quiet skip. So the probe resolves the runtime
itself and reads ``hipGetDeviceCount``'s own return code, which keeps the
outcomes apart.

``ROCKE_REQUIRE_GFX950=1`` turns even the confirmed-absence skip into a
failure, so a validation node that lost its device cannot report green.

The device probe and the harness each run in a subprocess, which keeps HIP
initialisation out of the pytest process and bounds a hung device.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

_ARCH = "gfx950"
_REQUIRE_ENV = "ROCKE_REQUIRE_GFX950"
# The harness is run as a module, so the only thing that has to be found is the
# package -- not where the examples directory happens to sit relative to this file.
_HARNESS_MODULE = "rocke.examples.gfx950.xf32.bf16x3_parity"


def _import_root() -> Path:
    """The directory that makes ``import rocke`` resolve, wherever that is.

    ``python/`` in a source checkout; ``bin/<subdir>/`` in an installed provider,
    where the package sits next to ``tests/`` and there is no ``python/`` at all.
    Counting path components up from this file only ever described the first, so
    the harness was looked for at a path that does not exist once installed.
    Asking the import system also guarantees the child processes below import the
    same ``rocke`` this suite is testing.
    """
    spec = importlib.util.find_spec("rocke")
    if spec is None or not spec.submodule_search_locations:
        # Not importable from here -- a bare checkout run without PYTHONPATH --
        # and the source layout is the only one that can be.
        return Path(__file__).resolve().parents[2] / "python"
    return Path(next(iter(spec.submodule_search_locations))).resolve().parent


# Prints `NO_HIP <reason>` when the runtime cannot be loaded at all, otherwise
# one `DEVICE <index> <arch>` line per visible device, with `?` for an
# architecture HIP would not report. Any other failure -- including a
# `hipGetDeviceCount` that returns an error -- exits non-zero, which the caller
# turns into a test failure rather than a skip.
_NO_HIP = "NO_HIP"
_UNKNOWN_ARCH = "?"
# hipErrorNoDevice. The one error code that is still an answer: the runtime
# looked and there is no GPU here (this is what a host with ROCm installed but
# no usable /dev/kfd reports). Every other code means the query did not
# complete, which proves nothing either way.
_HIP_ERROR_NO_DEVICE = 100
_PROBE = f"""
import ctypes
from rocke.runtime import hip_module as hip

try:
    hip._resolve_hip()
except hip.HipError as exc:
    print("{_NO_HIP}", exc)
    raise SystemExit(0)

count = ctypes.c_int(0)
rc = hip._hipGetDeviceCount(ctypes.byref(count))
if rc == {_HIP_ERROR_NO_DEVICE}:
    raise SystemExit(0)  # hipErrorNoDevice: answered, and the answer is none
if rc != 0:
    raise SystemExit(f"hipGetDeviceCount failed: hipError({{rc}})")

for i in range(count.value):
    print("DEVICE", i, hip.get_device_arch(i) or "{_UNKNOWN_ARCH}")
"""

# HIP reads the first of these that is set; the chosen index must be resolved
# against that same list, since the indices a filtered process sees are
# positions within it rather than physical device numbers.
_VISIBILITY_VARS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_import_root()), env.get("PYTHONPATH")) if p
    )
    return env


def _visible_devices(env: dict):
    """``[(index, arch)]`` for every visible device, or ``None`` if HIP is absent.

    ``None`` is a confirmed "there is no HIP runtime here" rather than an empty
    list, which is a successful count of zero devices. An architecture HIP
    would not report comes back as ``_UNKNOWN_ARCH``; the caller decides what
    that means. Raises :class:`AssertionError` if the probe itself fails -- see
    the module docstring on why that is not a skip.
    """
    probe = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if probe.returncode != 0:
        raise AssertionError(
            "the HIP device probe failed, so device presence is unknown and a "
            f"skip would be unsound (exit {probe.returncode}):\n"
            f"{(probe.stderr or probe.stdout).strip()}"
        )
    found = []
    for line in probe.stdout.strip().splitlines():
        kind, _, rest = line.partition(" ")
        if kind == _NO_HIP:
            return None
        if kind != "DEVICE":
            raise AssertionError(f"unreadable device probe output: {line!r}")
        index, _, arch = rest.partition(" ")
        found.append((int(index), arch.strip()))
    return found


def _select(env: dict, index: int) -> dict:
    """``env`` with the chosen device made device 0 for the child process."""
    for var in _VISIBILITY_VARS:
        current = env.get(var)
        if current:
            # Already filtered: `index` is a position in this list, not a
            # device number, so hand back the entry rather than the index.
            entries = [e for e in current.split(",") if e.strip()]
            return dict(env, **{var: entries[index]})
    return dict(env, HIP_VISIBLE_DEVICES=str(index))


class TestBf16x3OnDevice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = _env()
        visible = _visible_devices(env)
        for index, arch in visible or ():
            if arch == _ARCH:
                cls.env = _select(env, index)
                return
        if visible is None:
            reason = "no HIP runtime on this host"
        else:
            # Only now does an unreadable architecture matter: one of these
            # devices may be the gfx950 this gate exists for, so concluding
            # absence from them would be a guess.
            unknown = [i for i, arch in visible if arch == _UNKNOWN_ARCH]
            if unknown:
                raise AssertionError(
                    f"HIP reported {len(visible)} device(s) but would not name "
                    f"the architecture of {unknown}, so the absence of a "
                    f"{_ARCH} is unproven and a skip would be unsound"
                )
            seen = ", ".join(f"{i}:{a}" for i, a in visible) or "none"
            reason = f"no visible {_ARCH} device (visible: {seen})"
        if env.get(_REQUIRE_ENV) == "1":
            raise AssertionError(f"{_REQUIRE_ENV}=1, but {reason}")
        raise unittest.SkipTest(reason)

    def test_the_expansion_is_faithful_and_accurate_on_device(self):
        # --require-device: the probe above saw a gfx950 and this process
        # selected it, so a harness that cannot find one is broken and must
        # not exit 0 as a skip.
        done = subprocess.run(
            [sys.executable, "-m", _HARNESS_MODULE, "--require-device"],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("PASSED", done.stdout)


if __name__ == "__main__":
    unittest.main()
