# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host data helpers shared by the parity harnesses.

The harnesses (``ck_tile_parity``, ``parity_extended_kernels`` and the
library's ``builders.common.parity_fmha_extended``) draw numpy inputs, store
them in the kernel's element type, launch on
:class:`~rocke.runtime.device_array.DeviceArray` buffers, and compare the
kernel's output values with a float64 reference computed from the stored
inputs. This module is the one home of those steps.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from rocke.numeric import storage
from rocke.numeric.compare import max_abs_error
from rocke.runtime.device_array import DeviceArray
from rocke.runtime.launcher import KernelLauncher, LaunchConfig, synchronize_and_release

__all__ = [
    "CaseUnavailable",
    "Result",
    "default_arch",
    "device_available",
    "device_values",
    "draw",
    "launch",
    "require_ocp_fp8_arch",
    "round_to_f16",
    "run_cases",
    "summarise",
    "values",
]


def draw(
    rng: np.random.Generator,
    *shape: int,
    dtype: str = "fp16",
    scale: float = 1.0,
    shift: float = 0.0,
) -> np.ndarray:
    """``shift + scale * N(0, 1)`` draws stored as ``dtype`` (a host storage
    array)."""
    return storage.encode(shift + scale * rng.standard_normal(shape), dtype)


def values(stored: np.ndarray, dtype: str) -> np.ndarray:
    """The float64 values of a ``dtype`` host storage array."""
    return storage.decode(stored, dtype).astype(np.float64)


def device_values(array: DeviceArray) -> np.ndarray:
    """The float64 values of a floating-point :class:`DeviceArray`."""
    return values(array.values(), array.dtype)


def round_to_f16(reference: np.ndarray) -> np.ndarray:
    """``reference`` rounded to fp16, as an fp16 kernel stores its output, in
    float64."""
    return storage.round_to(reference, "fp16").astype(np.float64)


class CaseUnavailable(Exception):
    """A case that cannot run in this environment, on any arch (:func:`run_cases`
    reports it as SKIP with this exception's message)."""


@dataclass
class Result:
    """One case's verdict: the kernel output against its reference."""

    name: str
    passed: bool
    max_abs_diff: float
    rel_max: float
    range_min: float
    range_max: float
    note: str = ""


def summarise(
    out: np.ndarray, ref: np.ndarray, *, tol: float, note: str = ""
) -> Result:
    """Compare the kernel output values ``out`` with the reference ``ref`` in
    float64: passed when ``max|out - ref| <= tol`` and ``out`` is not trivially
    zero where ``ref`` is not. The caller names the result."""
    out = np.asarray(out, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    max_d = max_abs_error(out, ref)
    ref_max = float(np.abs(ref).max())
    rel = max_d / (ref_max + 1e-9)
    out_min = float(out.min())
    out_max = float(out.max())
    trivially_zero = max(abs(out_min), abs(out_max)) < 0.001 and ref_max > 0.01
    return Result(
        name="",
        passed=not trivially_zero and max_d <= tol,
        max_abs_diff=max_d,
        rel_max=rel,
        range_min=out_min,
        range_max=out_max,
        note=(
            f"output is trivially zero (ref range ~{ref_max:.3f})"
            if trivially_zero
            else note
        ),
    )


def launch(
    launcher: KernelLauncher,
    args: Mapping[str, Any],
    *,
    grid: tuple[int, int, int],
    block: tuple[int, int, int] = (64, 1, 1),
) -> None:
    """Launch once and wait. The block defaults to one wave64; a kernel that
    distributes work at another granularity passes its own ``block``."""
    launcher(args, config=LaunchConfig(grid=grid, block=block))
    synchronize_and_release()


def default_arch() -> str:
    """The running device's gfx target, or ``gfx950`` without a device."""
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch() or "gfx950"
    except Exception:  # noqa: BLE001 - no device / import issue: fall back
        return "gfx950"


def device_available() -> bool:
    """Whether the rocKE runtime sees a HIP device."""
    from rocke.runtime.hip_module import get_device_arch

    return bool(get_device_arch())


def require_ocp_fp8_arch(case: str, arch: str) -> None:
    """Refuse an OCP-fp8 (e4m3fn) parity case off gfx950 (a SKIP there).

    On gfx950 ``cvt_f32_fp8`` decodes the byte as OCP e4m3fn, matching
    :func:`rocke.numeric.fp8.fp8e4m3_decode_table` bit for bit; on gfx942 the
    same intrinsic decodes it as legacy ``e4m3fnuz`` (bias 8, 0x80 == NaN), so
    hardware and the OCP reference disagree. The kernel builds and runs on
    gfx942; only the byte format differs, so the OCP-reference check is
    gfx950-only.
    """
    if arch != "gfx950":
        raise NotImplementedError(
            f"{case}: OCP fp8e4m3fn dequant parity is gfx950-only; gfx942 "
            "cvt_f32_fp8 decodes bytes as legacy e4m3fnuz (bias 8, 0x80=NaN), "
            "which does not match the OCP e4m3 reference"
        )


def _exception_result(name: str, note: str) -> Result:
    traceback.print_exc()
    return Result(
        name=name,
        passed=False,
        max_abs_diff=float("inf"),
        rel_max=float("inf"),
        range_min=0.0,
        range_max=0.0,
        note=note,
    )


def run_cases(cases: list[tuple[str, Callable[[], Result]]], *, arch: str) -> int:
    """Run ``cases`` built for ``arch``, print each result and a summary, and
    return the exit code: 0 when every case that ran passed, else 1.

    A case that raises :class:`CaseUnavailable` is a SKIP on any arch. A
    ``ValueError`` / ``NotImplementedError`` is a SKIP off gfx950 (a builder
    refusing a gfx950-only atom) and a FAIL on gfx950, where builders never
    raise them.
    """
    results: list[Result] = []
    skipped: list[tuple[str, str]] = []
    for name, fn in cases:
        print(f"\n=== {name} ===")
        try:
            r = fn()
            results.append(r)
            print(f"  {r.name}")
            print(
                f"    max_abs={r.max_abs_diff:.4g}  rel={r.rel_max:.4g}  "
                f"range=({r.range_min:.4f}, {r.range_max:.4f})  "
                f"{'PASS' if r.passed else 'FAIL'}"
                f"{('  ' + r.note) if r.note else ''}"
            )
        except CaseUnavailable as exc:
            print(f"  SKIP {name}: {exc}")
            skipped.append((name, str(exc)))
        except (ValueError, NotImplementedError) as exc:
            if arch == "gfx950":
                results.append(
                    _exception_result(name, f"EXCEPTION (unexpected on gfx950): {exc}")
                )
            else:
                print(f"  SKIP {name} (gfx950-only on {arch}): {exc}")
                skipped.append((name, f"gfx950-only on {arch}"))
        except Exception as exc:  # noqa: BLE001 - reported as a FAIL
            results.append(_exception_result(name, f"EXCEPTION: {exc}"))

    print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    for r in results:
        tag = "PASS" if r.passed else "FAIL"
        print(f"  {tag}  {r.name:50s} max_abs={r.max_abs_diff:.4g}")
    for name, reason in skipped:
        print(f"  SKIP  {name:50s} ({reason})")
    n_pass = sum(1 for r in results if r.passed)
    n_run = len(results)
    n_skip = len(skipped)
    print(
        f"\narch={arch}: {n_pass}/{n_run} pass"
        + (f", {n_skip} skipped" if n_skip else "")
    )
    # Skipped cases do not fail the gate; every case that ran must pass.
    return 0 if n_pass == n_run else 1
