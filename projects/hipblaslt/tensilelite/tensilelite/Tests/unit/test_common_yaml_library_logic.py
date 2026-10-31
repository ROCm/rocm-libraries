################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
"""Guard common and custom test YAMLs against unneeded LibraryLogic/LibraryClient steps."""

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml


pytestmark = pytest.mark.unit

_TESTS_DIR = Path(__file__).resolve().parent.parent
_SCANNED_DIRS = (_TESTS_DIR / "common", _TESTS_DIR / "custom")
_LIBRARY_SECTIONS = ("LibraryLogic", "LibraryClient")


@dataclass(frozen=True)
class LibrarySectionsAllowed:
    path: str
    reason: str

    def __post_init__(self):
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError(
                f"LibraryLogic/LibraryClient whitelist entry for {self.path} requires a reason"
            )


_DUCTILE_REASON = "Ductile backend config: LibraryLogic writes a logic file from the GA results"
_MXFP8_REASON = "MXFP8 config: Prediction LibraryLogic and LibraryClient (Origami selection)"
_CUSTOM_REASON = "custom kernel config: LibraryLogic writes a GridBased logic file"

_LIBRARY_SECTIONS_ALLOWED = {
    LibrarySectionsAllowed("common/ductile/gfx1250/bbs_tn_dp.yaml", _DUCTILE_REASON),
    LibrarySectionsAllowed(
        "common/ductile/gfx1250/bbs_tn_multi_size_single_kernel.yaml", _DUCTILE_REASON
    ),
    LibrarySectionsAllowed("common/ductile/gfx1250/bbs_tn_streamK.yaml", _DUCTILE_REASON),
    LibrarySectionsAllowed("common/ductile/gfx1250/f32_tt_batched.yaml", _DUCTILE_REASON),
    LibrarySectionsAllowed("common/ductile/gfx1250/f8_tn_mx.yaml", _DUCTILE_REASON),
    LibrarySectionsAllowed("common/ductile/gfx1250/hhs_nt_multi_size.yaml", _DUCTILE_REASON),
    LibrarySectionsAllowed("common/ductile/gfx950/bf16_tn_dp.yaml", _DUCTILE_REASON),
    LibrarySectionsAllowed(
        "common/ductile/gfx950/bf16_tn_multi_size_single_kernel.yaml", _DUCTILE_REASON
    ),
    LibrarySectionsAllowed("common/ductile/gfx950/bf16_tn_streamK.yaml", _DUCTILE_REASON),
    LibrarySectionsAllowed("common/ductile/gfx950/f16_nt_multi_size.yaml", _DUCTILE_REASON),
    LibrarySectionsAllowed("common/ductile/gfx950/f32_tt_batched.yaml", _DUCTILE_REASON),
    LibrarySectionsAllowed("common/ductile/gfx950/f8_tn_mx_subtile.yaml", _DUCTILE_REASON),
    LibrarySectionsAllowed("common/ductile/gfx950/f8bf16_tn_cms.yaml", _DUCTILE_REASON),
    LibrarySectionsAllowed("common/ductile/gfx950/xf32_nn.yaml", _DUCTILE_REASON),
    LibrarySectionsAllowed(
        "common/gemm/fp8nfp16mix_hfp8ns.yaml",
        "LibraryLogic built from real benchmark data (mixed FP8/F16 inputs)",
    ),
    LibrarySectionsAllowed(
        "common/gemm/gfx950/mxfp8_residual_fail_classes_repro.yaml", _MXFP8_REASON
    ),
    LibrarySectionsAllowed("common/gemm/gfx950/mxfp8_verify_fail_repro.yaml", _MXFP8_REASON),
    LibrarySectionsAllowed(
        "common/gemm/gfx950/prediction_bf16.yaml",
        "only end-to-end Prediction coverage: logic write, library build, Origami selection",
    ),
    LibrarySectionsAllowed(
        "common/gemm/gfx950/subtile_mxfp8_correctness_regression.yaml", _MXFP8_REASON
    ),
    LibrarySectionsAllowed(
        "common/gemm/gfx950/subtile_mxfp8_correctness_regression_mt192x256_plr1_wgm4.yaml",
        _MXFP8_REASON,
    ),
    LibrarySectionsAllowed(
        "common/gemm/gfx950/subtile_mxfp8_correctness_regression_mt256x192_plr1_wgm1.yaml",
        _MXFP8_REASON,
    ),
    LibrarySectionsAllowed(
        "common/gemm/gfx950/subtile_mxfp8_correctness_regression_mt256x320_plr0_wgm1.yaml",
        _MXFP8_REASON,
    ),
    LibrarySectionsAllowed(
        "common/gemm/gfx950/subtile_mxfp8_correctness_regression_mt320x256_plr1_wgm4.yaml",
        _MXFP8_REASON,
    ),
    LibrarySectionsAllowed(
        "common/gemm/gfx950/subtile_mxfp8_multi_du_multi_partition.yaml", _MXFP8_REASON
    ),
    LibrarySectionsAllowed("common/gemm/gfx950/subtile_mxfp8_tail_loop.yaml", _MXFP8_REASON),
    LibrarySectionsAllowed("common/gemm/gfx950/subtile_mxfp8_tail_loop_smoke.yaml", _MXFP8_REASON),
    LibrarySectionsAllowed(
        "common/gemm/xfp32.yaml",
        "GridBased LibraryLogic and LibraryClient built from real benchmark data (XF32)",
    ),
    LibrarySectionsAllowed(
        "common/groupedgemm/gfx11/grouped_gemm_gfx11.yaml",
        "grouped GEMM LibraryLogic built from real benchmark data (gfx11)",
    ),
    LibrarySectionsAllowed(
        "common/groupedgemm/grouped_gemm.yaml",
        "grouped GEMM FreeSize LibraryLogic built from real benchmark data",
    ),
    LibrarySectionsAllowed(
        "common/groupedgemm/grouped_gemm_ck_gfx942.yaml",
        "grouped GEMM LibraryLogic built from real benchmark data (gfx942)",
    ),
    LibrarySectionsAllowed(
        "common/sparse/gfx94x/libray_logic.yaml",
        "sparse GEMM LibraryLogic built from real benchmark data",
    ),
    LibrarySectionsAllowed("custom/custom_aiter_bf16.yaml", _CUSTOM_REASON),
    LibrarySectionsAllowed("custom/custom_aiter_f4.yaml", _CUSTOM_REASON),
    LibrarySectionsAllowed("custom/custom_ck.yaml", _CUSTOM_REASON),
    LibrarySectionsAllowed("custom/custom_rocblas_gemv_bf16_tn_m1.yaml", _CUSTOM_REASON),
    LibrarySectionsAllowed("custom/custom_rocblas_gemv_bf16_tn_m2.yaml", _CUSTOM_REASON),
    LibrarySectionsAllowed("custom/custom_rocblas_gemv_bf16_tn_m4.yaml", _CUSTOM_REASON),
    LibrarySectionsAllowed("custom/custom_rr.yaml", _CUSTOM_REASON),
    LibrarySectionsAllowed("custom/custom_rr_f6.yaml", _CUSTOM_REASON),
    LibrarySectionsAllowed("custom/custom_rr_mx.yaml", _CUSTOM_REASON),
    LibrarySectionsAllowed("custom/custom_tensile_sk.yaml", _CUSTOM_REASON),
    LibrarySectionsAllowed("custom/custom_triton_f4.yaml", _CUSTOM_REASON),
    LibrarySectionsAllowed("custom/custom_wave_bf16.yaml", _CUSTOM_REASON),
}
_LIBRARY_SECTIONS_ALLOWED_PATHS = {entry.path for entry in _LIBRARY_SECTIONS_ALLOWED}


def _scanned_yaml_files():
    return sorted({
        path
        for directory in _SCANNED_DIRS
        for pattern in ("*.yaml", "*.yml")
        for path in directory.rglob(pattern)
    })


def _relative_to_tests(path):
    return path.relative_to(_TESTS_DIR).as_posix()


def _library_sections(path):
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return []

    return [section for section in _LIBRARY_SECTIONS if section in data]


def test_library_sections_only_in_whitelisted_yaml_files():
    """LibraryLogic/LibraryClient steps cost run time; only listed YAMLs may have them."""
    stale_entries = []
    for entry in sorted(_LIBRARY_SECTIONS_ALLOWED, key=lambda entry: entry.path):
        yaml_path = _TESTS_DIR / entry.path
        if not yaml_path.is_file():
            stale_entries.append(f"{entry.path} (file does not exist; {entry.reason})")
        elif not _library_sections(yaml_path):
            stale_entries.append(
                f"{entry.path} (no LibraryLogic/LibraryClient section; {entry.reason})"
            )

    failures = []
    for yaml_path in _scanned_yaml_files():
        rel_path = _relative_to_tests(yaml_path)
        if rel_path in _LIBRARY_SECTIONS_ALLOWED_PATHS:
            continue

        sections = _library_sections(yaml_path)
        if sections:
            failures.append(f"{rel_path} ({', '.join(sections)})")

    if stale_entries or failures:
        message_parts = []
        if stale_entries:
            message_parts.append(
                "Stale LibraryLogic/LibraryClient whitelist entries:\n"
                + "\n".join(f"  - {entry}" for entry in stale_entries)
            )
        if failures:
            message_parts.append(
                "Only whitelisted common/custom YAML files may have a top-level "
                "LibraryLogic or LibraryClient section (add an entry with a reason "
                "if the test needs the library path). Failing paths:\n"
                + "\n".join(f"  - {failure}" for failure in failures)
            )
        pytest.fail("\n\n".join(message_parts))
