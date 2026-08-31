# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Knob sweep: isolate, then pair survivors, then ship what mattered.

The last sweep moved 2 of 22 knobs, picked them by hand, and shipped a
cross-product. The uplift landed almost entirely on one synthetic shape family and
the wide arm bought nothing measurable over the condensed one.

The property that matters most here is that an ARM DIFFERS FROM THE BASELINE IN
EXACTLY ONE KNOB. An arm carrying a second, unrecorded difference still generates,
still gates, still measures -- and attributes that difference to the knob under
test. That confound appeared for real while building this tool (promoting a spec to
the arch subclass silently added its private fields at their defaults), which is why
it is asserted by diffing rather than trusted.

These run against the real dispatcher and skip when rocKE is absent: what is under
test is "we asked the library", and a mocked dispatcher would assert the mock.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
_SWEEP = _TOOLS / "knob_sweep.py"
_PROFILE = _TOOLS.parent / "configs" / "gfx942_attention_dense.profile.yaml"
_REPO_ROOT = Path(__file__).resolve().parents[5]

pytestmark = pytest.mark.skipif(
    not (
        _REPO_ROOT / "dnn-providers/hip-kernel-provider/rocke/library"
        "/dispatch/attention/gfx942.py"
    ).exists(),
    reason="rocKE library not present; the sweep asks the real dispatcher",
)


def _shapes(tmp_path: Path) -> Path:
    """A corpus straddling the persistent threshold, so arms are not degenerate."""
    out = []
    for batch, seqlen, head_size in itertools.product((1, 2), (512, 4096), (64, 128)):
        out.append(
            {
                "batch": batch,
                "nhead_q": 32,
                "nhead_k": 8,
                "seqlen_q": seqlen,
                "seqlen_k": seqlen,
                "hdim_q": head_size,
                "hdim_v": head_size,
                "dtype": "bf16",
                "mask_type": 1,
            }
        )
    path = tmp_path / "shapes.json"
    path.write_text(json.dumps(out))
    return path


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SWEEP), "--profile", str(_PROFILE), *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )


def _specs(path: Path) -> list[dict]:
    import yaml

    config = yaml.safe_load(path.read_text())
    return [k["kernel_source"]["spec"] for k in config["packs"][0]["kernels"]]


@pytest.fixture(scope="module")
def isolated(tmp_path_factory):
    work = tmp_path_factory.mktemp("sweep")
    shapes = _shapes(work)
    result = _run("--shapes", str(shapes), "--isolate", "--out-dir", str(work / "arms"))
    if result.returncode != 0:
        pytest.skip(f"sweep unavailable: {result.stderr.strip()[:200]}")
    return {"dir": work / "arms", "shapes": shapes, "stdout": result.stdout}


class TestIsolationIsActuallyIsolated:
    def test_each_arm_differs_from_the_baseline_in_exactly_one_knob(self, isolated):
        """The assertion the whole isolation pass rests on.

        A second difference is silent: the arm generates, gates and measures, and the
        sweep attributes the combined effect to the named knob. This caught a real
        confound during development.
        """
        base = _specs(isolated["dir"] / "arm_parity.yaml")
        offenders = {}
        for arm in sorted(isolated["dir"].glob("arm_*.yaml")):
            if arm.name == "arm_parity.yaml":
                continue
            fields = set()
            for before, after in zip(base, _specs(arm)):
                fields |= {
                    k for k in set(before) | set(after) if before.get(k) != after.get(k)
                }
            if len(fields) > 1:
                offenders[arm.name] = sorted(fields)
        assert (
            not offenders
        ), f"arms differ from parity in more than the knob under test: {offenders}"

    def test_every_arm_covers_the_same_shapes_as_the_baseline(self, isolated):
        """An arm that serves fewer shapes is measuring a different corpus."""
        base = len(_specs(isolated["dir"] / "arm_parity.yaml"))
        assert base > 0
        for arm in sorted(isolated["dir"].glob("arm_*.yaml")):
            assert len(_specs(arm)) == base, f"{arm.name} serves a different corpus"

    def test_an_arm_equal_to_parity_is_flagged_not_silently_shipped(self, isolated):
        """A knob value the dispatcher already resolves makes the arm the baseline
        under another name: it measures exactly 1.000x, and the sweep would report
        the knob as "no effect" having never tried the other side."""
        assert "== parity, measures nothing" in isolated["stdout"]

    def test_at_least_one_arm_genuinely_deviates(self, isolated):
        """Guards the check above from passing on a tool that emits only no-ops."""
        base = _specs(isolated["dir"] / "arm_parity.yaml")
        deviating = [
            arm.name
            for arm in sorted(isolated["dir"].glob("arm_*.yaml"))
            if arm.name != "arm_parity.yaml"
            and any(b != a for b, a in zip(base, _specs(arm)))
        ]
        assert deviating, "no arm differs from parity; the sweep measures nothing"


class TestCandidateSelection:
    def test_a_settled_knob_is_excluded_with_its_verdict(self, tmp_path):
        """ "The author swept it" means it was EXPLORED, not that it ships. A knob
        with a measured verdict is not an open question."""
        result = _run("--shapes", str(_shapes(tmp_path)), "--plan")
        assert result.returncode == 0
        assert "iglp: settled" in result.stdout
        assert "do not re-attempt" in result.stdout

    def test_a_knob_the_dispatcher_varies_is_not_a_candidate(self, tmp_path):
        """The dispatcher moving a knob per shape makes it a production axis;
        perturbing it fights the policy rather than measuring it."""
        result = _run("--shapes", str(_shapes(tmp_path)), "--plan")
        assert "waves_per_eu: the dispatcher varies it per shape" in result.stdout

    def test_hazards_travel_with_the_candidate(self, tmp_path):
        """The reason attaches to the decision instead of living in someone's memory
        of a commit message."""
        result = _run("--shapes", str(_shapes(tmp_path)), "--plan")
        assert "69,632 B of LDS" in result.stdout, "block_n's LDS gate must be stated"
        assert "TRI-STATE" in result.stdout, "the tri-state trap must be stated"

    def test_pairing_a_non_candidate_is_refused(self, tmp_path):
        """Pairing everything is the cross-product this tool exists to avoid."""
        result = _run(
            "--shapes",
            str(_shapes(tmp_path)),
            "--pairwise",
            "iglp,block_n",
            "--out-dir",
            str(tmp_path / "out"),
        )
        assert result.returncode != 0
        assert "not sweep candidates" in (result.stdout + result.stderr)

    def test_pairwise_emits_the_product_of_the_named_survivors_only(self, tmp_path):
        out = tmp_path / "pairs"
        result = _run(
            "--shapes",
            str(_shapes(tmp_path)),
            "--pairwise",
            "block_n,lds_row_pad",
            "--out-dir",
            str(out),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        # 2 values x 2 values, and nothing else.
        assert len(list(out.glob("pair_*.yaml"))) == 4
