# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The informational provenance a generation stamps: ``provenance.generated_from`` on
each KDP and ``provenance.rocke_*`` on each kernel. Each case is one thing the stamp
must not do: leak into a render that did not ask for it, or turn into a difference
`catalog_diff` reports."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from codegen import provenance
from codegen.config_loader import ConfigError, load_config
from codegen.generator import IngestorGenerator

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

import catalog_diff  # noqa: E402

_STAMP = {
    "manifest": "m.engine.yaml",
    "generator_commit": "g" * 40,
    "rocke_commit": "r" * 40,
}


def _render(config, out: Path, **kwargs) -> list[dict]:
    IngestorGenerator(_ROOT / "templates").render(config, out, **kwargs)
    return [
        json.loads(p.read_text())
        for p in sorted((out / config.descriptor_dir).glob("*.kdp.json"))
    ]


class TestGeneratedFromOnTheKdp:
    def test_it_is_stamped_on_every_kdp_beside_the_contract(
        self, scale_add_config, tmp_path
    ):
        kdps = _render(scale_add_config, tmp_path, generated_from=_STAMP)
        assert kdps
        for kdp in kdps:
            assert kdp["provenance"]["generated_from"] == _STAMP
            assert "specialization_contract" in kdp["provenance"]

    def test_a_render_that_did_not_ask_carries_none(self, scale_add_config, tmp_path):
        """A default render stays a pure function of the config: a commit id in every
        test render would make each one differ with the checkout."""
        for kdp in _render(scale_add_config, tmp_path):
            assert "generated_from" not in kdp["provenance"]


class TestGeneratedFromHelper:
    def _repo(self, tmp_path: Path) -> Path:
        def git(*args):
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(tmp_path),
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@t",
                ]
                + list(args),
                check=True,
                capture_output=True,
            )

        git("init", "-q")
        (tmp_path / "tracked").mkdir()
        (tmp_path / "tracked" / "f.txt").write_text("a")
        git("add", ".")
        git("commit", "-q", "-m", "init")
        return tmp_path

    def test_a_clean_path_is_its_last_commit(self, tmp_path):
        root = self._repo(tmp_path)
        sha = provenance.path_commit(root, root / "tracked")
        assert len(sha) == 40 and not sha.endswith("-dirty")

    def test_a_locally_edited_path_is_marked_dirty(self, tmp_path):
        root = self._repo(tmp_path)
        (root / "tracked" / "f.txt").write_text("b")
        assert provenance.path_commit(root, root / "tracked").endswith("-dirty")

    def test_a_missing_path_is_unknown_not_an_error(self, tmp_path):
        root = self._repo(tmp_path)
        assert provenance.path_commit(root, root / "nope") == provenance.UNKNOWN

    def test_the_record_names_inputs_repo_relative(self):
        manifest = _ROOT / "configs" / "gfx950_attention_dense.engine.yaml"
        record = provenance.generated_from(manifest)
        assert record["manifest"].endswith(
            "IngestorGenerator/configs/gfx950_attention_dense.engine.yaml"
        )
        assert not Path(record["manifest"]).is_absolute()
        assert {"generator_commit", "rocke_commit"} <= set(record)
        assert "config" not in record


class TestGenerateCliStampsTheKdp:
    def test_manifest_and_commits_reach_the_emitted_kdp(self, tmp_path):
        out = tmp_path / "out"
        subprocess.run(
            [
                sys.executable,
                str(_ROOT / "generate.py"),
                "--config",
                str(_ROOT / "configs" / "scale_add.yaml"),
                "--manifest",
                str(_ROOT / "configs" / "scale_add.yaml"),
                "--output-dir",
                str(out),
            ],
            check=True,
            capture_output=True,
            cwd=_ROOT,
        )
        (kdp_path,) = out.rglob("*.kdp.json")
        stamp = json.loads(kdp_path.read_text())["provenance"]["generated_from"]
        assert stamp["manifest"].endswith("configs/scale_add.yaml")
        assert stamp["config"].endswith("configs/scale_add.yaml")
        assert "generator_commit" in stamp and "rocke_commit" in stamp


def _config_with_kernel_provenance(tmp_path: Path, provenance_value) -> Path:
    raw = yaml.safe_load((_ROOT / "configs" / "scale_add.yaml").read_text())
    raw["packs"][0]["kernels"][0]["provenance"] = provenance_value
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


class TestKernelProvenance:
    def test_it_is_emitted_verbatim_on_the_kernel_descriptor(self, tmp_path):
        value = {"rocke_spec_id": "s", "rocke_tuning_id": "t@1"}
        config = load_config(_config_with_kernel_provenance(tmp_path, value))
        (kdp,) = _render(config, tmp_path / "out")
        assert kdp["kernelDescriptors"][0]["provenance"] == value

    def test_a_kernel_without_it_emits_no_key(self, scale_add_config, tmp_path):
        (kdp,) = _render(scale_add_config, tmp_path)
        assert all("provenance" not in k for k in kdp["kernelDescriptors"])

    @pytest.mark.parametrize(
        "value",
        [
            {"effective_spec": {}},
            {"rocke_spec_id": ""},
            {"rocke_spec_id": 3},
            "x",
        ],
        ids=["compiler-owned-key", "empty", "non-string", "not-a-mapping"],
    )
    def test_a_value_the_packer_would_not_accept_is_refused(self, tmp_path, value):
        """`effective_spec` is the producing compiler's statement, so authoring one
        must fail here rather than at hkp_pack."""
        with pytest.raises(ConfigError, match="provenance"):
            load_config(_config_with_kernel_provenance(tmp_path, value))


class TestCatalogDiffIgnoresProvenance:
    def _kdp(self, **provenance):
        return {
            "version": "1.0",
            "id": "k",
            "name": "hipkernel:stub",
            "arch": ["gfx950"],
            "matchers": ["m"],
            "engine": "e",
            "dispatch": "d",
            "provenance": provenance,
            "kernelDescriptors": [
                {
                    "version": "1.0",
                    "id": "i",
                    "name": "k0",
                    "kernel_source": {
                        "kind": "rocke",
                        "source": "s.py",
                        "builder": "b",
                        "spec": {"a": 1},
                    },
                    "metadata": {"a": 1},
                    "priority": 0,
                }
            ],
        }

    def test_generated_from_and_kernel_ids_are_not_differences(self, tmp_path):
        shipped = self._kdp(generated_from={**_STAMP, "generator_commit": "old"})
        regenerated = self._kdp(generated_from=_STAMP)
        regenerated["kernelDescriptors"][0]["provenance"] = {"rocke_spec_id": "s"}
        left, right = tmp_path / "a.json", tmp_path / "b.json"
        left.write_text(json.dumps(shipped))
        right.write_text(json.dumps(regenerated))
        rc = catalog_diff.main(
            [
                "--shipped",
                str(left),
                "--regenerated",
                str(right),
                "--ignore-ids",
                "--strict-spec",
            ]
        )
        assert rc == 0
