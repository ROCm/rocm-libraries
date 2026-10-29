# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CLI tests for generate.py: subprocess for the exit-code contract, in-process where
the provider tree has to be substituted.

Exit-code contract: exit 0 success; exit 1 a gate refusal or internal failure (a
strict budget overrun, --check-placeholders findings, an unreadable emitted file, a
template error); exit 2 unusable input (a missing or invalid config, a non-existent
--emitted-root, an --output-dir that is a file, a non-empty output directory without
--force, an --ids-from or --engine-ids-from that is not a directory, a ValueError
from resolving ids or rendering), which is also argparse's code for a bad flag.
--dry-run must not create the output dir. --force is required to overwrite an
existing non-empty output directory.
"""

import copy
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from codegen.config_loader import load_config
from codegen.generator import IngestorGenerator, resolve_ids
from tests.helpers import (
    all_ids,
    home_and_satellite,
    json_bytes,
    make_engine,
    make_minimal_config,
    make_pack,
    make_packaged_config,
    make_rocke_kernel,
    read_docs,
)

TOOL_ROOT = Path(__file__).parent.parent
SCALE_ADD_CONFIG = TOOL_ROOT / "configs" / "scale_add.yaml"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(TOOL_ROOT / "generate.py"), *args],
        cwd=TOOL_ROOT,
        capture_output=True,
        text=True,
    )


class TestExitCodes:
    def test_success_exits_zero(self, tmp_path):
        result = run_cli(
            "--config", str(SCALE_ADD_CONFIG), "--output-dir", str(tmp_path / "out")
        )
        assert result.returncode == 0, result.stderr

    def test_missing_config_file_exits_two(self, tmp_path):
        result = run_cli(
            "--config",
            str(tmp_path / "does_not_exist.yaml"),
            "--output-dir",
            str(tmp_path / "out"),
        )
        assert result.returncode == 2

    def test_config_error_exits_two(self, tmp_path):
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("engine:\n  name: unscoped\n")
        result = run_cli(
            "--config", str(bad_config), "--output-dir", str(tmp_path / "out")
        )
        assert result.returncode == 2
        assert not (tmp_path / "out").exists()

    def test_a_missing_emitted_root_exits_two(self, tmp_path):
        result = run_cli(
            "--config",
            str(SCALE_ADD_CONFIG),
            "--output-dir",
            str(tmp_path / "out"),
            "--check-placeholders",
            "--emitted-root",
            str(tmp_path / "no_such_root"),
        )
        assert result.returncode == 2

    def test_an_output_dir_that_is_a_file_exits_two(self, tmp_path):
        target = tmp_path / "out"
        target.write_text("not a directory")
        result = run_cli("--config", str(SCALE_ADD_CONFIG), "--output-dir", str(target))
        assert result.returncode == 2
        assert "Traceback" not in result.stderr
        assert str(target) in result.stderr

    def test_bad_flag_exits_two(self, tmp_path):
        result = run_cli("--nonexistent-flag", "value")
        assert result.returncode == 2

    def test_missing_required_flag_exits_two(self):
        result = run_cli("--config", str(SCALE_ADD_CONFIG))
        assert result.returncode == 2

    def test_an_ids_from_that_is_not_a_directory_exits_two(self, tmp_path):
        result = run_cli(
            "--config",
            str(SCALE_ADD_CONFIG),
            "--output-dir",
            str(tmp_path / "out"),
            "--ids-from",
            str(tmp_path / "no_such_bundle"),
        )
        assert result.returncode == 2
        assert "--ids-from" in result.stderr
        assert not (tmp_path / "out").exists()

    def test_an_engine_ids_from_that_is_not_a_directory_exits_two(self, tmp_path):
        result = run_cli(
            "--config",
            str(SCALE_ADD_CONFIG),
            "--output-dir",
            str(tmp_path / "out"),
            "--engine-ids-from",
            str(tmp_path / "no_such_bundle"),
        )
        assert result.returncode == 2
        assert "--engine-ids-from" in result.stderr


class TestIdSummary:
    def test_a_regeneration_reports_every_id_reused_and_none_minted(self, tmp_path):
        out = tmp_path / "out"
        first = run_cli("--config", str(SCALE_ADD_CONFIG), "--output-dir", str(out))
        assert first.returncode == 0, first.stderr
        assert "none found to reuse" in first.stdout

        second = run_cli(
            "--config", str(SCALE_ADD_CONFIG), "--output-dir", str(out), "--force"
        )
        assert second.returncode == 0, second.stderr
        assert "minted 0, retired 0" in second.stdout
        assert "none found" not in second.stdout


class TestInstalledEngineDir:
    """``INSTALLED_ENGINE_DIR`` is found by walking up from the script: the first
    ancestor that holds it, whatever the depth of the checkout."""

    def test_the_shipped_constant_resolves_to_this_checkouts_provider(self):
        import generate

        config = load_config(TOOL_ROOT / "configs" / "gfx1151_wmma_attention.yaml")

        installed = generate.installed_descriptor_dir(config)

        assert installed is not None
        assert installed.is_dir(), installed

    def test_a_relative_engine_dir_is_found_from_a_deeply_nested_script(
        self, tmp_path, monkeypatch
    ):
        import generate

        repo = tmp_path / "repo"
        engine_dir = repo / "providers" / "engine"
        config = make_packaged_config("rocKE/a", [64])
        (engine_dir / config.descriptor_dir).mkdir(parents=True)
        script = repo / "tools" / "deep" / "er" / "generate.py"
        script.parent.mkdir(parents=True)
        monkeypatch.setattr(generate, "__file__", str(script))
        monkeypatch.setattr(generate, "INSTALLED_ENGINE_DIR", "providers/engine")

        assert generate.installed_descriptor_dir(config) == (
            engine_dir / config.descriptor_dir
        )


class TestDryRun:
    def test_dry_run_does_not_create_output_dir(self, tmp_path):
        output_dir = tmp_path / "out"
        result = run_cli(
            "--config",
            str(SCALE_ADD_CONFIG),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        )
        assert result.returncode == 0, result.stderr
        assert not output_dir.exists()

    def test_dry_run_lists_expected_files(self, tmp_path):
        result = run_cli(
            "--config",
            str(SCALE_ADD_CONFIG),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        )
        assert "test_descriptors/unit/scale_add/scale_add.kmd.json" in result.stdout
        assert "packs/ScaleAddNative.cpp" in result.stdout


class TestForceOverwrite:
    def test_existing_nonempty_dir_without_force_exits_two(self, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "existing_file.txt").write_text("hand-authored content")

        result = run_cli(
            "--config", str(SCALE_ADD_CONFIG), "--output-dir", str(output_dir)
        )
        assert result.returncode == 2
        assert (output_dir / "existing_file.txt").read_text() == "hand-authored content"

    def test_existing_nonempty_dir_with_force_succeeds(self, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "existing_file.txt").write_text("stale")

        result = run_cli(
            "--config",
            str(SCALE_ADD_CONFIG),
            "--output-dir",
            str(output_dir),
            "--force",
        )
        assert result.returncode == 0, result.stderr

    def test_existing_empty_dir_without_force_succeeds(self, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        result = run_cli(
            "--config", str(SCALE_ADD_CONFIG), "--output-dir", str(output_dir)
        )
        assert result.returncode == 0, result.stderr


class TestGeneratedOutput:
    def test_success_reports_generated_file_count(self, tmp_path):
        result = run_cli(
            "--config", str(SCALE_ADD_CONFIG), "--output-dir", str(tmp_path / "out")
        )
        assert "Generated 14 files" in result.stdout

    def test_success_writes_files_to_disk(self, tmp_path):
        output_dir = tmp_path / "out"
        result = run_cli(
            "--config", str(SCALE_ADD_CONFIG), "--output-dir", str(output_dir)
        )
        assert result.returncode == 0, result.stderr
        assert (
            output_dir
            / "test_descriptors"
            / "unit"
            / "scale_add"
            / "scale_add.kmd.json"
        ).exists()
        assert (output_dir / "packs" / "ScaleAddNative.cpp").exists()


class TestLineEndings:
    """Every committed descriptor and source is LF. A CRLF render (Python's text-mode
    default on Windows) turns a copied file into a whole-file diff."""

    def test_every_written_file_is_lf_only(self, tmp_path):
        # binary_ops is multi-pack, so the operation UMDs are written too.
        for config in ("scale_add.yaml", "binary_ops.yaml"):
            out = tmp_path / config
            result = run_cli(
                "--config",
                str(TOOL_ROOT / "configs" / config),
                "--output-dir",
                str(out),
            )
            assert result.returncode == 0, result.stderr
            files = [p for p in out.rglob("*") if p.is_file()]
            assert any(p.name.endswith(".umd.json") for p in files)
            with_cr = [
                str(p.relative_to(out)) for p in files if b"\r" in p.read_bytes()
            ]
            assert with_cr == []


class TestNewIds:
    def test_new_ids_over_a_populated_output_directory_replaces_every_id(
        self, tmp_path
    ):
        out = tmp_path / "out"
        first = run_cli("--config", str(SCALE_ADD_CONFIG), "--output-dir", str(out))
        assert first.returncode == 0, first.stderr
        held = out / "test_descriptors" / "unit" / "scale_add"
        before = all_ids(read_docs(held))

        second = run_cli(
            "--config",
            str(SCALE_ADD_CONFIG),
            "--output-dir",
            str(out),
            "--force",
            "--new-ids",
        )

        assert second.returncode == 0, second.stderr
        assert "every id is new" in second.stdout
        after = all_ids(read_docs(held))
        assert len(after) == len(before)
        assert after.isdisjoint(before)


class TestInstalledBundle:
    """The bundle installed in the provider tree is where an engine's ids live, so a
    plain run reuses them. Driven in-process, with the provider tree pointed at a
    scratch directory."""

    @pytest.fixture
    def installed(self, tmp_path, monkeypatch):
        """``(engine_dir, descriptor_dir)``: a scale_add bundle installed under a
        scratch provider tree. Its ids are not the ones a fresh run would mint."""
        import generate

        config = load_config(SCALE_ADD_CONFIG)
        seed = tmp_path / "seed"
        IngestorGenerator(TOOL_ROOT / "templates").render(config, seed)
        engine_dir = tmp_path / "engine"
        descriptor_dir = engine_dir / config.descriptor_dir
        shutil.copytree(seed / config.descriptor_dir, descriptor_dir)
        monkeypatch.setattr(generate, "INSTALLED_ENGINE_DIR", str(engine_dir))
        return engine_dir, descriptor_dir

    @staticmethod
    def run_main(monkeypatch, capsys, *args, config=SCALE_ADD_CONFIG):
        import generate

        monkeypatch.setattr(
            sys, "argv", ["generate.py", "--config", str(config), *args]
        )
        code = 0
        try:
            generate.main()
        except SystemExit as exit_info:
            code = exit_info.code
        captured = capsys.readouterr()
        return code, captured

    def test_an_installed_bundle_is_reused_without_a_flag(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        _engine_dir, descriptor_dir = installed
        out = tmp_path / "out"

        code, captured = self.run_main(monkeypatch, capsys, "--output-dir", str(out))

        assert code == 0, captured.err
        assert "Reusing ids from the installed bundle" in captured.out
        generated = out / "test_descriptors" / "unit" / "scale_add"
        assert json_bytes(generated) == json_bytes(descriptor_dir)

    def test_new_ids_ignores_the_installed_bundle(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        _engine_dir, descriptor_dir = installed
        out = tmp_path / "out"

        code, captured = self.run_main(
            monkeypatch, capsys, "--output-dir", str(out), "--new-ids"
        )

        assert code == 0, captured.err
        assert "Reusing ids" not in captured.out
        generated = out / "test_descriptors" / "unit" / "scale_add"
        assert all_ids(read_docs(generated)).isdisjoint(
            all_ids(read_docs(descriptor_dir))
        )

    @pytest.mark.parametrize("flag", ["--ids-from", "--engine-ids-from"])
    def test_new_ids_contradicts_the_reuse_flags(
        self, installed, tmp_path, monkeypatch, capsys, flag
    ):
        _engine_dir, descriptor_dir = installed
        code, captured = self.run_main(
            monkeypatch,
            capsys,
            "--output-dir",
            str(tmp_path / "out"),
            "--new-ids",
            flag,
            str(descriptor_dir),
        )
        assert code == 2
        assert "--new-ids contradicts" in captured.err

    def test_an_installed_ued_without_its_kdp_refuses_and_names_new_ids(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        """gfx950's KDP is DVC-only; reminting every id because it was not fetched
        would be silent, so the run stops and says how to override."""
        _engine_dir, descriptor_dir = installed
        (descriptor_dir / "scale_add.kdp.json").unlink()
        (descriptor_dir / "artifacts.dvc").write_text("outs: []\n", encoding="utf-8")
        out = tmp_path / "out"

        code, captured = self.run_main(monkeypatch, capsys, "--output-dir", str(out))

        assert code == 2
        assert "--new-ids" in captured.err
        assert not out.exists()

    def test_an_output_directory_bundle_from_other_ids_is_ignored(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        """A scratch generated without the installed ids shares nothing with them: it
        is named, ignored with a warning, and the installed ids win."""
        _engine_dir, descriptor_dir = installed
        out = tmp_path / "out"
        IngestorGenerator(TOOL_ROOT / "templates").render(
            load_config(SCALE_ADD_CONFIG), out
        )
        held = out / "test_descriptors" / "unit" / "scale_add"
        assert all_ids(read_docs(held)).isdisjoint(all_ids(read_docs(descriptor_dir)))

        code, captured = self.run_main(
            monkeypatch, capsys, "--output-dir", str(out), "--force"
        )

        assert code == 0, captured.err
        assert "Also reading" in captured.out
        assert "Warning" in captured.err and "it is ignored" in captured.err
        assert json_bytes(held) == json_bytes(descriptor_dir)

    @staticmethod
    def with_kernels(*blocks):
        """scale_add plus one extra kernel per block, added since installation."""
        config = load_config(SCALE_ADD_CONFIG)
        pack = config.packs[0]
        for block in blocks:
            kernel = copy.deepcopy(pack.kernels[0])
            kernel.name = f"scale_add.f32_block{block}"
            kernel.metadata["block_size"] = block
            pack.kernels.append(kernel)
        return config

    @staticmethod
    def run_with(monkeypatch, capsys, config, out, *flags):
        import generate

        monkeypatch.setattr(generate, "load_config", lambda _path: config)
        return TestInstalledBundle.run_main(
            monkeypatch, capsys, "--output-dir", str(out), *flags
        )

    @staticmethod
    def kernel_ids(directory: Path) -> dict:
        kdp = json.loads((directory / "scale_add.kdp.json").read_text(encoding="utf-8"))
        return {k["name"]: k["id"] for k in kdp["kernelDescriptors"]}

    def test_iterating_on_an_extension_keeps_installed_and_scratch_ids(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        """Between installation and the next install, the scratch directory is where
        the new kernels' ids live; the installed bundle does not know them."""
        _engine_dir, descriptor_dir = installed
        out = tmp_path / "out"
        scratch = out / "test_descriptors" / "unit" / "scale_add"
        code, captured = self.run_with(monkeypatch, capsys, self.with_kernels(128), out)
        assert code == 0, captured.err
        assert "Also reading" not in captured.out
        first = self.kernel_ids(scratch)

        code, captured = self.run_with(
            monkeypatch, capsys, self.with_kernels(128, 512), out, "--force"
        )

        assert code == 0, captured.err
        assert "Also reading" in captured.out
        second = self.kernel_ids(scratch)
        shipped = self.kernel_ids(descriptor_dir)
        assert {name: second[name] for name in shipped} == shipped
        assert second["scale_add.f32_block128"] == first["scale_add.f32_block128"]
        assert second["scale_add.f32_block512"] not in {
            *first.values(),
            *shipped.values(),
        }
        for name in ("scale_add.ued.json", "scale_add.kmd.json"):
            assert (scratch / name).read_bytes() == (descriptor_dir / name).read_bytes()

    def test_the_installed_id_wins_where_the_scratch_disagrees(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        _engine_dir, descriptor_dir = installed
        out = tmp_path / "out"
        scratch = out / "test_descriptors" / "unit" / "scale_add"
        code, captured = self.run_with(monkeypatch, capsys, self.with_kernels(128), out)
        assert code == 0, captured.err
        kdp_path = scratch / "scale_add.kdp.json"
        kdp = json.loads(kdp_path.read_text(encoding="utf-8"))
        kdp["id"] = str(uuid.uuid4())
        kdp["kernelDescriptors"][0]["id"] = str(uuid.uuid4())
        kdp_path.write_text(json.dumps(kdp, indent=2) + "\n", encoding="utf-8")
        kept = self.kernel_ids(scratch)["scale_add.f32_block128"]

        code, captured = self.run_with(
            monkeypatch, capsys, self.with_kernels(128), out, "--force"
        )

        assert code == 0, captured.err
        ids = self.kernel_ids(scratch)
        shipped = self.kernel_ids(descriptor_dir)
        assert {name: ids[name] for name in shipped} == shipped
        assert ids["scale_add.f32_block128"] == kept
        installed_kdp = json.loads(
            (descriptor_dir / "scale_add.kdp.json").read_text(encoding="utf-8")
        )
        regenerated = json.loads(kdp_path.read_text(encoding="utf-8"))
        assert regenerated["id"] == installed_kdp["id"]

    def test_an_unreadable_scratch_is_ignored_with_a_warning(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        """It only fills gaps, so a scratch the reader refuses (a UED and no KDP) must
        not fail a run the installed bundle can answer alone."""
        _engine_dir, descriptor_dir = installed
        out = tmp_path / "out"
        scratch = out / "test_descriptors" / "unit" / "scale_add"
        scratch.mkdir(parents=True)
        shutil.copy(descriptor_dir / "scale_add.ued.json", scratch)

        code, captured = self.run_main(
            monkeypatch, capsys, "--output-dir", str(out), "--force"
        )

        assert code == 0, captured.err
        assert "Also reading" in captured.out
        assert "ignored as a source of ids" in captured.err
        assert self.kernel_ids(scratch) == self.kernel_ids(descriptor_dir)

    def test_a_dry_run_refuses_an_unusable_ids_from(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        """Dry runs resolve ids too, so a bad prior fails here instead of at the
        real run."""
        _engine_dir, descriptor_dir = installed
        bad = tmp_path / "bad" / "test_descriptors" / "unit" / "scale_add"
        bad.mkdir(parents=True)
        shutil.copy(descriptor_dir / "scale_add.ued.json", bad)
        out = tmp_path / "out"

        code, _captured = self.run_main(
            monkeypatch,
            capsys,
            "--output-dir",
            str(out),
            "--dry-run",
            "--ids-from",
            str(bad),
        )

        assert code == 2
        assert not out.exists()

    def test_a_dry_run_says_which_ids_it_would_reuse(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        _engine_dir, descriptor_dir = installed
        out = tmp_path / "out"

        code, captured = self.run_main(
            monkeypatch,
            capsys,
            "--output-dir",
            str(out),
            "--dry-run",
            "--ids-from",
            str(descriptor_dir),
        )

        assert code == 0, captured.err
        lines = captured.out.splitlines()
        assert any(line.startswith("Ids (would be): reused") for line in lines)
        assert not out.exists()

    EARLIER = "holds an earlier generation of this engine"

    def test_an_earlier_pass_beside_the_output_directory_is_named(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        """One output directory per pass reads nothing from the last one: the new
        kernels' ids are minted again, which is what the warning is for."""
        work = tmp_path / "work"
        previous = work / "pass1"
        code, captured = self.run_with(
            monkeypatch, capsys, self.with_kernels(128), previous
        )
        assert code == 0, captured.err
        assert self.EARLIER not in captured.err

        code, captured = self.run_with(
            monkeypatch, capsys, self.with_kernels(128, 512), work / "pass2"
        )

        assert code == 0, captured.err
        assert self.EARLIER in captured.err
        assert str(previous / "test_descriptors" / "unit" / "scale_add") in captured.err

    def test_naming_the_previous_pass_reads_it_and_silences_the_warning(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        work = tmp_path / "work"
        previous = work / "pass1"
        code, captured = self.run_with(
            monkeypatch, capsys, self.with_kernels(128), previous
        )
        assert code == 0, captured.err
        bundle = previous / "test_descriptors" / "unit" / "scale_add"

        code, captured = self.run_with(
            monkeypatch,
            capsys,
            self.with_kernels(128),
            work / "pass2",
            "--ids-from",
            str(bundle),
        )

        assert code == 0, captured.err
        assert self.EARLIER not in captured.err
        assert "minted 0" in captured.out

    def test_no_warning_without_an_earlier_pass(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        code, captured = self.run_with(
            monkeypatch, capsys, self.with_kernels(128), tmp_path / "solo" / "pass1"
        )
        assert code == 0, captured.err
        assert "minted 1" in captured.out
        assert self.EARLIER not in captured.err

    def test_another_engines_scratch_is_not_an_earlier_pass(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        """Same directory name, different engine: nothing of it is this bundle's."""
        work = tmp_path / "work"
        other = make_minimal_config(
            engine=make_engine(name="other:ScaleAdd"), authored_subpath="unit"
        )
        IngestorGenerator(TOOL_ROOT / "templates").render(other, work / "other")
        assert (work / "other" / other.descriptor_dir).is_dir()

        code, captured = self.run_with(
            monkeypatch, capsys, self.with_kernels(128), work / "pass1"
        )

        assert code == 0, captured.err
        assert "minted 1" in captured.out
        assert self.EARLIER not in captured.err

    def test_an_unreadable_neighbour_of_the_output_directory_is_skipped(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        """The scratch parent is often shared (a /tmp), and a neighbour the user
        cannot read is not theirs to be warned about."""
        import generate

        work = tmp_path / "work"
        previous = work / "pass1"
        code, captured = self.run_with(
            monkeypatch, capsys, self.with_kernels(128), previous
        )
        assert code == 0, captured.err
        (work / "locked").mkdir()
        read = generate.prior_ids.holds_docs

        def holds_docs(root, *kinds):
            if "locked" in root.parts:
                raise PermissionError(13, "Permission denied", str(root))
            return read(root, *kinds)

        monkeypatch.setattr(generate.prior_ids, "holds_docs", holds_docs)

        code, captured = self.run_with(
            monkeypatch, capsys, self.with_kernels(128, 512), work / "pass2"
        )

        assert code == 0, captured.err
        assert str(previous / "test_descriptors" / "unit" / "scale_add") in captured.err

    def test_the_newest_earlier_pass_is_the_one_named(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        """Passes are named newest first, whatever order the directory lists them
        in: the newest is the previous pass."""
        work = tmp_path / "work"
        bundles = {}
        for name, age in (("pass1", 300), ("pass2", 100), ("pass3", 200)):
            code, captured = self.run_with(
                monkeypatch, capsys, self.with_kernels(128), work / name
            )
            assert code == 0, captured.err
            bundles[name] = work / name / "test_descriptors" / "unit" / "scale_add"
            stamp = 2_000_000_000 - age
            os.utime(bundles[name], (stamp, stamp))

        code, captured = self.run_with(
            monkeypatch, capsys, self.with_kernels(128, 512), work / "pass4"
        )

        assert code == 0, captured.err
        assert f"--ids-from {bundles['pass2']}" in captured.err

    def test_a_removed_packs_files_are_reported_as_left_behind(
        self, tmp_path, monkeypatch, capsys
    ):
        """The next run reads a leftover KDP as part of the previous bundle."""
        import generate

        monkeypatch.setattr(generate, "INSTALLED_ENGINE_DIR", str(tmp_path / "none"))
        doc = yaml.safe_load((TOOL_ROOT / "configs" / "binary_ops.yaml").read_text())
        smaller = tmp_path / "smaller.yaml"
        smaller.write_text(yaml.safe_dump(doc), encoding="utf-8")
        extra = copy.deepcopy(doc["packs"][1])
        extra.update(name="min", discriminator="min")
        extra["kernels"][0].update(name="binary_ops_min.f32_block64")
        extra["kernels"][0]["metadata"]["operation"] = "MIN"
        doc["packs"].append(extra)
        larger = tmp_path / "larger.yaml"
        larger.write_text(yaml.safe_dump(doc), encoding="utf-8")
        out = tmp_path / "out"

        code, captured = self.run_main(
            monkeypatch, capsys, "--output-dir", str(out), config=larger
        )
        assert code == 0, captured.err
        assert "Warning" not in captured.err

        seen = {}
        original = generate.IngestorGenerator.render

        def spy(generator, *args, **kwargs):
            seen["stderr"] = capsys.readouterr().err
            return original(generator, *args, **kwargs)

        monkeypatch.setattr(generate.IngestorGenerator, "render", spy)

        code, captured = self.run_main(
            monkeypatch, capsys, "--output-dir", str(out), "--force", config=smaller
        )

        assert code == 0, captured.err
        # Said while the files can still be dealt with, not after they are written.
        assert "will not be written" in seen["stderr"]
        assert "binary_ops_min.kdp.json" in seen["stderr"]
        assert "operation_is_min.umd.json" in seen["stderr"]


class TestInstalledSiblings:
    """The other bundles of an engine, installed beside this one, are found without a
    flag: a satellite needs its home's engine ids and a home needs its satellites'
    kernels in its duplicate check, and a forgotten flag is the failure."""

    ENGINE_LINE = "Reusing the engine's ids from the installed bundle(s)"

    @pytest.fixture
    def installed(self, tmp_path, monkeypatch):
        """Home ``rocKE/a`` (blocks 64, 128) and satellite ``rocKE/b`` (blocks 128,
        256) installed under one scratch provider tree."""
        import generate

        installed = home_and_satellite(
            IngestorGenerator(TOOL_ROOT / "templates"), tmp_path, same_tree=True
        )
        monkeypatch.setattr(generate, "INSTALLED_ENGINE_DIR", str(installed.home_root))
        return installed

    @staticmethod
    def run_main(monkeypatch, capsys, config, out, *flags):
        import generate

        monkeypatch.setattr(generate, "load_config", lambda _path: config)
        return TestInstalledBundle.run_main(
            monkeypatch, capsys, "--output-dir", str(out), *flags
        )

    def test_a_satellite_regenerates_against_its_home_without_a_flag(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        config = make_packaged_config("rocKE/b", [128, 256])
        out = tmp_path / "out"

        code, captured = self.run_main(monkeypatch, capsys, config, out)

        assert code == 0, captured.err
        assert self.ENGINE_LINE in captured.out
        assert str(installed.home_dir) in captured.out
        written = read_docs(out / config.descriptor_dir)
        assert sorted(written) == ["test.kdp.json"]
        assert written == {
            "test.kdp.json": read_docs(installed.satellite_dir)["test.kdp.json"]
        }
        home_ued = read_docs(installed.home_dir)["test.ued.json"]["id"]
        assert written["test.kdp.json"]["engine"] == home_ued

    def test_a_home_regenerates_with_its_satellites_kernels_in_the_duplicate_check(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        config = make_packaged_config("rocKE/a", [64, 128])
        config.packs[0].kernels.append(make_rocke_kernel("a.dup_of_b", 256))
        out = tmp_path / "out"

        code, captured = self.run_main(monkeypatch, capsys, config, out)

        assert code == 0, captured.err
        assert str(installed.satellite_dir) in captured.out
        assert "not named" not in captured.err
        (kdp,) = [
            d for n, d in read_docs(out / config.descriptor_dir).items() if "kdp" in n
        ]
        assert [k["name"] for k in kdp["kernelDescriptors"]] == ["a.b64", "a.b128"]
        assert sorted(read_docs(out / config.descriptor_dir)) == [
            "kernel_dtype_matches_graph.umd.json",
            "test.kdp.json",
            "test.kmd.json",
            "test.udd.json",
            "test.ued.json",
            "test.uhd.json",
        ]

    def test_a_new_satellite_takes_its_engine_ids_from_the_installed_home(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        """Its own directory is not installed yet; the siblings are found beside
        where it will go, and both of their kernel sets are in its duplicate check."""
        config = make_packaged_config("rocKE/c", [128, 256, 512])
        out = tmp_path / "out"

        code, captured = self.run_main(monkeypatch, capsys, config, out)

        assert code == 0, captured.err
        assert self.ENGINE_LINE in captured.out
        written = read_docs(out / config.descriptor_dir)
        assert sorted(written) == ["test.kdp.json"]
        kdp = written["test.kdp.json"]
        assert kdp["engine"] == read_docs(installed.home_dir)["test.ued.json"]["id"]
        assert [k["name"] for k in kdp["kernelDescriptors"]] == ["c.b512"]

    def test_naming_a_sibling_replaces_the_discovered_ones(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        """An explicit flag is the override: the bundle it leaves out is no longer
        seeded, and the generator says so."""
        config = make_packaged_config("rocKE/c", [128, 256, 512])
        out = tmp_path / "out"

        code, captured = self.run_main(
            monkeypatch,
            capsys,
            config,
            out,
            "--engine-ids-from",
            str(installed.home_dir),
        )

        assert code == 0, captured.err
        assert self.ENGINE_LINE not in captured.out
        (kdp,) = [
            d for n, d in read_docs(out / config.descriptor_dir).items() if "kdp" in n
        ]
        assert [k["name"] for k in kdp["kernelDescriptors"]] == ["c.b256", "c.b512"]
        assert "not named" in captured.err
        assert str(installed.satellite_dir) in captured.err

    def test_a_satellite_passes_check_placeholders_on_its_own_kdp(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        config = make_packaged_config("rocKE/b", [128, 256])
        out = tmp_path / "out"
        code, captured = self.run_main(monkeypatch, capsys, config, out)
        assert code == 0, captured.err

        code, captured = self.run_main(
            monkeypatch, capsys, config, out, "--check-placeholders"
        )

        assert code == 0, captured.err + captured.out
        assert "Located 1 of this engine's 1 shippable files" in captured.out
        assert "not found" not in captured.out

    def test_a_multi_pack_config_beside_a_sibling_is_refused_without_a_flag_hint(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        config = make_packaged_config("rocKE/c", [64])
        config.packs.append(
            make_pack(
                name="second",
                kernels=[make_rocke_kernel("c.b32", 32)],
                arch=["gfx950"],
            )
        )
        out = tmp_path / "out"

        code, captured = self.run_main(monkeypatch, capsys, config, out)

        assert code == 2
        assert "single-pack" in captured.err
        assert "--engine-ids-from" not in captured.err
        assert not out.exists()

    def test_new_ids_ignores_the_installed_siblings(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        config = make_packaged_config("rocKE/b", [128, 256])
        out = tmp_path / "out"

        code, captured = self.run_main(monkeypatch, capsys, config, out, "--new-ids")

        assert code == 0, captured.err
        assert "Reusing" not in captured.out
        generated = read_docs(out / config.descriptor_dir)
        assert "test.ued.json" in generated, "a fresh bundle is its own home"
        installed_ids = all_ids(read_docs(installed.home_dir)) | all_ids(
            read_docs(installed.satellite_dir)
        )
        assert all_ids(generated).isdisjoint(installed_ids)

    def test_a_satellite_run_names_an_earlier_pass_it_did_not_read(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        """A satellite pass holds a KDP and no UED, so the engine is told by the KDP's
        engine id; the kernel added since installation is minted again."""
        config = make_packaged_config("rocKE/b", [128, 256, 512])
        work = tmp_path / "work"
        code, captured = self.run_main(monkeypatch, capsys, config, work / "pass1")
        assert code == 0, captured.err
        assert "earlier generation" not in captured.err

        code, captured = self.run_main(monkeypatch, capsys, config, work / "pass2")

        assert code == 0, captured.err
        assert "minted 1" in captured.out
        assert "earlier generation of this engine" in captured.err
        assert str(work / "pass1" / config.descriptor_dir) in captured.err

    def test_a_satellite_whose_kdp_is_an_unfetched_dvc_artifact_is_refused(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        """Its kernel ids live in the artifact, so a run that cannot read them would
        mint every one again."""
        config = make_packaged_config("rocKE/b", [128, 256])
        (installed.satellite_dir / "test.kdp.json").unlink()
        pointer = installed.satellite_dir / "artifacts.dvc"
        pointer.write_text(
            "outs:\n- md5: 106ebd4caeba7b508fa33fcd8383c130.dir\n  path: artifacts\n",
            encoding="utf-8",
        )
        out = tmp_path / "out"

        code, captured = self.run_main(monkeypatch, capsys, config, out)

        assert code == 2, captured.out
        assert pointer.name in captured.err
        assert "dvc pull" in captured.err
        assert "--new-ids" in captured.err
        assert not out.exists()


class TestRefusedRuns:
    """Runs the generator stops before writing, driven in-process with no installed
    provider tree: the output directory is the only previous bundle."""

    @pytest.fixture(autouse=True)
    def no_installed_tree(self, tmp_path, monkeypatch):
        import generate

        monkeypatch.setattr(generate, "INSTALLED_ENGINE_DIR", str(tmp_path / "none"))

    @staticmethod
    def run(monkeypatch, capsys, config, out, *flags):
        return TestInstalledSiblings.run_main(monkeypatch, capsys, config, out, *flags)

    @staticmethod
    def snapshot(root: Path) -> dict:
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    @pytest.mark.parametrize("flag", ["--force", "--dry-run"])
    def test_a_pack_whose_kdp_is_a_dvc_artifact_is_not_given_a_second_kdp(
        self, flag, tmp_path, monkeypatch, capsys
    ):
        config = make_packaged_config("rocKE/a", [64, 128])
        out = tmp_path / "out"
        IngestorGenerator(TOOL_ROOT / "templates").render(config, out)
        held = out / config.descriptor_dir
        (held / "artifacts").mkdir()
        (held / "test.kdp.json").rename(held / "artifacts" / "test.kdp.json")
        before = self.snapshot(out)

        code, captured = self.run(monkeypatch, capsys, config, out, "--force", flag)

        assert code == 2, captured.out
        assert "artifacts/test.kdp.json" in captured.err
        assert "splice" in captured.err
        assert self.snapshot(out) == before

    def test_a_stale_kdp_beside_a_renamed_packs_is_named_for_deletion(
        self, tmp_path, monkeypatch, capsys
    ):
        config = load_config(TOOL_ROOT / "configs" / "binary_ops.yaml")
        out = tmp_path / "out"
        code, captured = self.run(monkeypatch, capsys, config, out)
        assert code == 0, captured.err
        renamed = copy.deepcopy(config)
        renamed.packs[1].name += "_x"
        code, captured = self.run(monkeypatch, capsys, renamed, out, "--force")
        assert code == 0, captured.err

        code, captured = self.run(monkeypatch, capsys, renamed, out, "--force")

        assert code == 2
        assert "appears twice under" in captured.err
        assert "delete the stale file" in captured.err
        assert "override them" not in captured.err

    def test_the_id_summary_names_retired_kernels_before_the_arch_clause(
        self, tmp_path, monkeypatch, capsys
    ):
        def kernel(name, block, arch):
            made = make_rocke_kernel(name, block)
            made.arch = arch
            return made

        config = make_packaged_config("rocKE/a", [64])
        config.packs[0].arch = ["gfx1100", "gfx1201"]
        config.packs[0].kernels = [
            kernel("k64", 64, []),
            kernel("k256", 256, ["gfx1100"]),
            kernel("k512", 512, []),
        ]
        out = tmp_path / "out"
        IngestorGenerator(TOOL_ROOT / "templates").render(config, out)
        moved = copy.deepcopy(config)
        moved.packs[0].kernels = [
            kernel("k64", 64, []),
            kernel("k256", 256, ["gfx1201"]),
        ]

        code, captured = self.run(monkeypatch, capsys, moved, out, "--force")

        assert code == 0, captured.err
        (line,) = [l for l in captured.out.splitlines() if l.startswith("Ids:")]
        assert line.endswith(
            "minted 0, retired 1: k512; 1 kept an id across an arch change: k256"
        ), line
