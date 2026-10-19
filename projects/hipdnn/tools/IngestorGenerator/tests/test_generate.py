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
    make_engine,
    make_minimal_config,
    make_packaged_config,
    make_rocke_kernel,
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

    @staticmethod
    def descriptors(directory: Path) -> dict:
        return {p.name: p.read_bytes() for p in sorted(directory.glob("*.json"))}

    @staticmethod
    def ids_of(directory: Path) -> set:
        ids = set()
        for path in directory.glob("*.json"):
            doc = json.loads(path.read_text(encoding="utf-8"))
            ids.add(doc["id"])
            ids.update(k["id"] for k in doc.get("kernelDescriptors", []))
        return ids

    def test_an_installed_bundle_is_reused_without_a_flag(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        _engine_dir, descriptor_dir = installed
        out = tmp_path / "out"

        code, captured = self.run_main(monkeypatch, capsys, "--output-dir", str(out))

        assert code == 0, captured.err
        assert "Reusing ids from the installed bundle" in captured.out
        generated = out / "test_descriptors" / "unit" / "scale_add"
        assert self.descriptors(generated) == self.descriptors(descriptor_dir)

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
        assert self.ids_of(generated).isdisjoint(self.ids_of(descriptor_dir))

    def test_new_ids_contradicts_ids_from(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        _engine_dir, descriptor_dir = installed
        code, _captured = self.run_main(
            monkeypatch,
            capsys,
            "--output-dir",
            str(tmp_path / "out"),
            "--new-ids",
            "--ids-from",
            str(descriptor_dir),
        )
        assert code == 2

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
        assert self.ids_of(held).isdisjoint(self.ids_of(descriptor_dir))

        code, captured = self.run_main(
            monkeypatch, capsys, "--output-dir", str(out), "--force"
        )

        assert code == 0, captured.err
        assert "Also reading" in captured.out
        assert "Warning" in captured.err and "it is ignored" in captured.err
        assert self.descriptors(held) == self.descriptors(descriptor_dir)

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

        generator = IngestorGenerator(TOOL_ROOT / "templates")
        root = tmp_path / "engine"
        home = make_packaged_config("rocKE/a", [64, 128])
        generator.render(home, root, resolution=resolve_ids(home, root))
        satellite = make_packaged_config("rocKE/b", [128, 256])
        generator.render(
            satellite,
            root,
            resolution=resolve_ids(
                satellite, root, engine_ids_from=[root / home.descriptor_dir]
            ),
        )
        monkeypatch.setattr(generate, "INSTALLED_ENGINE_DIR", str(root))
        return SimpleNamespace(
            home_dir=root / home.descriptor_dir,
            satellite_dir=root / satellite.descriptor_dir,
        )

    @staticmethod
    def run_main(monkeypatch, capsys, config, out, *flags):
        import generate

        monkeypatch.setattr(generate, "load_config", lambda _path: config)
        return TestInstalledBundle.run_main(
            monkeypatch, capsys, "--output-dir", str(out), *flags
        )

    @staticmethod
    def docs(directory: Path) -> dict:
        return {
            p.name: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(directory.glob("*.json"))
        }

    def test_a_satellite_regenerates_against_its_home_without_a_flag(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        config = make_packaged_config("rocKE/b", [128, 256])
        out = tmp_path / "out"

        code, captured = self.run_main(monkeypatch, capsys, config, out)

        assert code == 0, captured.err
        assert self.ENGINE_LINE in captured.out
        assert str(installed.home_dir) in captured.out
        written = self.docs(out / config.descriptor_dir)
        assert sorted(written) == ["test.kdp.json"]
        assert written == {
            "test.kdp.json": self.docs(installed.satellite_dir)["test.kdp.json"]
        }
        home_ued = self.docs(installed.home_dir)["test.ued.json"]["id"]
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
            d for n, d in self.docs(out / config.descriptor_dir).items() if "kdp" in n
        ]
        assert [k["name"] for k in kdp["kernelDescriptors"]] == ["a.b64", "a.b128"]

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
        written = self.docs(out / config.descriptor_dir)
        assert sorted(written) == ["test.kdp.json"]
        kdp = written["test.kdp.json"]
        assert kdp["engine"] == self.docs(installed.home_dir)["test.ued.json"]["id"]
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
            d for n, d in self.docs(out / config.descriptor_dir).items() if "kdp" in n
        ]
        assert [k["name"] for k in kdp["kernelDescriptors"]] == ["c.b256", "c.b512"]
        assert "not named" in captured.err
        assert str(installed.satellite_dir) in captured.err

    def test_new_ids_ignores_the_installed_siblings(
        self, installed, tmp_path, monkeypatch, capsys
    ):
        config = make_packaged_config("rocKE/b", [128, 256])
        out = tmp_path / "out"

        code, captured = self.run_main(monkeypatch, capsys, config, out, "--new-ids")

        assert code == 0, captured.err
        assert "Reusing" not in captured.out
        generated = self.docs(out / config.descriptor_dir)
        assert "test.ued.json" in generated, "a fresh bundle is its own home"
        installed_ids = set()
        for directory in (installed.home_dir, installed.satellite_dir):
            for doc in self.docs(directory).values():
                installed_ids.add(doc["id"])
                installed_ids.update(k["id"] for k in doc.get("kernelDescriptors", []))
        ids = set()
        for doc in generated.values():
            ids.add(doc["id"])
            ids.update(k["id"] for k in doc.get("kernelDescriptors", []))
        assert ids.isdisjoint(installed_ids)
