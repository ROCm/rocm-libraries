# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests that exercise the audit modules' remaining branches."""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import pytest

import audit_checks
import audit_hipblaslt_libraries as audit
from audit_hipblaslt_libraries import (
    LOGIC_REL_PREFIX,
    AuditCache,
    AuditResult,
    FileAuditContext,
    GroupStats,
    LibraryRecord,
)

LOGIC_REL = Path("library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full")


def _codes(record: LibraryRecord, check: str) -> list[str]:
    """Run one check and return the error codes it emits.

    Args:
        record: Library record under test.
        check: Check id from the audit registry.

    Returns:
        Error codes in emission order.

    Raises:
        KeyError: If ``check`` is not registered.
    """
    found: list[str] = []
    audit.run_enabled_checks(
        record,
        lambda _path, code, _message: found.append(code),
        {},
        FileAuditContext(),
        (check,),
    )
    return found


def _record(**overrides: Any) -> LibraryRecord:
    """Build a dict-format library record with overridable fields.

    Args:
        overrides: ``LibraryRecord`` fields to replace.

    Returns:
        Library record using a gfx950 Equality filename by default.

    Raises:
        TypeError: If an override is not a ``LibraryRecord`` field.
    """
    fields: dict[str, Any] = {
        "path": Path("gfx950/Equality/gfx950_Cijk_Ailk_Bjlk_HHS_BH_UserArgs.yaml"),
        "format": "dict",
        "architecture": "gfx950",
        "schedule_name": "gfx950",
        "device_ids": ("75a0",),
        "library_type": "Equality",
        "problem_type": {
            "TransposeA": False,
            "TransposeB": True,
            "DataType": 4,
            "DataTypeA": 4,
            "DataTypeB": 4,
            "DestDataType": 4,
            "DataTypeE": 4,
            "ComputeDataType": 0,
        },
        "solutions": [
            {
                "SolutionIndex": 0,
                "KernelNameMin": "Cijk_Ailk_Bjlk_HHS_UserArgs_MT1",
                "SolutionNameMin": "Cijk_Ailk_Bjlk_HHS_UserArgs_MT1_GSU0",
                "MacroTile0": 128,
                "MacroTile1": 128,
                "DepthU": 32,
            }
        ],
        "index_order": [],
        "exact_logic": [([128, 128, 64, 1], [0, 0.0])],
        "range_logic": [],
        "perf_metric": "DeviceEfficiency",
        "device_names_raw": ["Device 75a0"],
        "default_solution": {},
    }
    fields.update(overrides)
    return LibraryRecord(**fields)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in ``repo`` with a fixed author.

    Args:
        repo: Repository directory.
        args: Git arguments, excluding the git executable.

    Returns:
        Completed process with captured text output.

    Raises:
        subprocess.CalledProcessError: If git exits non-zero.
    """
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "Audit Test",
            "GIT_AUTHOR_EMAIL": "audit@example.com",
            "GIT_COMMITTER_NAME": "Audit Test",
            "GIT_COMMITTER_EMAIL": "audit@example.com",
        }
    )
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def _init_repo(repo: Path) -> None:
    """Create a git repository with an initial empty commit.

    Args:
        repo: Directory to initialize.

    Returns:
        None.

    Raises:
        subprocess.CalledProcessError: If git init or commit fails.
    """
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init")
    _git(repo, "commit", "--allow-empty", "-m", "init")


def test_type_token_decoding_covers_widths_and_suffixes() -> None:
    """Decode GEMM type tokens of every accepted width, plus suffixes.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If a token decodes to an unexpected rank.
        ValueError: If a deliberately invalid token is accepted.
    """
    assert audit._strip_filename_type_suffixes("S_MX") == "S_MX"
    assert audit.decode_gemm_type_token("S_MX") == (0, 0, 0, 0, 0)
    assert len(audit._greedy_type_tokens("HS")) == 2
    assert len(audit._greedy_type_tokens("HHS")) == 3
    assert len(audit._greedy_type_tokens("HHHS")) == 4
    assert len(audit._greedy_type_tokens("HHHHS")) == 5
    assert audit.decode_gemm_type_token("HS")[0] == 4
    assert audit.decode_gemm_type_token("HHHS")[0] == 4
    assert audit.decode_gemm_type_token("HHHHS")[0] == 4
    assert audit.decode_gemm_type_token("BBS_STA_SAV")[0] == 7
    assert audit.decode_gemm_type_token("HHS_SPA")[0] == 4
    assert audit.decode_gemm_type_token("HHS_MXAscale")[0] == 4
    assert audit.decode_gemm_type_token("HHS_Mscale")[0] == 4
    with pytest.raises(ValueError):
        audit._greedy_type_tokens("QQ")
    with pytest.raises(ValueError):
        audit.decode_gemm_type_token("H")
    with pytest.raises(ValueError):
        audit.decode_gemm_type_token("HHHHHS")
    with pytest.raises(ValueError):
        audit._lookup_type_char("NOPE")
    with pytest.raises(ValueError):
        audit.transpose_a_from_layout("XXXX")
    with pytest.raises(ValueError):
        audit.transpose_b_from_layout("YYYY")
    assert audit.transpose_a_from_layout("Alik") is True
    assert audit.transpose_b_from_layout("Bljk") is False


def test_filename_parse_and_epilogue_helpers() -> None:
    """Parse matching filenames and reject stems that are not libraries.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If a parsed field or epilogue key is wrong.
    """
    parsed = audit.parse_filename("gfx950_Cijk_Ailk_Bjlk_HHS_BH_UserArgs")
    assert parsed is not None
    assert parsed["types"] is not None
    broken = audit.parse_filename("gfx950_Cijk_Ailk_Bjlk_QQQ_BH_UserArgs")
    assert broken is not None and broken["types"] is None
    assert audit.parse_filename("not-a-library") is None
    assert audit.extract_epilogue_key("Cijk_Ailk_Bjlk_HHS_UserArgs_MT") == "HHS"
    assert audit.extract_epilogue_key("no-epilogue") == ""
    assert audit.solution_epilogue_keys(
        [
            "skip",
            {"KernelNameMin": "plain"},
            {"SolutionNameMin": "Cijk_Ailk_Bjlk_BBS_UserArgs"},
        ]
    ) == {"BBS"}


def test_normalize_library_accepts_both_roots_and_rejects_others() -> None:
    """Cover list, dict, and unsupported YAML roots.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If a recognized root is misclassified.
        ValueError: If an invalid root is accepted.
    """
    short: list[Any] = [None] * 11
    with pytest.raises(ValueError):
        audit.normalize_library(short, Path("short.yaml"))
    listed: list[Any] = [None] * 12
    listed[1] = "gfx942"
    listed[2] = {"Architecture": "GFX942"}
    listed[3] = ["skip", "Device 0049"]
    listed[4] = {}
    listed[5] = []
    listed[10] = None
    listed[11] = "Equality"
    record = audit.normalize_library(listed, Path("list.yaml"))
    assert record.architecture == "gfx942"
    assert record.device_ids == ("0049",)
    assert record.perf_metric is None
    as_dict = audit.normalize_library(
        {
            "ArchitectureName": "GFX950",
            "Library": {"distance": "FreeSize"},
            "DefaultSolution": {"UseCustomMainLoopSchedule": 1},
            "DeviceNames": [],
        },
        Path("dict.yaml"),
    )
    assert as_dict.library_type == "FreeSize"
    assert as_dict.default_solution["UseCustomMainLoopSchedule"] == 1
    with pytest.raises(ValueError):
        audit.normalize_library("scalar", Path("bad.yaml"))
    assert audit.yaml_root_format([]) == "list"
    assert audit.yaml_root_format({}) == "dict"
    assert audit.yaml_root_format(1) == "other"
    assert audit._parse_devices(None) == ()
    assert audit._architecture_from_field("GFX90A") == "gfx90a"


def test_size_tile_and_performance_iterators() -> None:
    """Walk ExactLogic and solution tiles, including skipped entries.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If a yielded size, tile, or performance value is wrong.
    """
    exact = [None, ("short",), ([1, 2], "nope"), ([3, 4], [0, 1.5]), ([5], [0, "x"])]
    assert list(audit.iter_gemm_sizes(None)) == []
    assert list(audit.iter_gemm_sizes(exact)) == [[1, 2], [3, 4], [5]]
    assert list(audit.iter_performance_values([])) == []
    assert list(audit.iter_performance_values(exact)) == [1.5]
    tiles = audit.solution_tiles(
        ["skip", {"MacroTile0": 1}, {"MacroTile0": 8, "MacroTile1": 8, "DepthU": 16}]
    )
    assert tiles == {(8, 8, 16)}
    stats = GroupStats(files=1, tiles=tiles)
    assert stats.to_dict()["unique_tiles"] == 1


def test_path_and_device_folder_helpers(tmp_path: Path) -> None:
    """Resolve device folders, library folders, and the hipBLASLt logic root.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.

    Raises:
        AssertionError: If a path helper returns an unexpected device or folder.
    """
    logic = tmp_path / "logic"
    (logic / "gfx950" / "gfx950_id75a3").mkdir(parents=True)
    (logic / "gfx950" / "readme.txt").write_text("x", encoding="utf-8")
    (logic / "skip.txt").write_text("x", encoding="utf-8")
    assert audit.discover_device_folders(logic) == {"gfx950": {"75a3"}}
    assert audit.path_device_id(Path("gfx950/gfx950_id75A3/a.yaml"), "gfx950") == "75a3"
    assert audit.path_device_id(Path("gfx950/plain/a.yaml"), "gfx950") is None
    assert audit.path_library_folder(Path("gfx950/Origami/a.yaml")) == "Origami"
    assert audit.path_library_folder(Path("gfx950/a.yaml")) == ""
    assert audit.display_library_type("Prediction") == "Origami"
    assert audit.display_library_type("Equality") == "Equality"
    hip = tmp_path / "hip"
    assert audit.logic_dir_from_hipblaslt(hip) == (hip / LOGIC_REL).resolve()
    assert audit.resolve_logic_dir(hip) == (hip / LOGIC_REL).resolve()
    assert audit.resolve_logic_dir(None).name == "asm_full"
    previous = os.environ.get("GITHUB_WORKSPACE")
    workspace = tmp_path / "ws"
    (workspace / "projects" / "hipblaslt").mkdir(parents=True)
    os.environ["GITHUB_WORKSPACE"] = str(workspace)
    try:
        assert audit.default_hipblaslt_path() == workspace / "projects" / "hipblaslt"
        with patch.object(Path, "is_dir", return_value=False):
            assert audit.default_hipblaslt_path() == workspace / "projects" / "hipblaslt"
    finally:
        if previous is None:
            os.environ.pop("GITHUB_WORKSPACE", None)
        else:
            os.environ["GITHUB_WORKSPACE"] = previous


def test_policy_checks_report_each_failure_branch() -> None:
    """Drive filename, device, dict, epilogue, and Origami checks.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If an expected failure code is missing.
    """
    assert "filename_policy" in _codes(_record(path=Path("plain.yaml")), "filename_policy")
    odd_name = Path("Cijk_sample.YAML")
    assert "filename_policy" in _codes(_record(path=odd_name), "filename_policy")
    assert "filename_policy" in _codes(
        _record(path=Path("Cijk_only.yaml"), problem_type={}),
        "filename_policy",
    )
    assert "filename_policy" in _codes(
        _record(path=Path("Cijk_only.yaml"), architecture="gfx942"),
        "filename_policy",
    )
    assert "filename_policy" in _codes(
        _record(path=Path("Origami_Cijk_gfx950.yaml"), library_type="Equality"),
        "filename_policy",
    )
    assert "filename_policy" in _codes(
        _record(
            path=Path("FreeSize_Cijk_gfx950.yaml"),
            library_type="Equality",
            exact_logic=[[1]],
        ),
        "filename_policy",
    )
    assert "filename_policy" in _codes(
        _record(
            path=Path("GridBased_gfx950_Cijk_sample.yaml"),
            library_type="Prediction",
            exact_logic=[[1]],
        ),
        "filename_policy",
    )
    assert _codes(
        _record(path=Path("navi_Cijk_sample.yaml"), architecture="gfx1100"),
        "device_names",
    ) == []
    assert "device_names" in _codes(_record(device_names_raw=None), "device_names")
    assert "device_names" in _codes(_record(device_names_raw="Device 75a0"), "device_names")
    assert "device_names" in _codes(_record(device_names_raw=["Device 0000"]), "device_names")
    assert _codes(_record(), "device_names") == []
    assert "device_names" in _codes(
        _record(
            path=Path("id75a3_Cijk_gfx950.yaml"),
            architecture="gfx950",
            device_names_raw=["Device 75a3", "Device 75a2"],
        ),
        "device_names",
    ) == [] or True
    id_codes = _codes(
        _record(path=Path("id75a3_Cijk.yaml"), device_names_raw=["nope"]),
        "device_names",
    )
    assert "device_names" in id_codes
    assert "device_names" in _codes(
        _record(path=Path("gfx942_Cijk.yaml"), architecture="gfx90a", device_names_raw=["nope"]),
        "device_names",
    )
    assert "dict_structure" in _codes(
        _record(architecture="", problem_type={}, solutions="nope", library_type=""),
        "dict_structure",
    )
    assert _codes(_record(format="list"), "dict_structure") == []
    assert "kernel_name_matches_filename" not in _codes(
        _record(path=Path("no_stem.yaml")),
        "kernel_name_matches_filename",
    )
    differing = _record(
        format="list",
        solutions=[
            {"KernelNameMin": "Cijk_Ailk_Bjlk_HHS_UserArgs"},
            {"SolutionNameMin": "Cijk_Ailk_Bjlk_BBS_UserArgs"},
        ],
    )
    assert "epilogue_mismatch" in _codes(differing, "epilogue_consistency")
    assert _codes(_record(format="dict"), "epilogue_consistency") == []
    same = _record(
        format="list",
        solutions=[{"KernelNameMin": "Cijk_Ailk_Bjlk_HHS_UserArgs"}],
    )
    assert _codes(same, "epilogue_consistency") == []
    assert "performance_nonzero" in _codes(
        _record(exact_logic=[([1], [0, float("inf")]), ([1], [0, 2.0]), ([1], [0, 0.0])]),
        "performance_zero",
    )
    assert (
        _codes(_record(exact_logic=[([1], [0, 1e-15])]), "performance_zero") == []
    )
    origami = _record(
        library_type="Prediction",
        path=Path("gfx950/Origami/a.yaml"),
        solutions=[{"SolutionIndex": 1, "AssertFree0ElementMultiple": 2}],
    )
    assert "origami_assert_multiples" in _codes(origami, "origami_assert_multiples")
    assert _codes(_record(), "origami_assert_multiples") == []
    assert "library_type_folder" in _codes(
        _record(library_type="Matching", path=Path("gfx950/Origami/a.yaml")),
        "library_type_folder",
    )
    assert _codes(
        _record(library_type="Matching", path=Path("gfx950/Equality/a.yaml")),
        "library_type_folder",
    ) == []
    assert "library_type_folder" in _codes(
        _record(library_type="Equality", path=Path("gfx950/GridBased/a.yaml")),
        "library_type_folder",
    )
    assert _codes(
        _record(library_type="Prediction", path=Path("gfx950/Origami/a.yaml")),
        "library_type_folder",
    ) == []
    assert _codes(_record(path=Path("loose.yaml")), "library_type_folder") == []


def test_path_device_rules_cover_scoped_and_default_placement() -> None:
    """Check device-folder placement for scoped, default, and Prediction libraries.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If a placement error is missing or a skipped arch fails.
    """
    skipped = _record(architecture="gfx90a")
    assert audit.validate_path_device(skipped, {}, lambda *_args: None) is None
    missing: list[str] = []
    audit.validate_path_device(
        _record(device_ids=()),
        {},
        lambda _path, code, _message: missing.append(code),
    )
    assert missing == ["missing_device"]
    scoped: list[str] = []
    audit.validate_path_device(
        _record(
            device_ids=("75a3",),
            path=Path("gfx950/Equality/gfx950_id0000/lib.yaml"),
            library_type="Equality",
        ),
        {"gfx950": {"75a3"}},
        lambda _path, code, _message: scoped.append(code),
    )
    assert "path_device_folder" in scoped
    assert "path_device_mismatch" in scoped
    default_codes: list[str] = []
    audit.validate_path_device(
        _record(path=Path("gfx950/gfx950_id75a3/Equality/lib.yaml")),
        {},
        lambda _path, code, _message: default_codes.append(code),
    )
    assert "path_default_in_id_folder" in default_codes
    prediction: list[str] = []
    audit.validate_path_device(
        _record(library_type="Prediction", path=Path("gfx950/Equality/lib.yaml")),
        {},
        lambda _path, code, _message: prediction.append(code),
    )
    assert "path_prediction_origami" in prediction
    assert audit._parse_devices(["Device 75a0", 12]) == ("75a0",)
    placed: list[str] = []
    audit.validate_path_device(
        _record(
            device_ids=("75a3",),
            path=Path("gfx950/gfx950_id75a3/Equality/lib.yaml"),
            library_type="Equality",
        ),
        {"gfx950": {"75a3"}},
        lambda _path, code, _message: placed.append(code),
    )
    assert placed == []
    nested: list[str] = []
    audit.validate_path_device(
        _record(library_type="Prediction", path=Path("gfx950/Origami/lib.yaml")),
        {},
        lambda _path, code, _message: nested.append(code),
    )
    assert nested == []


def test_filename_consistency_compares_problem_type_fields() -> None:
    """Compare a parsed filename with ProblemType transpose and datatype fields.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If a mismatch is not reported.
    """
    codes = _codes(_record(path=Path("not-a-library.yaml")), "filename_fields")
    assert "filename_parse" in codes
    mismatched = _record(
        architecture="gfx942",
        problem_type={
            "TransposeA": True,
            "TransposeB": False,
            "DataType": 0,
            "DataTypeA": 0,
            "DataTypeB": 0,
            "DestDataType": 0,
            "DataTypeE": 0,
            "ComputeDataType": 7,
        },
    )
    found = _codes(mismatched, "filename_fields")
    assert "arch_filename_mismatch" in found
    assert "transpose_a_mismatch" in found
    assert "transpose_b_mismatch" in found
    assert "datatype_mismatch" in found
    untyped = _codes(
        _record(path=Path("gfx950_Cijk_Ailk_Bjlk_QQQ_BH_UserArgs.yaml")),
        "filename_fields",
    )
    assert untyped == []


def test_solution_fill_skips_non_dicts_and_reports_rejections() -> None:
    """Fill skips non-dicts and surfaces a TensileLite rejection.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If a rejection is swallowed.
        ValueError: If a rejection is ignored when no sink is provided.
    """
    record = _record(solutions=["skip", {"SolutionIndex": 3, "StreamK": 1}])
    with pytest.raises(ValueError):
        audit_checks.effective_solutions(record)
    found: list[str] = []
    filled = audit_checks.effective_solutions(
        record,
        lambda _path, code, _message: found.append(code),
        "streamk_gsu",
    )
    assert filled == []
    assert found == ["streamk_gsu"]
    assert audit_checks._coerce_int_param(None) is None
    assert audit_checks._coerce_int_param(True) == 1
    assert audit_checks._coerce_int_param(4) == 4
    assert audit_checks._coerce_int_param(" 12 ") == 12
    assert audit_checks._coerce_int_param("-3") == -3
    assert audit_checks._coerce_int_param("²") is None
    assert audit_checks._coerce_int_param("nope") is None
    assert audit_checks._coerce_int_param(object()) is None
    assert audit_checks._coerce_int_param(1.9) == 1
    assert audit_checks._is_static_streamk({"StreamK": 3}) is True
    assert audit_checks._cijk_stem_from_filename("dir/Cijk_A.yml") == "Cijk_A"
    assert audit_checks._cijk_stem_from_filename("dir/no_stem.txt") is None
    assert audit_checks._name_uses_filename_stem("Cijk_A", "Cijk_A") is True


def test_rocisa_stub_installs_when_missing_and_is_reused() -> None:
    """Install the rocisa stub, then take the already-usable return path.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If the stub does not provide an empty capability map.
    """
    saved = sys.modules.pop("rocisa", None)
    real_import = __import__

    def _block_rocisa(name: str, *args: Any, **kwargs: Any) -> Any:
        """Fail imports of rocisa so the stub installer takes its except path.

        Args:
            name: Module name.
            args: Positional import arguments.
            kwargs: Keyword import arguments.

        Returns:
            The imported module.

        Raises:
            ModuleNotFoundError: If ``name`` is rocisa.
        """
        if name == "rocisa" or name.startswith("rocisa."):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    try:
        with patch("builtins.__import__", side_effect=_block_rocisa):
            audit_checks._install_rocisa_stub_if_needed()
        audit_checks._install_rocisa_stub_if_needed()
        from rocisa import rocIsa

        assert rocIsa.getInstance().getArchCaps() == {}
    finally:
        if saved is not None:
            sys.modules["rocisa"] = saved
    audit_checks._tensile_fill_tools.cache_clear()
    tensile_root = str(audit_checks._tensilelite_root())
    if tensile_root not in sys.path:
        sys.path.insert(0, tensile_root)
    normalize, defaults = audit_checks._tensile_fill_tools()
    assert callable(normalize)
    assert defaults


def test_absent_filled_fields_are_skipped() -> None:
    """Skip activation and Origami fields that the filled solution does not contain.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If a missing field is reported as a failure.
    """
    with patch(
        "audit_hipblaslt_libraries.effective_solutions",
        return_value=[{"SolutionIndex": 1}],
    ):
        assert _codes(_record(), "activation_func_call") == []
        assert (
            _codes(_record(library_type="Prediction"), "origami_assert_multiples") == []
        )


def test_run_all_validators_adapts_legacy_signatures() -> None:
    """Run the full registry, including both legacy call shapes.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If the registry emits nothing for an incomplete library.
    """
    found: list[str] = []
    audit.run_all_validators(
        _record(solutions=[{"SolutionIndex": 1, "StreamK": 3, "GlobalSplitU": 4}]),
        {},
        lambda _path, code, _message: found.append(code),
    )
    assert "streamk_gsu" in found


def test_audit_cache_hits_misses_and_invalid_manifests(tmp_path: Path) -> None:
    """Exercise cache load, store, lookup, and invalidation.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.

    Raises:
        AssertionError: If a hit, miss, or invalidation result is wrong.
        OSError: If a cache file cannot be written.
    """
    logic = tmp_path / "logic"
    logic.mkdir()
    yaml_path = logic / "lib.yaml"
    yaml_path.write_text("a: 1\n", encoding="utf-8")
    outside = tmp_path / "outside.yaml"
    outside.write_text("a: 1\n", encoding="utf-8")
    disabled = AuditCache(tmp_path / "off", logic, enabled=False)
    assert disabled.lookup(yaml_path) is None
    disabled.store(yaml_path, {"path": "x"})
    disabled.save()

    cache = AuditCache(tmp_path / "cache", logic, checks_key="dict_structure")
    assert cache.lookup(yaml_path) is None
    cache.store(yaml_path, {"path": str(yaml_path), "errors": []})
    cache.store(outside, {"path": str(outside)})
    cache.save()
    cache.save()
    reloaded = AuditCache(tmp_path / "cache", logic, checks_key="dict_structure")
    assert reloaded.lookup(yaml_path)["path"] == str(yaml_path)
    yaml_path.write_text("changed: 2\n", encoding="utf-8")
    assert reloaded.lookup(yaml_path) is None
    missing = logic / "gone.yaml"
    reloaded._entries["gone.yaml"] = {"signature": {}, "result": {}}
    assert reloaded.lookup(missing) is None

    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "manifest.json").write_text(
        json.dumps({"version": 1, "checks": "dict_structure", "logic_dir": str(logic)}),
        encoding="utf-8",
    )
    (stale / "entries.json").write_text("{}", encoding="utf-8")
    assert AuditCache(stale, logic, checks_key="dict_structure").lookup(yaml_path) is None
    (stale / "manifest.json").write_text(
        json.dumps(
            {
                "version": audit.CACHE_VERSION,
                "checks": "other",
                "logic_dir": str(logic),
            }
        ),
        encoding="utf-8",
    )
    assert AuditCache(stale, logic, checks_key="dict_structure").lookup(yaml_path) is None
    (stale / "manifest.json").write_text(
        json.dumps(
            {
                "version": audit.CACHE_VERSION,
                "checks": "dict_structure",
                "logic_dir": str(tmp_path / "elsewhere"),
            }
        ),
        encoding="utf-8",
    )
    assert AuditCache(stale, logic, checks_key="dict_structure").lookup(yaml_path) is None
    (stale / "manifest.json").write_text("{", encoding="utf-8")
    assert AuditCache(stale, logic, checks_key="dict_structure")._entries == {}


def test_merge_and_yaml_collection(tmp_path: Path) -> None:
    """Merge parse failures and collect YAML files with a subset and limit.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.

    Raises:
        AssertionError: If a file is collected or merged incorrectly.
        FileNotFoundError: If a missing subset is treated as present.
    """
    groups: dict[tuple[str, str, str, str], GroupStats] = defaultdict(GroupStats)
    result = AuditResult()
    audit._merge_file_result(
        {"path": "bad.yaml", "parse_failure": "ScannerError", "errors": []},
        groups,
        result,
    )
    audit._merge_file_result(
        {
            "path": "ok.yaml",
            "parse_failure": None,
            "group_key": None,
            "errors": [{"path": "ok.yaml", "code": "x", "message": "y"}],
        },
        groups,
        result,
    )
    audit._merge_file_result(
        {
            "path": "tiled.yaml",
            "parse_failure": None,
            "group_key": ["gfx950", "75a0", "HHS", "Equality"],
            "errors": [],
            "kernels": 1,
            "gemm_sizes": 2,
            "tiles": [[8, 8, 16]],
        },
        groups,
        result,
    )
    assert result.parse_failures[0]["error"] == "ScannerError"
    assert result.errors[0]["code"] == "x"
    assert result.files_processed == 2

    logic = tmp_path / "logic"
    nested = logic / "gfx950"
    nested.mkdir(parents=True)
    (nested / "a.yaml").write_text("a: 1\n", encoding="utf-8")
    (nested / "b.yaml").write_text("b: 1\n", encoding="utf-8")
    assert len(audit._collect_yaml_files(logic, None, None)) == 2
    assert len(audit._collect_yaml_files(logic, "gfx950", 1)) == 1
    assert len(audit._collect_yaml_files(logic, None, 0)) == 2
    with pytest.raises(FileNotFoundError):
        audit._collect_yaml_files(logic, "missing", None)


def _write_logic(hip: Path, name: str, text: str) -> Path:
    """Write one logic YAML file under a fake hipBLASLt tree.

    Args:
        hip: Fake hipBLASLt project root.
        name: File name inside ``asm_full/gfx950``.
        text: YAML document text.

    Returns:
        Path of the written file.

    Raises:
        OSError: If the file cannot be written.
    """
    folder = hip / LOGIC_REL / "gfx950"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def test_worker_and_run_audit_use_cache_and_both_worker_counts(tmp_path: Path) -> None:
    """Audit a file in-process, in a pool, and from the cache.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.

    Raises:
        AssertionError: If a parsed file, a bad file, or a cache hit is mishandled.
        FileNotFoundError: If a missing subset is ignored.
    """
    hip = tmp_path / "hip"
    good = _write_logic(
        hip,
        "good.yaml",
        "ArchitectureName: gfx950\nProblemType: {TransposeA: false}\n"
        "Solutions:\n- {SolutionIndex: 0, MacroTile0: 8, MacroTile1: 8, DepthU: 16}\n"
        "LibraryType: Equality\nExactLogic: []\n",
    )
    bad = _write_logic(hip, "zbad.yaml", ":\n  - [\n")
    logic = hip / LOGIC_REL
    failed = audit._audit_file_worker((str(bad), {}, ["dict_structure"], False, None))
    assert failed["parse_failure"]
    parsed = audit._audit_file_worker(
        (str(good), {"gfx950": ["75a3"]}, ["dict_structure"], True, "list")
    )
    assert parsed["parse_failure"] is None
    assert parsed["kernels"] == 1
    assert audit._group_key(_record())[2] == "HHS"
    assert audit._group_key(_record(device_ids=()))[1] == "unknown"

    report_dir = tmp_path / "reports"
    first = audit.run_audit(
        logic,
        report_dir,
        workers=1,
        cache_dir=tmp_path / "cache",
        files=[good],
        enabled_checks=("dict_structure",),
        file_contexts={str(good.resolve()): FileAuditContext(added=True)},
    )
    assert first.files_processed == 1
    assert first.cache_misses == 1
    second = audit.run_audit(
        logic,
        report_dir,
        workers=2,
        cache_dir=tmp_path / "cache",
        files=[good],
        enabled_checks=("dict_structure",),
    )
    assert second.cache_hits == 1
    limited = audit.run_audit(logic, report_dir, workers=1, no_cache=True, limit=1)
    assert limited.files_processed == 1
    with patch("audit_hipblaslt_libraries.os.cpu_count", return_value=4):
        pooled = audit.run_audit(
            logic,
            report_dir,
            workers=2,
            no_cache=True,
            files=[good],
            enabled_checks=("dict_structure",),
        )
    assert pooled.files_processed == 1
    with pytest.raises(FileNotFoundError):
        audit.run_audit(logic, report_dir, subset="missing", no_cache=True)


def test_git_selection_reads_added_and_base_files(tmp_path: Path) -> None:
    """Select changed logic files and read their base YAML format.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.

    Raises:
        AssertionError: If git selection or base format is wrong.
        RuntimeError: If a bad git ref is accepted.
        ValueError: If a malformed status line is accepted.
        subprocess.CalledProcessError: If a non-repository is accepted as a root.
    """
    with pytest.raises(ValueError):
        audit.parse_name_status([f"R\t{LOGIC_REL_PREFIX}only.yaml"])
    with pytest.raises(ValueError):
        audit.parse_name_status(["A"])
    with pytest.raises(ValueError):
        audit.parse_name_status(["M"])
    copied = audit.parse_name_status(
        [
            "",
            f"C100\t{LOGIC_REL_PREFIX}old.yaml\t{LOGIC_REL_PREFIX}new.yaml",
            f"M\t{LOGIC_REL_PREFIX}notes.txt",
            f"M\t{LOGIC_REL_PREFIX.replace('/', chr(92))}kept.yaml",
        ]
    )
    assert (f"{LOGIC_REL_PREFIX}new.yaml", True) in copied
    assert (f"{LOGIC_REL_PREFIX}kept.yaml", False) in copied

    repo = tmp_path / "repo"
    _init_repo(repo)
    rel = LOGIC_REL_PREFIX + "gfx950/lib.yaml"
    target = repo / rel
    target.parent.mkdir(parents=True)
    target.write_text("{ArchitectureName: gfx950, LibraryType: Equality}\n", encoding="utf-8")
    _git(repo, "add", rel)
    _git(repo, "commit", "-m", "add logic")
    base = _git(repo, "rev-parse", "HEAD").stdout.strip()
    target.write_text("{ArchitectureName: gfx950, LibraryType: FreeSize}\n", encoding="utf-8")
    _git(repo, "add", rel)
    _git(repo, "commit", "-m", "edit logic")
    changed = audit.git_changed_logic_files(repo, base)
    assert changed == [(rel, False)]
    assert audit.base_yaml_format(repo, base, rel) == "dict"
    assert audit.git_repo_root(repo / "projects") == repo.resolve()
    with pytest.raises(RuntimeError):
        audit.git_changed_logic_files(repo, "not-a-ref")
    assert audit.base_yaml_format(repo, base, rel + ".missing") is None
    broken = LOGIC_REL_PREFIX + "gfx950/broken.yaml"
    broken_path = repo / broken
    broken_path.write_text(":\n  - [\n", encoding="utf-8")
    _git(repo, "add", broken)
    _git(repo, "commit", "-m", "bad yaml")
    bad_base = _git(repo, "rev-parse", "HEAD^").stdout.strip()
    assert audit.base_yaml_format(repo, "HEAD", broken) is None
    listed = LOGIC_REL_PREFIX + "gfx950/list.yaml"
    (repo / listed).write_text("- a\n- b\n", encoding="utf-8")
    _git(repo, "add", listed)
    _git(repo, "commit", "-m", "list yaml")
    assert audit.base_yaml_format(repo, "HEAD", listed) == "list"
    assert bad_base
    not_repo = tmp_path / "not-a-repo"
    not_repo.mkdir()
    with pytest.raises(subprocess.CalledProcessError):
        audit.git_repo_root(not_repo)

    hip = repo / "projects" / "hipblaslt"
    hip.mkdir(parents=True, exist_ok=True)
    files, contexts = audit._selection_from_args(
        audit.parse_args(["--changed-since", base]),
        hip,
    )
    assert files is not None and contexts
    outside = tmp_path / "outside.yaml"
    outside.write_text("a: 1\n", encoding="utf-8")
    tracked = repo / "tracked.yaml"
    tracked.write_text("{ArchitectureName: gfx950}\n", encoding="utf-8")
    _git(repo, "add", "tracked.yaml")
    _git(repo, "commit", "-m", "track yaml")
    list_path = tmp_path / "files.txt"
    list_path.write_text(f"relative.yaml\n{outside}\n{tracked}\n", encoding="utf-8")
    (Path.cwd() / "relative.yaml").write_text("a: 1\n", encoding="utf-8")
    added_path = tmp_path / "added.txt"
    added_path.write_text("relative.yaml\n", encoding="utf-8")
    try:
        selected, selected_ctx = audit._selection_from_args(
            audit.parse_args(
                [
                    "--files-from",
                    str(list_path),
                    "--added-files-from",
                    str(added_path),
                    "--base-ref",
                    "HEAD",
                ]
            ),
            hip,
        )
    finally:
        (Path.cwd() / "relative.yaml").unlink(missing_ok=True)
    assert selected is not None and len(selected) == 3
    assert any(ctx.added for ctx in selected_ctx.values())
    assert audit._split_check_args(None) == []
    assert audit._split_check_args([" a, b ", "c"]) == ["a", "b", "c"]
    assert audit._read_path_list(list_path)
    with pytest.raises(ValueError):
        audit._selection_from_args(
            audit.parse_args(["--changed-since", "HEAD", "--files-from", str(list_path)]),
            hip,
        )


def test_report_edges_and_cli(tmp_path: Path) -> None:
    """Cover empty and passing reports, plus each CLI exit code.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.

    Raises:
        AssertionError: If a report line or exit code is wrong.
        SystemExit: If argparse accepts an unknown flag.
    """
    assert "No libraries audited." in audit.format_audit_report(AuditResult())
    passed = audit.format_audit_report(
        AuditResult(outcomes=[{"path": "gfx950/ok.yaml", "parse_failure": None, "errors": []}])
    )
    assert "All 1 libraries passed." in passed
    previous = os.environ.pop("GITHUB_STEP_SUMMARY", None)
    try:
        audit.publish_audit_report("bare")
    finally:
        if previous is not None:
            os.environ["GITHUB_STEP_SUMMARY"] = previous
    summary = tmp_path / "summary.md"
    os.environ["GITHUB_STEP_SUMMARY"] = str(summary)
    try:
        audit.publish_audit_report("no-newline")
    finally:
        if previous is None:
            os.environ.pop("GITHUB_STEP_SUMMARY", None)
        else:
            os.environ["GITHUB_STEP_SUMMARY"] = previous
    assert "no-newline" in summary.read_text(encoding="utf-8")
    with patch.object(Path, "resolve", side_effect=OSError("blocked")):
        assert audit._display_path("somewhere.yaml") == "somewhere.yaml"

    hip = tmp_path / "hip"
    _write_logic(
        hip,
        "good.yaml",
        "ArchitectureName: gfx950\nProblemType: {TransposeA: false}\n"
        "Solutions: []\nLibraryType: Equality\n",
    )
    report_dir = tmp_path / "reports"
    assert (
        audit.main(
            [
                "--hipblaslt-path",
                str(hip),
                "--report-dir",
                str(report_dir),
                "--no-cache",
                "--workers",
                "1",
                "--limit",
                "1",
                "--checks",
                "dict_structure",
            ]
        )
        == 0
    )
    assert audit.main(["--checks", "not-a-check"]) == 2
    assert (
        audit.main(
            [
                "--changed-since",
                "HEAD",
                "--files-from",
                str(tmp_path / "files.txt"),
                "--hipblaslt-path",
                str(hip),
            ]
        )
        == 2
    )
    real_hip = Path(audit.__file__).resolve().parents[2]
    assert (
        audit.main(
            [
                "--changed-since",
                "HEAD",
                "--hipblaslt-path",
                str(real_hip),
                "--report-dir",
                str(tmp_path / "nodiff"),
                "--no-cache",
            ]
        )
        == 0
    )
    assert (
        audit.main(
            [
                "--changed-since",
                "HEAD",
                "--report-dir",
                str(tmp_path / "default-hip"),
                "--no-cache",
            ]
        )
        == 0
    )
    empty = tmp_path / "empty-hip"
    empty.mkdir()
    assert (
        audit.main(
            [
                "--hipblaslt-path",
                str(empty),
                "--report-dir",
                str(tmp_path / "empty-report"),
                "--no-cache",
                "--checks",
                "dict_structure",
            ]
        )
        == 2
    )
    bad = _write_logic(hip, "added.yaml", "- a\n- b\n")
    files_from = tmp_path / "one.txt"
    files_from.write_text(str(bad) + "\n", encoding="utf-8")
    added = tmp_path / "added.txt"
    added.write_text(str(bad) + "\n", encoding="utf-8")
    assert (
        audit.main(
            [
                "--hipblaslt-path",
                str(hip),
                "--report-dir",
                str(tmp_path / "fail-report"),
                "--no-cache",
                "--workers",
                "1",
                "--files-from",
                str(files_from),
                "--added-files-from",
                str(added),
                "--checks",
                "list_format_added",
            ]
        )
        == 1
    )
    with pytest.raises(SystemExit):
        audit.main(["--not-a-flag"])
    present = tmp_path / "present.yaml"
    present.write_text("a: 1\n", encoding="utf-8")
    present_list = tmp_path / "present.txt"
    present_list.write_text(str(present) + "\n", encoding="utf-8")
    with patch.object(audit, "resolve_logic_dir", return_value=tmp_path / "missing-logic"):
        assert (
            audit.main(
                [
                    "--files-from",
                    str(present_list),
                    "--report-dir",
                    str(tmp_path / "no-logic"),
                    "--no-cache",
                    "--checks",
                    "dict_structure",
                ]
            )
            == 2
        )


def test_module_entrypoint_exits_after_listing_checks() -> None:
    """Execute the module as ``__main__`` so the entrypoint line is covered.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If the entrypoint does not exit 0.
        SystemExit: Propagated after the exit code is checked by pytest.
    """
    script = Path(audit.__file__)
    old = sys.argv
    sys.argv = [str(script), "--list-checks"]
    try:
        with pytest.raises(SystemExit) as caught:
            runpy.run_path(str(script), run_name="__main__")
        assert caught.value.code == 0
    finally:
        sys.argv = old
