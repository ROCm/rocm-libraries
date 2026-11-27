"""The HIP MLOps ingestor engines' lanes agree with their descriptors and category YAMLs.

Each HKP_MLOPS_INGESTOR_ENGINES row in src/CMakeLists.txt names a HipMlops descriptor
bundle, its engine and the bundle folders of the graphs that engine serves. The engine's
external integration check passes a GTEST_FILTER over those folders at every tier, and
under YAML categorization its CTest suites take theirs from
hipkernel_<engine>_test_categories_integration.yaml instead, one per category. Both must
name the same folders, so editing one without the other fails here. Every HipMlops engine
must have a row, every row its TOML, and every graph-capture engine a row.

A pattern is "<tier>_<folder>_*.*", <tier> one of the four bundle tier directories or "*"
for all of them. Each category selects every folder at each tier it runs, and the
categories together run all four tiers, as the CMake filter does. Excludes are not
compared: the CMake filter has none, and the YAMLs' (DISABLED) narrow no folder.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.quick

PROVIDER = Path(__file__).resolve().parents[2]
SRC_CMAKE = PROVIDER / "src/CMakeLists.txt"
HIP_MLOPS = PROVIDER / "src/engines/kernel_ingestor_engine/descriptors/HipMlops"
BUNDLES = PROVIDER.parent / "integration-tests/integration-test-bundles"
TABLE = "HKP_MLOPS_INGESTOR_ENGINES"
GRAPH_CAPTURE = "_hkp_mlops_graph_capture_engines"
ROW_FIELDS = 3
TIERS = ("quick", "standard", "comprehensive", "full")
PATTERN = re.compile(
    r"(?P<tier>quick|standard|comprehensive|full|\*)_(?P<folder>\w+?)_\*\.\*"
)


def cmake_list(name):
    """The fields of the one set(<name> ...) in src/CMakeLists.txt, comments dropped."""
    text = SRC_CMAKE.read_text(encoding="utf-8")
    calls = re.findall(rf"\bset\(\s*{name}\s([^)]*)\)", text)
    assert len(calls) == 1, f"{SRC_CMAKE}: {name} set {len(calls)} times"
    return re.sub(r"#[^\n]*", "", calls[0]).split()


def mlops_engines():
    """The table's rows: descriptor slug, engine local name, bundle folders."""
    fields = cmake_list(TABLE)
    assert (
        fields and len(fields) % ROW_FIELDS == 0
    ), f"{SRC_CMAKE}: {TABLE} has {len(fields)} fields, not rows of {ROW_FIELDS}"
    return [
        (fields[i], fields[i + 1], fields[i + 2].split(","))
        for i in range(0, len(fields), ROW_FIELDS)
    ]


ROWS = mlops_engines()


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row[1])
def test_each_row_names_its_bundle_engine_toml_and_folders(row):
    """A misspelt slug or engine is offered by no product, and its lane drops out."""
    slug, local, folders = row
    ued = HIP_MLOPS / slug / f"{slug}.ued.json"
    assert ued.is_file(), f"{SRC_CMAKE}: {TABLE} {local}: no {ued}"
    name = json.loads(ued.read_text(encoding="utf-8"))["name"]
    assert name == f"hipkernel:{local}", f"{ued}: engine {name!r}, row says {local}"
    config = PROVIDER / "config" / f"hipkernel_{local}.toml"
    assert config.is_file(), f"{SRC_CMAKE}: {TABLE} {local}: no {config}"
    missing = [
        f for f in folders if not any((BUNDLES / t / f).is_dir() for t in TIERS)
    ]
    assert not missing, f"{SRC_CMAKE}: {TABLE} {local}: no bundle folder {missing}"


def test_every_hip_mlops_engine_has_a_row():
    """A new HipMlops engine gets a lane, not silence."""
    declared = {
        json.loads(ued.read_text(encoding="utf-8"))["name"]
        for ued in HIP_MLOPS.glob("*/*.ued.json")
    }
    assert declared == {f"hipkernel:{local}" for _, local, _ in ROWS}


def test_graph_capture_engines_have_rows():
    unknown = set(cmake_list(GRAPH_CAPTURE)) - {local for _, local, _ in ROWS}
    assert (
        not unknown
    ), f"{SRC_CMAKE}: {GRAPH_CAPTURE} names {sorted(unknown)}, not in {TABLE}"


def selection(patterns, where):
    """{(tier, folder)} the positive gtest `patterns` select; a "*" tier is all four."""
    pairs = set()
    for pattern in patterns:
        match = PATTERN.fullmatch(pattern)
        assert match, f"{where}: {pattern!r} is not <tier>_<folder>_*.*"
        tiers = TIERS if match["tier"] == "*" else (match["tier"],)
        pairs.update((tier, match["folder"]) for tier in tiers)
    return pairs


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row[1])
def test_categories_select_the_row_folders(row):
    _slug, local, folders = row
    expected = {(tier, folder) for tier in TIERS for folder in folders}

    path = PROVIDER / f"hipkernel_{local}_test_categories_integration.yaml"
    assert path.is_file(), f"{SRC_CMAKE}: {TABLE} {local}: no {path}"
    categories = yaml.safe_load(path.read_text(encoding="utf-8"))["test_categories"]
    assert categories, f"{path}: no test_categories"
    union = set()
    for category, info in categories.items():
        selected = selection(info["test_patterns"], f"{path} {category}")
        tiers = {tier for tier, _ in selected}
        want = {(tier, folder) for tier in tiers for folder in folders}
        assert selected == want, (
            f"{path} {category}: at tiers {sorted(tiers)}, missing "
            f"{sorted(want - selected)} and extra {sorted(selected - want)} against "
            f"{local}'s row in {SRC_CMAKE}"
        )
        union |= selected
    assert union == expected, (
        f"{path}: categories select {sorted(union)}; "
        f"{local}'s row in {SRC_CMAKE} selects {sorted(expected)}"
    )
