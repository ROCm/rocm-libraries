"""The attention engines' bundle selection agrees between CMake and the category YAMLs.

Each HKP_PINNED_ATTENTION_ENGINES row in src/CMakeLists.txt names the GTEST_FILTER
variable its engine's external integration check (and GPU-reference target in
src/integration_tests/) passes, and the ../CMakeLists.txt variable holding its
hipkernel_*_test_categories_integration.yaml, from which the check's CTest suites take
theirs under YAML categorization, one per category. Both must name the same bundles, so
editing one without the other fails here. Rows sharing a YAML must share a filter, and
every attention categories YAML must be some row's.

A pattern is "<tier>_<bundle>", <tier> one of the four bundle tier directories or "*"
for all of them. Each category selects every bundle the CMake filter does at each tier
it runs, and the categories together run every tier the CMake filter does.
Excludes are not compared: the CMake filters have none, and the YAMLs' (DISABLED,
SdpaBackward) reach no SdpaFwd bundle.
"""

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.quick

PROVIDER = Path(__file__).resolve().parents[2]
PROVIDER_CMAKE = PROVIDER / "CMakeLists.txt"
SRC_CMAKE = PROVIDER / "src/CMakeLists.txt"
TABLE = "HKP_PINNED_ATTENTION_ENGINES"
ROW_FIELDS = 6


def pinned_engines():
    """The table's rows: arch, slug, engine, filter, TOML, categories YAML variable."""
    text = SRC_CMAKE.read_text(encoding="utf-8")
    calls = re.findall(rf"\bset\(\s*{TABLE}\s([^)]*)\)", text)
    assert len(calls) == 1, f"{SRC_CMAKE}: {TABLE} set {len(calls)} times"
    fields = re.sub(r"#[^\n]*", "", calls[0]).split()
    assert (
        fields and len(fields) % ROW_FIELDS == 0
    ), f"{SRC_CMAKE}: {TABLE} has {len(fields)} fields, not rows of {ROW_FIELDS}"
    return [fields[i : i + ROW_FIELDS] for i in range(0, len(fields), ROW_FIELDS)]


def categories_yaml(var):
    """The file name ../CMakeLists.txt sets `var` to."""
    text = PROVIDER_CMAKE.read_text(encoding="utf-8")
    names = re.findall(
        rf'\bset\(\s*{var}\s+"\$\{{CMAKE_CURRENT_SOURCE_DIR\}}/([^"/]+)"\s*\)', text
    )
    assert len(names) == 1, f"{PROVIDER_CMAKE}: {var} set {len(names)} times"
    return names[0]


def yaml_filters():
    """{categories YAML: {filter variables of the rows naming it}}."""
    pairs = {}
    for *_, filter_var, _toml, yaml_var in pinned_engines():
        pairs.setdefault(categories_yaml(yaml_var), set()).add(filter_var)
    return pairs


PINNED = yaml_filters()

TIERS = ("quick", "standard", "comprehensive", "full")
TIERED = re.compile(r"(?P<tier>quick|standard|comprehensive|full|\*)_(?P<bundle>.+)")


def cmake_filter(name):
    """`name`'s value as src/CMakeLists.txt defines it, ${...} references expanded."""
    text = SRC_CMAKE.read_text(encoding="utf-8")
    # set(<var> "<arg>"...) or string(JOIN "<glue>" <var> "<arg>"...).
    calls = re.findall(
        rf'\b(?:set|string)\(\s*(?:JOIN\s+"([^"]*)"\s+)?{name}\s([^)]*)\)', text
    )
    assert len(calls) == 1, f"{SRC_CMAKE}: {name} defined {len(calls)} times"
    glue, args = calls[0]
    # A set() list reaches --gtest_filter ':'-joined, as string(JOIN ":") does.
    value = (glue or ":").join(re.findall(r'"([^"]*)"', args))
    return re.sub(r"\$\{(\w+)\}", lambda ref: cmake_filter(ref[1]), value)


def selection(patterns, where):
    """{(tier, bundle)} the positive gtest `patterns` select; a "*" tier is all four."""
    pairs = set()
    for pattern in patterns:
        match = TIERED.fullmatch(pattern)
        assert match, f"{where}: {pattern!r} has no tier prefix ({', '.join(TIERS)}, *)"
        tiers = TIERS if match["tier"] == "*" else (match["tier"],)
        pairs.update((tier, match["bundle"]) for tier in tiers)
    return pairs


def test_rows_sharing_a_categories_yaml_share_a_filter():
    """One YAML selects one set of bundles, so it can agree with one filter only."""
    shared = {
        name: sorted(filters) for name, filters in PINNED.items() if len(filters) > 1
    }
    assert not shared, f"{SRC_CMAKE}: {TABLE} rows pair a YAML with filters {shared}"


def test_every_attention_category_yaml_is_compared():
    """A new attention engine's YAML is checked against its filter, not skipped."""
    found = {
        path.name
        for path in PROVIDER.glob("hipkernel_*Attention*_test_categories_*.yaml")
    }
    assert found == set(PINNED)


@pytest.mark.parametrize(
    "yaml_name, name",
    sorted((yaml_name, name) for yaml_name, names in PINNED.items() for name in names),
)
def test_categories_select_the_cmake_filter_bundles(yaml_name, name):
    positive = cmake_filter(name).partition("-")[0]
    expected = selection(positive.split(":"), f"{SRC_CMAKE} {name}")

    path = PROVIDER / yaml_name
    categories = yaml.safe_load(path.read_text(encoding="utf-8"))["test_categories"]
    assert categories, f"{path}: no test_categories"
    union = set()
    for category, info in categories.items():
        selected = selection(info["test_patterns"], f"{path} {category}")
        tiers = {tier for tier, _ in selected}
        want = {(tier, bundle) for tier in tiers for _, bundle in expected}
        assert selected == want, (
            f"{path} {category}: at tiers {sorted(tiers)}, missing "
            f"{sorted(want - selected)} and extra {sorted(selected - want)} against "
            f"{name} in {SRC_CMAKE}"
        )
        union |= selected
    assert union == expected, (
        f"{path}: categories select {sorted(union)}; "
        f"{name} in {SRC_CMAKE} selects {sorted(expected)}"
    )
