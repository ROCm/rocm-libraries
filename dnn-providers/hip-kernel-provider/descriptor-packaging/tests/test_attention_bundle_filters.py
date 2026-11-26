"""The attention engines' bundle selection agrees between CMake and the category YAMLs.

Each HKP_PINNED_ATTENTION_ENGINES row in src/CMakeLists.txt names the GTEST_FILTER
variable its engine's external integration check (and GPU-reference target in
src/integration_tests/) passes, and the ../CMakeLists.txt variable holding its
hipkernel_*_test_categories_integration.yaml, from which the check's CTest suites take
theirs under YAML categorization, one per category. An engine whose entry is written out
by hand names the same two variables in HKP_HANDWRITTEN_ATTENTION_ENGINES. Both must name
the same bundles, so editing one without the other fails here. Rows sharing a YAML must
share a filter, and every attention categories YAML must be some row's or pair's. Each
attention knob suite in src/integration_tests/CMakeLists.txt pins engines the table
names.

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
INTEGRATION_CMAKE = PROVIDER / "src/integration_tests/CMakeLists.txt"
KNOB_SUITES_DIR = PROVIDER / "src/integration_tests/kernel_ingestor_engine"
PACKS = PROVIDER / "src/tests/engines/kernel_ingestor_engine/packs"
TABLE = "HKP_PINNED_ATTENTION_ENGINES"
HANDWRITTEN = "HKP_HANDWRITTEN_ATTENTION_ENGINES"
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


def handwritten_engines():
    """HKP_HANDWRITTEN_ATTENTION_ENGINES' pairs: filter variable, categories YAML variable."""
    text = SRC_CMAKE.read_text(encoding="utf-8")
    calls = re.findall(rf"\bset\(\s*{HANDWRITTEN}\s([^)]*)\)", text)
    assert len(calls) == 1, f"{SRC_CMAKE}: {HANDWRITTEN} set {len(calls)} times"
    fields = re.sub(r"#[^\n]*", "", calls[0]).split()
    assert fields and len(fields) % 2 == 0, f"{SRC_CMAKE}: {HANDWRITTEN} {fields}"
    return [(fields[i], fields[i + 1]) for i in range(0, len(fields), 2)]


def yaml_filters():
    """{categories YAML: {filter variables of the rows and pairs naming it}}."""
    pairs = {}
    for *_, filter_var, _toml, yaml_var in pinned_engines():
        pairs.setdefault(categories_yaml(yaml_var), set()).add(filter_var)
    for filter_var, yaml_var in handwritten_engines():
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


@pytest.mark.parametrize("row", pinned_engines(), ids=lambda row: f"{row[0]}_{row[1]}")
def test_each_row_names_an_arch_its_toml_and_its_packs_suite(row):
    """A misspelt arch or engine local name ships nothing for the row, so its census,
    external integration check and GPU-reference target drop out silently; fail here."""
    arch, _slug, local, _filter_var, toml, _yaml_var = row
    assert re.fullmatch(r"gfx[0-9a-f]+", arch), f"{SRC_CMAKE}: {TABLE} arch {arch!r}"
    config = PROVIDER / "config" / toml
    assert config.is_file(), f"{SRC_CMAKE}: {TABLE} {local}: no {config}"
    suite = PACKS / f"Test{local}Packs.cpp"
    assert suite.is_file(), f"{SRC_CMAKE}: {TABLE} {local}: no {suite}"
    # The census keys its expected inventory by arch, so a wrong arch pins none.
    assert f'{{"{arch}",' in suite.read_text(
        encoding="utf-8"
    ), f"{SRC_CMAKE}: {TABLE} {local}: {suite.name} pins no {arch} inventory"


def knob_suites():
    """The _knob_suites rows of src/integration_tests/CMakeLists.txt: (source, engines)."""
    text = INTEGRATION_CMAKE.read_text(encoding="utf-8")
    calls = re.findall(r"\bset\(\s*_knob_suites\s([^)]*)\)", text)
    assert len(calls) == 1, f"{INTEGRATION_CMAKE}: _knob_suites set {len(calls)} times"
    fields = re.sub(r"#[^\n]*", "", calls[0]).split()
    assert (
        fields and len(fields) % 2 == 0
    ), f"{INTEGRATION_CMAKE}: _knob_suites {fields}"
    return [(fields[i], fields[i + 1].split(",")) for i in range(0, len(fields), 2)]


@pytest.mark.parametrize(
    "source, engines",
    [pytest.param(source, engines, id=source) for source, engines in knob_suites()],
)
def test_each_knob_suite_pins_engines_the_table_names(source, engines):
    """A misspelt engine is offered by no product, so its suite compiles silently out."""
    assert (KNOB_SUITES_DIR / source).is_file(), f"{INTEGRATION_CMAKE}: no {source}"
    pinned = {local for _arch, _slug, local, *_ in pinned_engines()}
    unknown = sorted(set(engines) - pinned)
    assert not unknown, f"{INTEGRATION_CMAKE}: {source} pins {unknown}, not in {TABLE}"


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
