"""The attention engines' bundle selection agrees between CMake and the category YAMLs.

src/CMakeLists.txt defines the GTEST_FILTER each attention engine's external integration
check (and its GPU-reference target in src/integration_tests/) passes. Under YAML
categorization the check's CTest suites take theirs from the engine's
hipkernel_*_test_categories_integration.yaml instead, one per category. Both must name
the same bundles, so editing one without the other fails here.

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
SRC_CMAKE = PROVIDER / "src/CMakeLists.txt"

# Each attention engine's category YAML and the src/CMakeLists.txt filter its check passes.
YAML_FILTERS = {
    "hipkernel_Gfx950Attention_test_categories_integration.yaml": (
        "HKP_SDPA_FWD_DEFAULT_GTEST_FILTER"
    ),
    "hipkernel_Gfx942AttentionUnified3d_test_categories_integration.yaml": (
        "HKP_SDPA_FWD_DEFAULT_GTEST_FILTER"
    ),
    "hipkernel_Gfx942AttentionDense_test_categories_integration.yaml": (
        "HKP_GFX942_ATTENTION_DENSE_GTEST_FILTER"
    ),
}

TIERS = ("quick", "standard", "comprehensive", "full")
TIERED = re.compile(r"(?P<tier>quick|standard|comprehensive|full|\*)_(?P<bundle>.+)")


def cmake_filter(name):
    """`name`'s value as src/CMakeLists.txt defines it, ${...} references expanded."""
    text = SRC_CMAKE.read_text(encoding="utf-8")
    raw = {}
    for var in set(YAML_FILTERS.values()):
        # set(<var> "<arg>"...) or string(JOIN "<glue>" <var> "<arg>"...).
        calls = re.findall(
            rf'\b(?:set|string)\(\s*(?:JOIN\s+"([^"]*)"\s+)?{var}\s([^)]*)\)', text
        )
        assert len(calls) == 1, f"{SRC_CMAKE}: {var} defined {len(calls)} times"
        glue, args = calls[0]
        # A set() list reaches --gtest_filter ':'-joined, as string(JOIN ":") does.
        raw[var] = (glue or ":").join(re.findall(r'"([^"]*)"', args))

    def expand(value):
        return re.sub(r"\$\{(\w+)\}", lambda ref: expand(raw[ref[1]]), value)

    return expand(raw[name])


def selection(patterns, where):
    """{(tier, bundle)} the positive gtest `patterns` select; a "*" tier is all four."""
    pairs = set()
    for pattern in patterns:
        match = TIERED.fullmatch(pattern)
        assert match, f"{where}: {pattern!r} has no tier prefix ({', '.join(TIERS)}, *)"
        tiers = TIERS if match["tier"] == "*" else (match["tier"],)
        pairs.update((tier, match["bundle"]) for tier in tiers)
    return pairs


def test_every_attention_category_yaml_is_compared():
    """A new attention engine's YAML is checked against its filter, not skipped."""
    found = {
        path.name
        for path in PROVIDER.glob("hipkernel_*Attention*_test_categories_*.yaml")
    }
    assert found == set(YAML_FILTERS)


@pytest.mark.parametrize("yaml_name", sorted(YAML_FILTERS))
def test_categories_select_the_cmake_filter_bundles(yaml_name):
    name = YAML_FILTERS[yaml_name]
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
