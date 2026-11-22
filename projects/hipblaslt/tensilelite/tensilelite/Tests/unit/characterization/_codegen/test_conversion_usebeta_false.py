# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Beta-free conversion bodies/identities, without changing stock helpers.

Feature: gate the executable beta/C read on UseBeta for BespokeNarrowing
solutions and distinguish their beta-off helper symbols. Restoring
unconditional beta emission / shared names breaks it.
Feature: a UseBeta:false solution without BespokeNarrowing keeps the helper
names and source bytes the stock generator emits for it (the stock generator ignores
UseBeta here, so they equal its beta-on helpers); the host does not guard it, so it
must still apply a runtime beta. Applying the beta-free body to keyless solutions breaks it.
Preservation: leave beta-on helper names and source bytes exactly as the stock generator
emits through this same derivation path. Changing beta-on emission breaks it.
The fixture provenance pins three unchanged selected stock conversion recipes.
"""
import hashlib
import json
import re

import pytest

from conversion_helper_common import DATA, beta_variant, conversion_helpers

pytestmark = pytest.mark.unit
RECIPES = ("gfx950_TN_pgr2.yaml", "gfx950_NN_pgr0.yaml", "gfx942_NT_gsu_auto.yaml")


def executable(source):
    # Comments cannot satisfy or trip a code-path assertion.
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)


def digests(helpers):
    return {name: hashlib.sha256(source.encode()).hexdigest() for name, source in helpers.items()}


def base_digests(recipe):
    # Independent pinned stock-generator output; never regenerate expectations from the SUT.
    return json.loads((DATA / "stock_conversion_digests.json").read_text())[recipe]


@pytest.mark.parametrize("recipe", RECIPES)
def test_beta_off_conversion_has_no_c_read(tmp_path, recipe):
    path = beta_variant(DATA / recipe, tmp_path / "beta_off.yaml", False, bespoke=True)
    solution, helpers = conversion_helpers(path)
    assert solution["ProblemType"]["UseBeta"] is False
    for name, source in helpers.items():
        code = executable(source)
        assert not re.search(r"\barg\.C\b|\bptr\s*\[\s*idxC\b", code), \
            f"beta-off conversion {name} still contains a C read"
        assert not re.search(r"if\s*\(\s*arg\.beta\s*!=", code), \
            f"beta-off conversion {name} still contains the runtime beta branch"


@pytest.mark.parametrize("recipe", RECIPES)
def test_beta_off_conversion_names_do_not_alias_beta_on(tmp_path, recipe):
    _, general = conversion_helpers(DATA / recipe)
    path = beta_variant(DATA / recipe, tmp_path / "beta_off.yaml", False, bespoke=True)
    _, bespoke = conversion_helpers(path)
    assert len(general) == len(bespoke)
    assert set(general).isdisjoint(bespoke), \
        f"beta-on/off conversion helper symbol collision: {sorted(set(general) & set(bespoke))}"


@pytest.mark.parametrize("recipe", RECIPES)
def test_keyless_beta_off_conversion_keeps_develop_helpers(tmp_path, recipe):
    path = beta_variant(DATA / recipe, tmp_path / "keyless_beta_off.yaml", False, bespoke=False)
    solution, helpers = conversion_helpers(path)
    assert solution["ProblemType"]["UseBeta"] is False
    assert digests(helpers) == base_digests(recipe), \
        "keyless UseBeta:false conversion helper names/source bytes differ from the stock generator's"


@pytest.mark.parametrize("recipe", RECIPES)
def test_stock_conversion_names_and_bytes_preserved(recipe):
    _, helpers = conversion_helpers(DATA / recipe)
    assert digests(helpers) == base_digests(recipe), "beta-on stock conversion helper names/source bytes changed"
