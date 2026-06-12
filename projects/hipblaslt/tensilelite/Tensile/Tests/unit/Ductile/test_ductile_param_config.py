################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import pytest
import yaml

from Tensile.ductile import config

pytestmark = pytest.mark.unit


def test_load_without_path_contains_expected_defaults():
    defaults = config.load()
    assert defaults["algorithm"] == "GA"
    assert defaults["pop_size"] > 2
    assert defaults["n_gen"] > 0
    assert "selection" in defaults
    assert "crossover" in defaults
    assert "mutation" in defaults


def test_load_with_path_merges_with_defaults(tmp_path):
    cfg_path = tmp_path / "ductile_test.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "n_gen": 3,
                "selection": {
                    "name": "tournament",
                    "common": {"ratio": 0.25},
                    "tournament": {"k": 3},
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = config.load(str(cfg_path))
    assert loaded["n_gen"] == 3
    assert loaded["selection"]["name"] == "tournament"
    assert loaded["selection"]["common"]["ratio"] == 0.25
    # Retains defaults for sections omitted by the override file.
    assert "survival" in loaded
    assert "mutation" in loaded


def test_update_does_not_mutate_defaults():
    original = dict(config.DEFAULTS)
    updated = config.update({"n_gen": 2, "pop_size": 16})
    assert updated["n_gen"] == 2
    assert updated["pop_size"] == 16
    # DEFAULTS remains immutable for subsequent test cases.
    assert dict(config.DEFAULTS)["n_gen"] == original["n_gen"]
    assert dict(config.DEFAULTS)["pop_size"] == original["pop_size"]


def test_populate_requires_name_field():
    with pytest.raises(ValueError, match="missing 'name' field"):
        config.populate({"selection": {"common": {"ratio": 0.5}}}, "selection")


def test_populate_merges_selected_and_common_options():
    conf = {
        "selection": {
            "name": "tournament",
            "tournament": {"k": 2},
            "common": {"ratio": 0.4, "elitism": 0.1},
        }
    }
    section = config.populate(conf, "selection")
    assert section["name"] == "tournament"
    assert section["k"] == 2
    assert section["ratio"] == 0.4
    assert section["elitism"] == 0.1
