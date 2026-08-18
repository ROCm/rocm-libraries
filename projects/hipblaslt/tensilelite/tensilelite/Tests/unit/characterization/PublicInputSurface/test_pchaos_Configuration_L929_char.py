################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""
Characterization tests for Tensile/Configuration.py:929
    if self.ConstraintsKey in self:   (inside ProjectConfig.checkConstraints)

Branch ID: 5e52e9474f011ad59cd92b98ce1af6cfe93e980b
Predicate:  self.ConstraintsKey in self
  - self.ConstraintsKey == "_Constraints"  (Configuration.py:818)
  - `in self` dispatches to ProjectConfig.__contains__  (Configuration.py:848),
    a dict membership test over the object and its flattened view.

TRUE branch:  addConstraint() was called -> "_Constraints" key is present
              -> checkConstraints() iterates and returns a result.
FALSE branch: no constraints were ever added -> key absent
              -> latent UnboundLocalError on `return value` (line 935),
                 because `value` is only bound inside the TRUE-branch loop.

Both behaviors are pinned ACTUAL (not corrected).
CPU-only; no GPU required.
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure-helper mirror of the predicate (extracted from the Solve fragment)
# ---------------------------------------------------------------------------

from typing import Dict


def constraints_key_present(constraints_key: str, config: Dict[str, int]) -> bool:
    """Mirror of Configuration.py:929 predicate `self.ConstraintsKey in self`.

    self.ConstraintsKey is the constant "_Constraints"; `in self` dispatches to
    ProjectConfig.__contains__, a dict membership test.  Returns True iff the
    constraints section key is present in the config dict.
    """
    return constraints_key in config


# ---------------------------------------------------------------------------
# Pure-helper tests (witness enumeration)
# ---------------------------------------------------------------------------


def test_constraints_key_present_true_when_key_exists():
    """TRUE witness: key present -> predicate is True."""
    assert constraints_key_present("_Constraints", {"_Constraints": 1}) is True


def test_constraints_key_present_false_when_key_absent():
    """FALSE witness: key absent -> predicate is False."""
    assert constraints_key_present("_Constraints", {}) is False


def test_constraints_key_present_false_when_other_keys_present():
    """FALSE witness: other keys present but not the constraints key -> False."""
    assert constraints_key_present("_Constraints", {"SomeOtherKey": 42}) is False


# ---------------------------------------------------------------------------
# Real-entry tests against ProjectConfig
# ---------------------------------------------------------------------------


def test_constraints_key_absent_in_fresh_config():
    """Fresh ProjectConfig has no constraints -> ConstraintsKey NOT in self (FALSE branch)."""
    from Tensile.Configuration import ProjectConfig

    cfg = ProjectConfig()
    assert (cfg.ConstraintsKey in cfg) is False


def test_constraints_key_present_after_add_constraint():
    """After addConstraint(), ConstraintsKey IS in self (TRUE branch)."""
    from Tensile.Configuration import ProjectConfig

    cfg = ProjectConfig()
    cfg.createValue("X", 1)
    cfg.addConstraint("X >= 0")
    assert (cfg.ConstraintsKey in cfg) is True


def test_constraints_key_value_is_underscore_constraints():
    """Pin the constant: ConstraintsKey == "_Constraints" (Configuration.py:818)."""
    from Tensile.Configuration import ProjectConfig

    cfg = ProjectConfig()
    assert cfg.ConstraintsKey == "_Constraints"


def test_check_constraints_returns_true_when_constraint_satisfied():
    """TRUE branch downstream: checkConstraints() returns True when constraint holds."""
    from Tensile.Configuration import ProjectConfig

    cfg = ProjectConfig()
    cfg.createValue("X", 1)
    cfg.addConstraint("X >= 0")
    # "_Constraints" is now present -> TRUE branch executes -> returns result
    result = cfg.checkConstraints()
    assert result is True


def test_check_constraints_raises_unbound_local_error_when_no_constraints():
    """FALSE branch latent bug (pinned ACTUAL): when no constraints exist,
    checkConstraints() skips the loop and hits `return value` (line 935)
    with `value` unbound -> raises UnboundLocalError."""
    from Tensile.Configuration import ProjectConfig

    cfg = ProjectConfig()
    # "_Constraints" NOT present -> FALSE branch -> UnboundLocalError
    with pytest.raises(UnboundLocalError):
        cfg.checkConstraints()
