# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The default heuristic mode of the Python graph bindings is ``[HeuristicMode.A]``.

Every bound method that takes a heuristic-mode list and has a default for it
declares ``[HeuristicMode.A]``. The default lives in the nanobind signature, so
these tests read it from ``__doc__`` and need no device, handle or engine. Each
case asserts on the default of the ``modes`` parameter itself, not merely that
``HeuristicMode.A`` appears somewhere in the doc, so a regression to
``HeuristicMode.FALLBACK`` names the method that changed.
"""

import re

import pytest

import hipdnn_frontend as hipdnn

# Matches the ``modes`` parameter of a rendered signature and captures its default.
# The type annotation holds no ``=`` or ``,``; ``(?<!\w)`` keeps ``heur_modes``
# (an overload with no default) from matching.
_MODES_DEFAULT = re.compile(r"(?<!\w)modes: [^=,]+ = (\[[^\]]*\])")

# (class, method) for every site whose ``modes`` argument defaults to a mode list.
_DEFAULTED_MODES_METHODS = [
    (hipdnn.Graph, "create_execution_plans"),
    (hipdnn.Graph, "get_ranked_engine_ids"),
    (hipdnn.Graph, "get_engine_configs"),
    (hipdnn.Graph, "add_all_engines"),
    (hipdnn.pygraph, "create_execution_plans"),
    (hipdnn.pygraph, "build"),
    (hipdnn.pygraph, "add_all_engines"),
]


def _modes_defaults(cls, method):
    """Return the ``modes`` default of every rendered overload of cls.method.

    nanobind renders each overload's signature on a line that starts with the
    method name; a single-overload method has exactly one such line. Overloads
    without a defaulted ``modes`` parameter contribute nothing.
    """
    doc = getattr(cls, method).__doc__
    signatures = [line for line in doc.splitlines() if line.startswith(f"{method}(")]
    assert signatures, f"{cls.__name__}.{method}: no signature in __doc__:\n{doc}"
    return [
        match.group(1)
        for signature in signatures
        for match in [_MODES_DEFAULT.search(signature)]
        if match
    ]


@pytest.mark.parametrize(
    ("cls", "method"),
    _DEFAULTED_MODES_METHODS,
    ids=[f"{cls.__name__}.{method}" for cls, method in _DEFAULTED_MODES_METHODS],
)
def test_modes_defaults_to_heuristic_mode_a(cls, method):
    """The ``modes`` argument defaults to ``[HeuristicMode.A]``, never FALLBACK."""
    defaults = _modes_defaults(cls, method)
    assert defaults == ["[HeuristicMode.A]"], (
        f"{cls.__name__}.{method} must default modes to [HeuristicMode.A]; "
        f"signature defaults: {defaults}"
    )
