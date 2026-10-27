# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The rocKE package a translation belongs to: its C naming and its Python modules.

Every name the translator invents or looks up for a package (rocke_attn_bwd_*,
ROCKE_ATTN_BWD_*, the int | Value union rocke_attn_bwd_iv_t, ...) comes from the
package selected here, so another package is one more PACKAGES entry, not an
edit of the translator. Select one with $TR_PACKAGE (read once, at import;
tr.regen --package sets it). The default is attention backward.

Per-function adapters stay data keyed by module name (tr/adapters.py, the
prelude headers); they are not part of this table.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Package:
    name: str
    prefix: str  # C prefix: rocke_<prefix>_*, ROCKE_<PREFIX>_*
    class_prefix: str  # dropped from Python class names in struct names (AttnBwdFoo -> rocke_<prefix>_foo_t)
    modules: tuple[
        str, ...
    ]  # Python module name prefixes of the package (siblings share the struct rule)
    included: tuple[
        str, ...
    ]  # modules whose generated header is included, not forward-declared

    @property
    def c(self) -> str:
        """'rocke_attn_bwd_': prefix of the package's C functions and types."""
        return f"rocke_{self.prefix}_"

    @property
    def macro(self) -> str:
        """'ROCKE_ATTN_BWD_': prefix of the package's constants and enumerators."""
        return f"ROCKE_{self.prefix.upper()}_"

    @property
    def iv_t(self) -> str:
        """The C union of a Python `int | Value`."""
        return f"{self.c}iv_t"

    @property
    def err_t(self) -> str:
        """The error record a C helper reports a Python raise through."""
        return f"{self.c}err_t"


PACKAGES = {
    "attention_bwd": Package(
        name="attention_bwd",
        prefix="attn_bwd",
        class_prefix="AttnBwd",
        modules=("_attention_bwd_", "attention_bwd_"),
        included=("_attention_bwd_addr", "_attention_bwd_caps"),
    ),
}

DEFAULT = "attention_bwd"


def select(name: str | None) -> Package:
    name = name or DEFAULT
    if name not in PACKAGES:
        sys.exit(
            f"error: unknown package {name!r} (TR_PACKAGE); known: {', '.join(sorted(PACKAGES))}"
        )
    return PACKAGES[name]


PKG = select(os.environ.get("TR_PACKAGE"))
