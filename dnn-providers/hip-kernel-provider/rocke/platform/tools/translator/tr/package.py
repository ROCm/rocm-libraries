# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The rocKE package a translation belongs to: its C naming, its Python modules
and where its files live.

Every name the translator invents or looks up for a package (rocke_attn_bwd_*,
ROCKE_ATTN_BWD_*, the int | Value union rocke_attn_bwd_iv_t, ...) comes from the
package selected here, and so does every path: the directory the Python modules
are read and imported from, the C++ directory a source is written to, the
generated header's name, the parity drivers' directory, and the files the code
maps are mined from. Another package is one more PACKAGES entry, not an edit of
the translator. Select one with $TR_PACKAGE (read once, at import;
tr.regen --package sets it). The default is attention backward.

Per-function adapters stay data keyed by module name (tr/adapters.py); they
are not part of this table. Support headers the translated sources include by
a quoted same-directory name come from tr/prelude/ (rocke_py.hpp and
rocke_py_adapters.hpp, generic) or tr/packages/<package>/ (package data, such
as rocke_py_adapters_attention_bwd.hpp, the adapters that need the attention
backward headers); support_file() finds them.
"""

from __future__ import annotations

import os
import re
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
    # Where the files live (paths relative to the rocKE root, "/" separated).
    py_root: str = "library"  # import root of the modules ($TR_LIBRARY replaces it)
    py_package: str = "kernels.common"  # dotted package the modules import as
    cpp_dir: str = "platform/cpp/instances/common"  # where a source is written
    header: str = "helper_rocke.instances.common.{m}.h"  # generated header name
    parity_dir: str = "library/tests/parity"  # the <module>_emit drivers
    # Files the code maps (tools/mine_maps.py) are mined from: the package's
    # Python modules (in the module directory), its C++ headers (include/rocke)
    # and sources (cpp_dir), and its parity drivers (parity_dir).
    py_globs: tuple[str, ...] = ("*attention*bwd*.py",)
    h_globs: tuple[str, ...] = (
        "helper_rocke.instances.common.*attention*bwd*.h",
        "instance_*attention*bwd*.h",
    )
    cpp_globs: tuple[str, ...] = ("*attention*bwd*.cpp",)
    parity_globs: tuple[str, ...] = ("*attention*bwd*_emit.c",)

    @property
    def c(self) -> str:
        """'rocke_attn_bwd_': prefix of the package's C functions and types."""
        return f"rocke_{self.prefix}_"

    @property
    def macro(self) -> str:
        """'ROCKE_ATTN_BWD_': prefix of the package's constants and enumerators."""
        return f"ROCKE_{self.prefix.upper()}_"

    @property
    def data_dir(self) -> str:
        """tr/packages/<name>/: the package's support headers."""
        return os.path.join(HERE, "packages", self.name)

    @property
    def py_dir(self) -> str:
        """The modules' directory relative to the import root ('kernels/common')."""
        return self.py_package.replace(".", "/")

    def module_name(self, m: str) -> str:
        """The import name of module m ('kernels.common._attention_bwd_addr');
        a dotted name is already one."""
        return m if "." in m else f"{self.py_package}.{m}"

    def import_source(self, module: str | None, level: int) -> str:
        """The module an import statement of a package module names, as the
        translator keys it: absolute and sibling imports as written
        ('rocke.helpers.spec', '_attention_bwd_valu' from `from ._attention_bwd_valu
        import`), an import from a parent package made absolute
        (`from ...helpers.spec import` in rocke.instances.common.<m> ->
        'rocke.helpers.spec')."""
        module = module or ""
        if level <= 1:
            return module
        parts = self.py_package.split(".")
        base = parts[: max(0, len(parts) - (level - 1))]
        return ".".join(base + ([module] if module else []))

    def header_name(self, m: str) -> str:
        """The generated header of module m."""
        return self.header.format(m=m)

    def header_names(self, m: str) -> tuple[str, ...]:
        """Every name a header of module m can have in the tree: the package's
        own, then the two rocKE conventions (a hand header may use either)."""
        out = [self.header_name(m)]
        for c in (f"helper_rocke.instances.common.{m}.h", f"instance_{m}.h"):
            if c not in out:
                out.append(c)
        return tuple(out)

    def header_module(self, filename: str) -> str | None:
        """The module a header file name belongs to under the names above, or None."""
        for pat in dict.fromkeys(
            (self.header, "helper_rocke.instances.common.{m}.h", "instance_{m}.h")
        ):
            pre, post = pat.split("{m}")
            if filename.startswith(pre) and filename.endswith(post):
                m = filename[len(pre) : len(filename) - len(post)]
                if m and "." not in m:
                    return m
        return None

    def adapters_header(self) -> str | None:
        """The package's own adapters header, rocke_py_adapters_<name>.hpp in
        its data directory (adapters that need the package's headers), or None."""
        name = f"rocke_py_adapters_{self.name}.hpp"
        return name if os.path.isfile(os.path.join(self.data_dir, name)) else None

    def adapter_names(self) -> tuple[str, ...]:
        """The py_adapt functions the package's own adapters header defines."""
        own = self.adapters_header()
        if own is None:
            return ()
        text = open(os.path.join(self.data_dir, own)).read()
        return tuple(sorted(set(re.findall(r"^[^\s/*#][^(\n]*?\b(\w+)\(", text, re.M))))

    @property
    def iv_t(self) -> str:
        """The C union of a Python `int | Value`."""
        return f"{self.c}iv_t"

    @property
    def err_t(self) -> str:
        """The error record a C helper reports a Python raise through."""
        return f"{self.c}err_t"


HERE = os.path.dirname(os.path.abspath(__file__))


def platform_instance(
    module: str, prefix: str, class_prefix: str = "", arch: str = "common"
) -> Package:
    """A kernel instance of the platform (platform/python/rocke/instances/<arch>/
    <module>.py, imported as rocke.instances.<arch>.<module>): its C++ twin goes
    to platform/cpp/instances/<arch>/, its header is instance_<module>.h (the
    platform convention), and its parity driver is platform/tests/instances/
    parity/<module>_emit.py. One module per package: each instance has its own
    C prefix (rocke_<prefix>_*)."""
    return Package(
        name=module,
        prefix=prefix,
        class_prefix=class_prefix,
        modules=(module,),
        included=(),
        py_root="platform/python",
        py_package=f"rocke.instances.{arch}",
        cpp_dir=f"platform/cpp/instances/{arch}",
        header="instance_{m}.h",
        parity_dir="platform/tests/instances/parity",
        py_globs=(f"{module}.py",),
        h_globs=(f"instance_{module}.h", f"instance_{module}_*.h"),
        cpp_globs=(f"{module}.cpp", f"{module}_*.cpp"),
        parity_globs=(f"{module}_emit.c",),
    )


PACKAGES = {
    "attention_bwd": Package(
        name="attention_bwd",
        prefix="attn_bwd",
        class_prefix="AttnBwd",
        modules=("_attention_bwd_", "attention_bwd_"),
        included=("_attention_bwd_addr", "_attention_bwd_caps"),
    ),
}
for _p in (
    # mirrored by hand already: the platform port's end-to-end check
    platform_instance("transpose", "transpose2d", "Transpose2D"),
    platform_instance("batched_transpose", "batched_transpose", "BatchedTranspose2D"),
    platform_instance("transpose_bc", "transpose_bc", "TransposeBc"),
    platform_instance(
        "batched_contraction", "batched_contraction", "BatchedContraction"
    ),
    platform_instance("qwen3_kv_cache", "qwen3_kv_cache", "Qwen3Kv", "gfx1250"),
    platform_instance(
        "qwen3_qk_norm_rope", "qwen3_qk_norm_rope", "Qwen3QkNormRope", "gfx1250"
    ),
    platform_instance(
        "qwen3_sampler", "qwen3_sampler", "Qwen3GreedySampler", "gfx1250"
    ),
    platform_instance(
        "qwen3_token_embedding",
        "qwen3_token_embedding",
        "Qwen3TokenEmbedding",
        "gfx1250",
    ),
):
    PACKAGES[_p.name] = _p

DEFAULT = "attention_bwd"


def select(name: str | None) -> Package:
    name = name or DEFAULT
    if name not in PACKAGES:
        sys.exit(
            f"error: unknown package {name!r} (TR_PACKAGE); known: {', '.join(sorted(PACKAGES))}"
        )
    return PACKAGES[name]


PKG = select(os.environ.get("TR_PACKAGE"))


def support_file(name: str, pkg: Package | None = None) -> str | None:
    """The master copy of support header `name` (tr/prelude/, then the
    package's data directory), or None if neither has it."""
    for d in (os.path.join(HERE, "prelude"), (pkg or PKG).data_dir):
        path = os.path.join(d, name)
        if os.path.isfile(path):
            return path
    return None
