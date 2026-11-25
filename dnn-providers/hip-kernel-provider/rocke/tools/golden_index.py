# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The golden index: every rocKE golden, the tests that check it, and how to
re-record it.

A *golden* here is the unit one re-record writes: a shard directory, a file
shared by several families, or a single-family file. Its name is what a failing
golden test prints in its re-record command and what ``tools/golden.py`` takes::

    python tools/golden.py rerecord kda_gfx942
    python tools/golden.py rerecord representative --only attention_dense,gemm

Every file under a ``golden/`` or ``_goldens/`` directory of the searched trees
must belong to exactly one golden here, and each golden needs a test that checks
it; ``python tools/golden.py check`` enforces both, and the fingerprint of every
one-family JSON document. Paths are relative to the rocke directory, with ``/``.

Adding a golden: add a :class:`Golden` with the files it owns, the tests that
check it and the command that writes it, and have the test's failure message
print ``rocke.core.golden_file.rerecord_hint(<name>)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

__all__ = [
    "DOCUMENT",
    "GOLDENS",
    "GOLDEN_DIR_NAMES",
    "NOT_GOLDEN_SUFFIXES",
    "PER_FAMILY",
    "SEARCH_ROOTS",
    "UNFINGERPRINTED",
    "Golden",
    "RecordStep",
    "golden_named",
]

# How a golden records its fingerprint (rocke.core.golden_file):
#: Each file is one family's JSON document with a top-level fingerprint;
#: ``tools/golden.py check`` verifies it.
DOCUMENT = "document"
#: A file shared by several families records one fingerprint per family; the
#: golden's own check verifies them (its format owner computes the entries).
PER_FAMILY = "per_family"
#: Each file is one value compared whole (an IR text, a GPU-run digest), so the
#: file carries no fingerprint.
UNFINGERPRINTED = "unfingerprinted"
_FINGERPRINT_KINDS = (DOCUMENT, PER_FAMILY, UNFINGERPRINTED)

#: Directories whose files are goldens, at any depth of a searched tree.
GOLDEN_DIR_NAMES = ("golden", "_goldens")
#: The trees searched for goldens.
SEARCH_ROOTS = ("platform/tests", "platform/python/rocke", "library/tests")
#: Files in a golden directory that are not goldens (code and notes).
NOT_GOLDEN_SUFFIXES = (".py", ".md")

# The placeholder a RecordStep's ``only`` arguments carry for the family list.
_FAMILIES = "{families}"


def _check_relative(owner: str, field_name: str, path: str) -> None:
    """Refuse a path that is absolute, escapes upward or uses ``\\``."""
    parts = PurePosixPath(path).parts
    if not path or path.startswith("/") or "\\" in path or ".." in parts:
        raise ValueError(
            f"{owner}: {field_name}={path!r}, expected a '/'-separated path "
            "relative to the rocke directory"
        )


@dataclass(frozen=True)
class RecordStep:
    """One command that writes (part of) a golden: ``python <argv...>`` run in
    ``cwd`` with ``pythonpath`` (relative to ``cwd``) ahead of PYTHONPATH.

    ``only`` holds the arguments that narrow the command to some families, with
    ``{families}`` standing for their comma-separated names; empty when the
    command always writes the whole golden.
    """

    cwd: str
    argv: tuple[str, ...]
    pythonpath: tuple[str, ...] = ()
    only: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_relative("RecordStep", "cwd", self.cwd)
        if not self.argv:
            raise ValueError(f"RecordStep in {self.cwd}: argv is empty")
        if self.only and not any(_FAMILIES in arg for arg in self.only):
            raise ValueError(
                f"RecordStep in {self.cwd}: only={self.only!r}, expected an "
                f"argument containing {_FAMILIES!r}"
            )

    def arguments(self, families: list[str] | None = None) -> list[str]:
        """The arguments after the interpreter; narrowed to ``families`` when
        given (the step must take a family list)."""
        if not families:
            return list(self.argv)
        if not self.only:
            raise ValueError(
                f"RecordStep in {self.cwd} writes its golden whole - "
                f"families={families!r}, expected none"
            )
        names = ",".join(families)
        return [*self.argv, *(arg.replace(_FAMILIES, names) for arg in self.only)]


@dataclass(frozen=True)
class Golden:
    """One golden: the files it owns (glob patterns, ``*`` within one
    directory), the tests that check it (at least one must exist), and the
    steps that re-record it, in order."""

    name: str
    files: tuple[str, ...]
    checked_by: tuple[str, ...]
    record: tuple[RecordStep, ...]
    fingerprint: str
    #: Re-recording runs on a GPU, so ``rerecord --all`` leaves it out.
    needs_gpu: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"Golden name={self.name!r}, expected an identifier")
        for field_name in ("files", "checked_by", "record"):
            if not getattr(self, field_name):
                raise ValueError(f"Golden {self.name}: {field_name} is empty")
        for path in (*self.files, *self.checked_by):
            _check_relative(f"Golden {self.name}", "path", path)
        if self.fingerprint not in _FINGERPRINT_KINDS:
            raise ValueError(
                f"Golden {self.name}: fingerprint={self.fingerprint!r}, expected "
                f"one of {list(_FINGERPRINT_KINDS)}"
            )
        if len({bool(step.only) for step in self.record}) != 1:
            raise ValueError(
                f"Golden {self.name}: either every record step takes a family "
                "list or none does"
            )

    @property
    def takes_families(self) -> bool:
        """Whether a re-record can be narrowed to some families (``--only``)."""
        return bool(self.record[0].only)

    def owns(self, path: str) -> bool:
        """Whether the rocke-relative ``path`` is one of this golden's files."""
        candidate = PurePosixPath(path)
        return any(
            len(candidate.parts) == len(PurePosixPath(pattern).parts)
            and candidate.match(pattern)
            for pattern in self.files
        )


def _library_test_golden(
    name: str,
    golden_file: str,
    test: str,
    argv: tuple[str, ...] | None = None,
) -> Golden:
    """A library single-family JSON golden, written by its test's ``--write``
    (``argv``, run from rocke/library, when the test needs another form)."""
    return Golden(
        name=name,
        files=(f"library/tests/golden/{golden_file}",),
        checked_by=(f"library/tests/{test}",),
        record=(
            RecordStep(
                cwd="library",
                argv=argv or (f"tests/{test}", "--write"),
                pythonpath=("../platform/python", "."),
            ),
        ),
        fingerprint=DOCUMENT,
    )


_DIFFERENTIAL_DIR = "platform/tests/instances/differential"
_DIFFERENTIAL_PYTHONPATH = ("../../../python", "../../../../library")
_ONLY = ("--only", _FAMILIES)

GOLDENS: tuple[Golden, ...] = (
    Golden(
        name="representative",
        files=("platform/tests/golden/rocke_representative_ir_sha256/*.json",),
        checked_by=(
            "platform/tests/rocke_installed_golden_test.py",
            "platform/tests/test_rocke_ci_static.py",
        ),
        record=(
            RecordStep(
                cwd="platform",
                argv=(
                    "tests/instances/rocke_ir_parity_harness.py",
                    "--write",
                    "tests/golden/rocke_representative_ir_sha256",
                ),
                pythonpath=("python", "../library"),
                only=_ONLY,
            ),
        ),
        fingerprint=DOCUMENT,
    ),
    Golden(
        name="differential",
        files=(
            f"{_DIFFERENTIAL_DIR}/golden/llvm_gfx_all.json",
            "library/tests/parity/golden/llvm_gfx_all.json",
        ),
        checked_by=(f"{_DIFFERENTIAL_DIR}/run_diff.py",),
        record=(
            RecordStep(
                cwd=_DIFFERENTIAL_DIR,
                argv=(
                    "run_diff.py",
                    "--mode",
                    "ll",
                    "--python-only",
                    "--record-golden",
                    "--flavor",
                    "all",
                ),
                pythonpath=_DIFFERENTIAL_PYTHONPATH,
                only=_ONLY,
            ),
            RecordStep(
                cwd=_DIFFERENTIAL_DIR,
                argv=(
                    "run_diff.py",
                    "--mode",
                    "ir",
                    "--canonical",
                    "--python-only",
                    "--record-golden",
                ),
                pythonpath=_DIFFERENTIAL_PYTHONPATH,
                only=_ONLY,
            ),
        ),
        fingerprint=PER_FAMILY,
    ),
    Golden(
        name="conv",
        files=("library/tests/golden/conv_representative_ir/*.json",),
        checked_by=("library/tests/test_conv_ir_golden.py",),
        record=(
            RecordStep(
                cwd="library",
                argv=("tests/test_conv_ir_golden.py", "--write"),
                pythonpath=("../platform/python", "."),
                only=_ONLY,
            ),
        ),
        fingerprint=DOCUMENT,
    ),
    _library_test_golden(
        "attention_bwd",
        "attention_bwd_ir_sha256.json",
        "test_attention_bwd_ir_golden.py",
    ),
    # The test imports a sibling module relatively, so it runs as a module.
    _library_test_golden(
        "attention_dense_bwd",
        "attention_dense_bwd_ir_sha256.json",
        "test_attention_dense_bwd_ir.py",
        argv=("-m", "tests.test_attention_dense_bwd_ir", "--write"),
    ),
    _library_test_golden(
        "attention_dense_gfx942",
        "attention_dense_gfx942_ir_sha256.json",
        "test_attention_dense_gfx942_golden.py",
    ),
    _library_test_golden(
        "attention_strided_kv",
        "attention_strided_kv_ir_sha256.json",
        "test_strided_kv_decode.py",
        argv=("-m", "tests.test_strided_kv_decode", "--write"),
    ),
    _library_test_golden(
        "dsa_indexer", "dsa_indexer_ir_sha256.json", "test_dsa_indexer_golden.py"
    ),
    _library_test_golden(
        "gdn_decode", "gdn_decode_gfx950_ir_sha256.json", "test_gdn_decode_golden.py"
    ),
    _library_test_golden(
        "kda_gfx942", "kda_gfx942_ir_sha256.json", "test_kda_gfx942_golden.py"
    ),
    _library_test_golden(
        "kda_gfx950", "kda_gfx950_ir_sha256.json", "test_kda_gfx950_golden.py"
    ),
    _library_test_golden(
        "mla_prefill_gfx942",
        "mla_prefill_gfx942_ir_sha256.json",
        "test_mla_prefill_gfx942_golden.py",
    ),
    # One generator records the shared and concrete native host contracts.
    Golden(
        name="attention_dense_spec_vectors",
        files=(
            "platform/tests/golden/attention_dense_spec_vectors.json",
            "platform/tests/golden/attention_dense_gfx950_spec_vectors.json",
        ),
        checked_by=(
            "library/tests/test_attention_dense_spec_vectors.py",
            "platform/tests/instances/test_attention_dense_spec.cpp",
            "platform/tests/instances/test_attention_dense_gfx950_spec.cpp",
        ),
        record=(
            RecordStep(
                cwd="library",
                argv=("tests/test_attention_dense_spec_vectors.py", "--regenerate"),
                pythonpath=("../platform/python", "."),
            ),
        ),
        fingerprint=DOCUMENT,
    ),
    Golden(
        name="tensor_transforms_emit",
        files=(
            (
                "platform/python/rocke/tests/helpers/tiling/tensor_transforms/golden/"
                "emit_*.txt"
            ),
        ),
        checked_by=(
            "platform/python/rocke/tests/helpers/tiling/tensor_transforms/test_emit.py",
        ),
        record=(
            RecordStep(
                cwd="platform/python",
                argv=(
                    "-m",
                    "rocke.tests.helpers.tiling.tensor_transforms.test_emit",
                    "--write",
                ),
                pythonpath=(".",),
            ),
        ),
        fingerprint=UNFINGERPRINTED,
    ),
    Golden(
        name="examples",
        files=("platform/python/rocke/examples/_goldens/*.json",),
        checked_by=("platform/python/rocke/examples/run_all.py",),
        record=(
            RecordStep(
                cwd="platform/python",
                argv=("-m", "rocke.examples.run_all", "--bless"),
                pythonpath=(".",),
                only=_ONLY,
            ),
        ),
        fingerprint=UNFINGERPRINTED,
        needs_gpu=True,
    ),
)


def _check_unique_names(goldens: tuple[Golden, ...]) -> None:
    names = [golden.name for golden in goldens]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"golden index names {repeated} more than once")


_check_unique_names(GOLDENS)


def golden_named(name: str) -> Golden:
    """The golden called ``name``; refuses an unknown name, listing the known."""
    for golden in GOLDENS:
        if golden.name == name:
            return golden
    raise KeyError(
        f"unknown golden - name={name!r}, expected one of "
        f"{sorted(golden.name for golden in GOLDENS)}"
    )
