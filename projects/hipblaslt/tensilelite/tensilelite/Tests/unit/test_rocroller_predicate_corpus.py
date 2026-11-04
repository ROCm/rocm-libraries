# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every shipped rocRoller custom kernel must carry the predicates that bound it.

A rocRoller custom kernel is a pre-assembled object captured at one tile. It is
correct only for the problems it was captured for, and the conditions it needs
are not visible in its name: they are carried as *selection predicates* attached
to the solution, which hipBLASLt evaluates when it picks a kernel. Those
predicates are not written in the logic files. They are derived at library-build
time by ``ProblemPredicate.CompoundPredicates`` from each solution's state, and a
kernel whose derivation does not fire ships with nothing bounding it.

The existing tests build a solution state by hand and assert what the derivation
emits for it. That proves the derivation is right when it runs; it proves
nothing about whether it ran for a kernel actually in the tree. A kernel whose
``macrotile`` went missing, or whose name stopped matching the ``RR_GEMM_``
prefix the derivation keys on, or which was dropped entirely by the
skip-on-missing-config path in ``LibraryIO`` (that path warns and returns
``None``), would ship unbounded with every one of those tests still green.

These tests close that gap from the other end. They parse the shipped logic
exactly the way a library build does, convert each solution with the same call
the build uses, and check the predicates that come out.

Two checks, deliberately different in kind:

* ``test_every_rocroller_kernel_carries_its_predicates`` states the rule
  directly, so it holds for kernels added later without anyone updating a file.
* ``test_predicates_match_the_checked_in_manifest`` compares against
  ``data/rocroller_predicate_manifest.yaml``, which records what each named
  kernel resolves to today. That makes a change to the derivation show up as a
  reviewable per-kernel diff rather than as a number moving in a test log.

Regenerating the manifest is deliberate, not a flag on a test run: set
``HIPBLASLT_UPDATE_RR_PREDICATE_MANIFEST=1``. Whatever it writes lands in the
diff and has to be reviewed like any other change.
"""

import os
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit


# The predicates this file is about. The derivation emits around forty-five
# distinct predicate kinds across the corpus, nearly all of them unrelated to
# rocRoller; pinning all of them would make an unrelated change to any other
# branch of the derivation churn the manifest and teach reviewers to ignore it.
RR_PREDICATE_TAGS = (
    "BatchSizeEqual",
    "BoundSizeMultiple",
    "BufferAddressingLimitCheck",
    "Free0SizeMultiple",
    "MXScaleHostPreSwizzle",
)

RR_KERNEL_PREFIX = "RR_GEMM_"

_HERE = Path(__file__).resolve().parent
_MANIFEST = _HERE / "data" / "rocroller_predicate_manifest.yaml"

# The rocRoller kernels are registered through this chip-id overlay only; the
# base gfx950 directory would fail the ValidChipId lint.
_CORPUS = (
    _HERE.parents[3]
    / "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full"
    / "gfx950/gfx950_id75a3/GridBased"
)


def _flatten(predicate, out):
    """Collect ``(tag, index, value)`` for every leaf of a predicate tree.

    A solution's predicate is one ``And`` over a flat list, but ``Or`` and
    nesting are both expressible, so walk rather than assume a shape.
    """
    if getattr(predicate, "tag", None) in ("And", "Or") and isinstance(
        predicate.value, (list, tuple)
    ):
        for child in predicate.value:
            _flatten(child, out)
    else:
        out.append(
            (
                predicate.tag,
                getattr(predicate, "index", None),
                getattr(predicate, "value", None),
            )
        )


@pytest.fixture(scope="module")
def toolchain():
    """The assembler and per-ISA capability map a library build parses with.

    Skips rather than fails where the ROCm toolchain is absent: this suite also
    runs on hosts that have the sources but not a compiler.
    """
    try:
        from tensilelite.Common.Architectures import SUPPORTED_ISA
        from tensilelite.Common.Capabilities import makeIsaInfoMap
        from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
        from tensilelite.Toolchain.Validators import validateToolchain, ToolchainDefaults

        cxx = validateToolchain("amdclang++")
        isa_info_map = makeIsaInfoMap(SUPPORTED_ISA, cxx)
        bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
        assembler = makeAssemblyToolchain(cxx, bundler, "default").assembler
    except Exception as exc:  # noqa: BLE001 - any toolchain gap is a skip, not a failure
        pytest.skip(f"ROCm toolchain not usable here: {exc}")
    return assembler, isa_info_map


@pytest.fixture(scope="module")
def corpus_predicates(toolchain):
    """Map every shipped ``RR_GEMM_`` kernel to its rocRoller predicates.

    Parsing the whole overlay takes on the order of a minute, so this is built
    once for the module. ``FromOriginalState`` is called with the same arguments
    ``MasterSolutionLibrary`` passes during a build, so what comes back is what
    the built library carries rather than an approximation of it.
    """
    if not _CORPUS.is_dir():
        pytest.skip(f"rocRoller logic overlay not present: {_CORPUS}")

    import tensilelite.Contractions as Contractions
    import tensilelite.LibraryIO as LibraryIO

    assembler, isa_info_map = toolchain
    predicates = {}
    macrotiles = {}

    for logic_file in sorted(_CORPUS.glob("*.yaml")):
        logic = LibraryIO.parseLibraryLogicFile(
            str(logic_file), assembler, False, False, False, isa_info_map, False
        )
        for parsed in logic.solutions:
            custom_kernel = parsed._state.get("CustomKernel") or {}
            name = custom_kernel.get("name")
            if not str(name).startswith(RR_KERNEL_PREFIX):
                continue
            solution = Contractions.Solution.FromOriginalState(
                parsed._state, False, False, False, assembler, isa_info_map
            )
            leaves = []
            _flatten(solution.problemPredicate, leaves)
            predicates[name] = sorted(
                leaf for leaf in leaves if leaf[0] in RR_PREDICATE_TAGS
            )
            macrotiles[name] = custom_kernel.get("macrotile")

    return predicates, macrotiles


def test_the_overlay_actually_holds_kernels(corpus_predicates):
    """Guard against passing for the wrong reason.

    Every other assertion here iterates the kernels found. If the overlay moved,
    or the prefix the search keys on changed, that iteration would be empty and
    the rest of this file would pass while checking nothing.
    """
    predicates, _ = corpus_predicates
    assert len(predicates) > 100, (
        f"only {len(predicates)} rocRoller kernels found under {_CORPUS}; "
        "the overlay or the kernel naming has moved and these checks are no "
        "longer looking at the shipped kernels"
    )


def test_every_rocroller_kernel_carries_its_predicates(corpus_predicates):
    """The rule itself, so a kernel added later is covered without an edit.

    Each kernel must declare the scale layout it reads, the M and K multiples
    its capture requires, that it handles one batch only, and that it cannot
    address a tensor past the limit of its buffer descriptors.
    """
    predicates, macrotiles = corpus_predicates
    problems = []

    for name in sorted(predicates):
        found = {tag: (index, value) for tag, index, value in predicates[name]}
        missing = [tag for tag in RR_PREDICATE_TAGS if tag not in found]
        if missing:
            problems.append(f"{name}: missing {', '.join(missing)}")
            continue

        macrotile = macrotiles[name]
        if not (isinstance(macrotile, list) and len(macrotile) == 3):
            problems.append(f"{name}: macrotile is {macrotile!r}, expected 3 values")
            continue
        tile_m, _, depth_u = macrotile

        # The kernel has no edge handling in M and no partial-k-tile drain, so
        # these two must track its own tile rather than any fixed number.
        if found["Free0SizeMultiple"] != (0, tile_m):
            problems.append(
                f"{name}: Free0SizeMultiple is {found['Free0SizeMultiple']}, "
                f"expected (0, {tile_m}) from its macrotile"
            )
        if found["BoundSizeMultiple"] != (-1, depth_u):
            problems.append(
                f"{name}: BoundSizeMultiple is {found['BoundSizeMultiple']}, "
                f"expected (-1, {depth_u}) from its macrotile"
            )
        # Two-dimensional kernels: no batch strides, and the batch component of
        # the launch grid would otherwise recompute batch zero into every slice.
        if found["BatchSizeEqual"] != (0, 1):
            problems.append(
                f"{name}: BatchSizeEqual is {found['BatchSizeEqual']}, expected (0, 1)"
            )
        # These kernels read a compact row-major E8M0 scale array, never the
        # host-pre-swizzled one, which has identical extents and element type.
        if found["MXScaleHostPreSwizzle"][1] is not False:
            problems.append(
                f"{name}: MXScaleHostPreSwizzle is "
                f"{found['MXScaleHostPreSwizzle'][1]!r}, expected False"
            )

    assert not problems, "\n".join(problems)


def test_predicates_match_the_checked_in_manifest(corpus_predicates):
    """Pin the per-kernel result so a derivation change is reviewable.

    The previous test says the shape is right. This one says the values have not
    moved since someone last looked at them, which is what catches a change that
    is self-consistent but not what was intended.
    """
    predicates, _ = corpus_predicates
    actual = {
        name: [list(leaf) for leaf in leaves] for name, leaves in predicates.items()
    }

    if os.environ.get("HIPBLASLT_UPDATE_RR_PREDICATE_MANIFEST") == "1":
        _MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        _MANIFEST.write_text(
            "# Generated. Regenerate with "
            "HIPBLASLT_UPDATE_RR_PREDICATE_MANIFEST=1, and review the diff:\n"
            "# a change here is a change to which problems these kernels accept.\n"
            # default_flow_style=None keeps each predicate on one line while
            # leaving the per-kernel lists block-style, so a changed predicate
            # shows up as a one-line diff.
            + yaml.safe_dump(actual, default_flow_style=None, sort_keys=True)
        )
        pytest.skip("manifest regenerated; re-run without the environment variable")

    assert _MANIFEST.is_file(), (
        f"{_MANIFEST} is missing; regenerate it with "
        "HIPBLASLT_UPDATE_RR_PREDICATE_MANIFEST=1"
    )
    expected = yaml.safe_load(_MANIFEST.read_text())

    added = sorted(set(actual) - set(expected))
    removed = sorted(set(expected) - set(actual))
    assert not added, (
        "kernels not in the manifest (add them deliberately): " + ", ".join(added)
    )
    assert not removed, (
        "kernels in the manifest that no longer resolve; they may have been "
        "dropped from the build: " + ", ".join(removed)
    )

    changed = [
        f"{name}: manifest {expected[name]} -> derived {actual[name]}"
        for name in sorted(actual)
        if expected[name] != actual[name]
    ]
    assert not changed, "\n".join(changed)
