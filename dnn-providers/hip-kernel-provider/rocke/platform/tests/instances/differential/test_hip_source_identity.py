# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Both engines lower every parity kernel to the same HIP source.

``tools/check_byte_identity.py`` holds the LLVM-IR route byte-identical between
the Python and C++ engines. The HIP-source route has a lowerer in each engine as
well -- ``rocke.core.lower_hip.lower_kernel_to_hip`` and
``rocke_engine.lower_serialized_ir_hip`` -- and this test holds it to the same
rule: for every kernel the parity emitters build, both lowerers return the same
text, or both refuse the kernel with the same error type and message. So that
the rule cannot hold by both refusing everything alike, each family must lower
at least one kernel to the same source, unless ``FAMILIES_WITHOUT_HIP_LOWERING``
names what the HIP lowering lacks for it; such a family must lower none, and
refuse each kernel with NotImplementedError.

The input is the kernel the Python builder made. Whether the two engines'
builders produce the same IR is the differential gate's ``ir`` mode; here the
Python engine lowers that kernel and the C++ engine its serialization, so a
difference comes from the two HIP lowerings, or from something the
serialization does not carry, which is a defect as well.

Each family's emitter runs in its own process through ``python_batch_worker.py``
(the gate's worker) in the ``hip-case`` mode, which prints, for every kernel a
config builds, its target, its serialization and the Python engine's HIP
outcome, taken in that process from the built kernel
(``rocke.tests.hip_identity_cases``). The emitter also requires each
serialization to read back through the Python parser to the same text, and a
config whose serialization does not fails the test, named. A config the Python
builder rejects prints
nothing and has nothing to lower; the worker runs it again in the ``ir`` mode,
in the same process, and the test checks that it is rejected there with the
same error, so a failure of the hip-case path cannot pass as a rejection. It
fails any config that exits non-zero after printing output.

Each family's check -- its emitter run and the C++ lowering of its kernels --
runs in a worker thread ahead of the test that reads it, several families at a
time. The emitter run is a subprocess, as is the C++ lowering through the
program below, so the families overlap on the CPUs this process may use.

The C++ lowering runs through the ``rocke_engine`` extension, or where it is not
built -- the installed test tree -- through the ``rocke_ir_lower_cli`` program
``ROCKE_IR_LOWER_CLI`` names; the family test is skipped without either.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pytest
from rocke.core.ir import F32, IRBuilder, KernelDef, PtrType
from rocke.core.ir_serialize import serialize
from rocke.tests.engines import (
    native_hip_outcomes,
    python_hip_outcome,
    skip_without_native_hip_lowering,
)
from rocke.tests.hip_identity_cases import (
    HipIdentityCase,
    SerializationRoundTripError,
    hip_case_text,
    parse_hip_cases,
    require_serialization_round_trip,
)
from rocke.tests.hip_outcomes import REFUSAL_ERROR_TYPES, HipRefusal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_diff

# Emitters that build no kernel, so they have no hip-case output.
NO_KERNEL_FAMILIES = {
    "conv_abi": "prints kernel argument lists, not a kernel",
}

# Families none of whose kernels the HIP lowering covers, with what it lacks:
# both engines refuse every kernel with NotImplementedError. Every other family
# must lower at least one kernel to the same source in both engines, so the
# check cannot pass by both refusing everything alike. A listed family that
# starts to lower a kernel fails until it is removed here.
FAMILIES_WITHOUT_HIP_LOWERING = {
    "attention_strided_kv": "math.exp2_fast",
    "buffer_atomic_fadd": "tile.buffer_atomic_fadd_f32",
    "coalesced_loader_async": "tile.global_load_async_to_lds",
    "conv_dgrad_patch_wmma": "tile.tensor_load_to_lds",
    "conv_dgrad_smallc_wmma": "tile.tensor_load_to_lds",
    "conv_direct_dgrad_wmma": "tile.tensor_load_to_lds; s_waitcnt where a target "
    "has split wait counters",
    "conv_direct_dw_fwd_wmma": "s_waitcnt where a target has split wait counters",
    "conv_direct_fwd_gstream_wmma": "tile.tensor_load_to_lds",
    "conv_direct_fwd_tapouter_wmma": "tile.tensor_load_to_lds",
    "conv_direct_fwd_wmma": "tile.tensor_load_to_lds; s_waitcnt where a target "
    "has split wait counters",
    "conv_fwd_small_wmma": "tile.tensor_load_to_lds",
    "conv_winograd_fused_1d_wmma": "tile.global_load_async_to_lds; s_waitcnt where "
    "a target has split wait counters",
    "gfx1151_wmma_fmha_decode": "math.exp2_fast; the bf16 WMMA atoms",
    "gfx1151_wmma_fmha_fwd_v2": "math.exp2_fast; the bf16 WMMA atoms",
    "gfx942_attention_tiled_3d": "math.exp2_fast; tile.readlane",
    "gfx950_attention_tiled_3d": "math.exp2_fast; tile.readlane",
    "xf32_bf16x3_mma": "the emulated XF32 atoms, which only the LLVM path expands",
}


def test_the_family_lists_name_families() -> None:
    """A family renamed or removed would leave a stale entry that no test reads."""
    names = {name for name, _ in FAMILIES}
    assert NO_KERNEL_FAMILIES.keys() <= names
    assert FAMILIES_WITHOUT_HIP_LOWERING.keys() <= names


# One family's emitter run: every config, one process. This bounds a hung
# emitter below the 120 s per-test limit of the rocke_hip_source_identity CTest,
# so the hang is reported here rather than as a pytest-timeout kill. A family's
# check then lowers its kernels with the C++ engine, a run that engines.py
# bounds on its own; it takes a few seconds for the largest family, so the two
# together stay well inside the per-test limit.
EMITTER_TIMEOUT_S = 100

# How a builder refusing a config ends: the exception line of one of these.
_REFUSAL_TYPE_NAMES = frozenset(
    error_type.__name__ for error_type in REFUSAL_ERROR_TYPES
)

FAMILIES = run_diff.find_families(python_only=True)

# At most this many families are checked at once, ahead of the tests that read
# them: one worker thread per CPU this process may use (run_diff's
# rocke.tests.host_cpus count). The cap bounds memory, as a family's check holds
# all of its kernels until it has compared them.
MAX_PREFETCHED_FAMILIES = 8


@dataclasses.dataclass(frozen=True)
class EmitterRun:
    """How one config of a family's hip-case emitter run ended.

    ``ir_recheck`` is the same config run again in the ``ir`` mode when the
    hip-case run failed without printing anything, or None. It runs in the same
    process, after the failed run, so it sees any module state that run left."""

    index: int
    output: bytes
    emission: run_diff.Emission
    ir_recheck: run_diff.Emission | None


def _hip_case_runs(name: str, src_dir: Path, indices: list[int]) -> list[EmitterRun]:
    """Run one family's emitter in the hip-case mode over the configs
    ``indices``, in one process; return every config it ran."""
    emitter = src_dir / f"{name}_emit.py"
    with tempfile.TemporaryDirectory(prefix=f"hip_identity_{name}_") as out_dir:
        proc = subprocess.run(
            [
                sys.executable,
                str(run_diff.PYTHON_BATCH_WORKER),
                "--output-dir",
                out_dir,
                "--recheck-failures-in",
                "ir",
                str(emitter),
                "hip-case",
                *map(str, indices),
            ],
            env=run_diff.python_emitter_env(src_dir),
            check=False,  # the assert below reports the worker's stderr
            capture_output=True,
            text=True,
            timeout=EMITTER_TIMEOUT_S,
        )
        assert (
            proc.returncode == 0
        ), f"{name}: batch worker failed:\n{proc.stderr[-4000:]}"
        runs = []
        for line in proc.stdout.splitlines():
            record = json.loads(line)
            out = Path(out_dir) / f"{record['idx']}.out"
            recheck = record.get("recheck")
            runs.append(
                EmitterRun(
                    index=record["idx"],
                    output=out.read_bytes() if out.exists() else b"",
                    emission=run_diff.emission_from_record(record),
                    ir_recheck=(
                        run_diff.emission_from_record(recheck) if recheck else None
                    ),
                )
            )
        return runs


def _unexpected_failures(runs: list[EmitterRun]) -> list[str]:
    """The configs of ``runs`` that failed for any reason but the end of the
    config list or the builder refusing the config.

    A config that exits non-zero after printing output, or exits zero with
    none, is a failure: its kernels would silently drop out of the comparison.
    A config that fails with no output is the builder's refusal only if it
    raised one of the refusal types (``REFUSAL_ERROR_TYPES``), so a builder
    crashing with, say, an AttributeError is a failure, and only if the ``ir``
    mode, which builds the same kernel and only serializes it, refuses it with
    the same error; otherwise the hip-case path itself failed."""
    failures = []
    refused: list[tuple[EmitterRun, str]] = []
    for run in runs:
        emission = run.emission
        if run_diff.is_end_emission(emission):
            continue
        if emission.returncode == 0 and emission.has_output:
            continue
        if emission.has_output:
            failures.append(
                f"config {run.index}: exited {emission.returncode} after printing "
                f"output: {run_diff.last_error_line(emission)}"
            )
        elif emission.returncode == 0:
            failures.append(f"config {run.index}: exited 0 without printing a kernel")
        else:
            error = run_diff.last_error_line(emission)
            if error.split(":", 1)[0] in _REFUSAL_TYPE_NAMES:
                refused.append((run, error))
            else:
                failures.append(f"config {run.index}: failed, not refused: {error}")
    for run, error in refused:
        recheck = run.ir_recheck
        assert recheck is not None, f"config {run.index}: the worker did not recheck it"
        if recheck.returncode == 0 or recheck.has_output:
            failures.append(
                f"config {run.index}: failed in hip-case ({error}) but builds in ir"
            )
        elif (ir_error := run_diff.last_error_line(recheck)) != error:
            failures.append(
                f"config {run.index}: failed in hip-case ({error}) and in ir with "
                f"another error ({ir_error})"
            )
    return failures


def _first_difference(python_outcome: str, native_outcome: str) -> str:
    python_lines, native_lines = (
        python_outcome.splitlines(),
        native_outcome.splitlines(),
    )
    for number, (python_line, native_line) in enumerate(
        zip(python_lines, native_lines), start=1
    ):
        if python_line != native_line:
            return (
                f"line {number}: Python {python_line.strip()!r}, "
                f"C++ {native_line.strip()!r}"
            )
    return f"Python has {len(python_lines)} lines, C++ {len(native_lines)}"


def _disagreement(
    python_outcome: str | HipRefusal, native_outcome: str | HipRefusal
) -> str | None:
    """How the two engines' outcomes for one kernel differ, or None if they agree."""
    if python_outcome == native_outcome:
        return None
    python_refuses = isinstance(python_outcome, HipRefusal)
    native_refuses = isinstance(native_outcome, HipRefusal)
    if python_refuses and native_refuses:
        return (
            f"both refuse, differently: Python {python_outcome}, C++ {native_outcome}"
        )
    if python_refuses:
        return f"only Python refuses: {python_outcome}"
    if native_refuses:
        return f"only C++ refuses: {native_outcome}"
    return _first_difference(python_outcome, native_outcome)


# An emitter with one config per way a config can end, for the check below.
_OUTCOME_EMITTER = """
import sys

def main():
    index, mode = int(sys.argv[1]), sys.argv[2]
    if index == 0:
        print("kernel")
    elif index == 1:
        raise ValueError("the builder refuses this config")
    elif index == 2:
        if mode == "hip-case":
            raise ValueError("the hip-case path broke")
        print("kernel")
    elif index == 3:
        print("half a kernel")
        raise RuntimeError("failed after printing")
    elif index == 4:
        return 0
    elif index == 5:
        raise ValueError(f"refused as {mode}")
    elif index == 6:
        raise AttributeError("the builder broke")
    elif index == 7:
        from rocke.tests.hip_identity_cases import require_serialization_round_trip

        if mode == "hip-case":
            require_serialization_round_trip("seeded", "not a kernel")
        print("kernel")
    else:
        sys.stderr.write(f"unknown config {index}\\n")
        return 1
"""


def test_only_builder_refusals_and_the_end_of_the_list_pass(tmp_path: Path) -> None:
    (tmp_path / "outcomes_emit.py").write_text(_OUTCOME_EMITTER, encoding="utf-8")
    runs = _hip_case_runs("outcomes", tmp_path, list(range(10)))
    assert [run.index for run in runs] == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # Only a config that fails without printing anything is run again in ir,
    # and the end of the list is not a failure.
    assert [run.index for run in runs if run.ir_recheck is not None] == [
        1,
        2,
        5,
        6,
        7,
    ]
    failures = _unexpected_failures(runs)
    assert failures == [
        "config 3: exited 1 after printing output: RuntimeError: failed after printing",
        "config 4: exited 0 without printing a kernel",
        "config 6: failed, not refused: AttributeError: the builder broke",
        (
            "config 7: failed, not refused: "
            "rocke.tests.hip_identity_cases.SerializationRoundTripError: "
            "require_serialization_round_trip failed - kernel='seeded', the Python "
            "parser raised ValueError: bad header 'not a kernel'; expected "
            "'ckdsl.ir <ver>', expected it to read the serialization"
        ),
        (
            "config 2: failed in hip-case (ValueError: the hip-case path "
            "broke) but builds in ir"
        ),
        (
            "config 5: failed in hip-case (ValueError: refused as hip-case) "
            "and in ir with another error (ValueError: refused as ir)"
        ),
    ]


def _store_one_kernel() -> KernelDef:
    builder = IRBuilder("store_one")
    out = builder.param("D", PtrType(F32, "global"))
    builder.global_store(out, builder.thread_id_x(), builder.const_f32(1.0))
    return builder.kernel


def test_a_hip_case_reads_back_as_its_serialization_and_python_outcome() -> None:
    """The Python outcome an emitter records for the built kernel is the one
    the Python engine gives its parsed serialization, lowered or refused."""
    kernel = _store_one_kernel()
    serialization = serialize(kernel)
    output = hip_case_text(kernel, "gfx942") + hip_case_text(kernel, "gfx000")
    lowered, refused = parse_hip_cases(output.encode("utf-8"))
    assert lowered == HipIdentityCase(
        "gfx942", serialization, python_hip_outcome(serialization, "gfx942")
    )
    assert "void store_one(" in lowered.python_outcome
    assert refused == HipIdentityCase(
        "gfx000", serialization, python_hip_outcome(serialization, "gfx000")
    )
    assert refused.python_outcome.error_type_name == "KeyError"


def test_hip_case_byte_counts_are_of_the_utf8_text() -> None:
    """Counted in characters, a non-ASCII message would shift where the next
    case starts."""
    output = "hip-case gfx942 3 refusal ValueError 6\nir\nnot \u00e9".encode()
    assert parse_hip_cases(output) == [
        HipIdentityCase("gfx942", "ir\n", HipRefusal("ValueError", "not \u00e9"))
    ]


_EXPECTED_HEADER = (
    "expected 'hip-case <arch> <bytes> source <bytes>' or "
    "'hip-case <arch> <bytes> refusal <error type> <bytes>'"
)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (
            b"target gfx942\nkernel",
            (
                "parse_hip_cases failed - header='target gfx942' at byte 0, "
                f"{_EXPECTED_HEADER}"
            ),
        ),
        (
            b"hip-case gfx942 2 source 1\nirxkernel",
            (
                "parse_hip_cases failed - header='kernel' at byte 30, "
                f"{_EXPECTED_HEADER}"
            ),
        ),
        (
            b"hip-case gfx942 10 source 5\nshort",
            (
                "parse_hip_cases failed - header='hip-case gfx942 10 source 5' at "
                "byte 0 counts 15 bytes, expected at most the 5 that follow it"
            ),
        ),
        (
            b"hip-case gfx942 1 source 0\n\xff",
            (
                "parse_hip_cases failed - header='hip-case gfx942 1 source 0' at "
                "byte 0, expected UTF-8 text after it: 'utf-8' codec can't decode "
                "byte 0xff in position 0: invalid start byte"
            ),
        ),
    ],
    ids=["not-a-case", "trailing-text", "truncated", "not-utf8"],
)
def test_output_that_is_not_hip_cases_is_refused(output: bytes, message: str) -> None:
    """Output that is not hip-case records fails loudly rather than dropping the
    kernels it holds from the comparison."""
    with pytest.raises(ValueError) as error:
        parse_hip_cases(output)
    assert str(error.value) == message


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            ("{ axis", "{  axis"),
            (
                "line 7 reads back as '    %tid_1 : i32 = gpu.thread_id (  ) "
                "{ axis = s:\"x\" }', expected '    %tid_1 : i32 = gpu.thread_id "
                '(  ) {  axis = s:"x" }\''
            ),
        ),
        (
            ("kernel @", "kernel  @"),
            (
                "the Python parser raised ValueError: bad kernel line 'kernel  "
                "@store_one {', expected it to read the serialization"
            ),
        ),
    ],
    ids=["reads-back-differently", "parser-rejects"],
)
def test_a_serialization_that_does_not_round_trip_is_refused(
    change: tuple[str, str], message: str
) -> None:
    """hip_case_text holds every serialization to reading back through the
    Python parser to the same text: the parser coverage the test had when it
    parsed each serialization itself."""
    serialization = serialize(_store_one_kernel())
    require_serialization_round_trip("store_one", serialization)
    with pytest.raises(SerializationRoundTripError) as error:
        require_serialization_round_trip("store_one", serialization.replace(*change, 1))
    assert str(error.value) == (
        f"require_serialization_round_trip failed - kernel='store_one', {message}"
    )


@pytest.mark.parametrize("arch", ["gfx 942", ""])
def test_a_hip_case_needs_a_one_word_arch(arch: str) -> None:
    """The header is split on whitespace, so an arch with any would be misread."""
    with pytest.raises(ValueError) as error:
        hip_case_text(_store_one_kernel(), arch)
    assert str(error.value) == (
        f"hip_case_text failed - arch={arch!r}, expected one word with no whitespace"
    )


@dataclasses.dataclass(frozen=True)
class FamilyCheck:
    """One family's hip-case run, reduced to what its test asserts, so that its
    kernels' sources are dropped once the two engines' outcomes are compared.

    ``failures`` are the configs that failed other than by the builder refusing
    them; ``disagreements`` say, for each kernel whose two outcomes differ,
    where it is and how they differ; ``python_refusals`` are the Python
    engine's outcomes, among the ``compared`` kernels, that refuse."""

    emissions: list[run_diff.Emission]
    failures: list[str]
    compared: int
    disagreements: list[str]
    python_refusals: list[HipRefusal]


def _check_family(name: str, src_dir: Path) -> FamilyCheck:
    """Run ``name``'s emitter in the hip-case mode over every config, lower
    the kernels of the configs that built them with the C++ engine, and compare
    each kernel's two outcomes."""
    runs = _hip_case_runs(name, src_dir, list(range(run_diff.config_limit(name))))
    emissions = [run.emission for run in runs]
    if name in NO_KERNEL_FAMILIES or any(map(run_diff.mode_unsupported, emissions)):
        return FamilyCheck(emissions, [], 0, [], [])
    failures = _unexpected_failures(runs)
    cases: list[tuple[str, HipIdentityCase]] = []
    for run in runs:
        if run.emission.returncode != 0:
            continue
        try:
            config_cases = parse_hip_cases(run.output)
        except ValueError as error:
            failures.append(f"config {run.index}: printed other than hip-case: {error}")
            continue
        cases += [
            (f"{name} config {run.index} kernel {number} ({case.arch})", case)
            for number, case in enumerate(config_cases)
        ]
    native_outcomes = (
        native_hip_outcomes([(case.serialized_kernel, case.arch) for _, case in cases])
        if cases
        else []
    )
    return FamilyCheck(
        emissions=emissions,
        failures=failures,
        compared=len(cases),
        disagreements=[
            f"{where}: {difference}"
            for (where, case), native_outcome in zip(cases, native_outcomes)
            if (difference := _disagreement(case.python_outcome, native_outcome))
            is not None
        ],
        python_refusals=[
            case.python_outcome
            for _, case in cases
            if isinstance(case.python_outcome, HipRefusal)
        ],
    )


@pytest.fixture(scope="module")
def family_checks(
    request: pytest.FixtureRequest,
) -> Iterator[dict[str, Future[FamilyCheck]]]:
    """Each selected family's check, started in the order the session runs the
    family tests, so a test waits only for its own family. The checks share one
    process safely because the ``rocke_engine`` binding holds the GIL: the C++
    HIP lowerer has mutable globals (``g_smem_registry``, ``g_dispatch``), so a
    binding that released the GIL would need a lock here. Started only where
    the C++ lowering can run; checks not yet started when the module ends are
    cancelled. Not meant for pytest-xdist, whose workers would each start every
    selected family."""
    skip_without_native_hip_lowering()
    source_dirs = dict(FAMILIES)
    selected = [
        item.callspec.params["name"]
        for item in request.session.items
        if getattr(item, "function", None) is test_both_engines_emit_the_same_hip_source
    ]
    pool = ThreadPoolExecutor(
        max_workers=min(MAX_PREFETCHED_FAMILIES, run_diff.available_cpus())
    )
    try:
        yield {
            name: pool.submit(_check_family, name, source_dirs[name])
            for name in selected
        }
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


@pytest.mark.parametrize(
    ("name", "src_dir"), FAMILIES, ids=[name for name, _ in FAMILIES]
)
def test_both_engines_emit_the_same_hip_source(
    name: str, src_dir: Path, family_checks
) -> None:
    check = family_checks[name].result()
    if name in NO_KERNEL_FAMILIES:
        assert check.emissions and all(
            map(run_diff.mode_unsupported, check.emissions)
        ), (
            f"{name} is listed as building no kernel "
            f"({NO_KERNEL_FAMILIES[name]}) but accepted the hip-case mode"
        )
        return
    assert not any(
        map(run_diff.mode_unsupported, check.emissions)
    ), f"{name}_emit.py does not accept the hip-case mode"
    assert not check.failures, (
        f"{name}: {len(check.failures)} configs failed other than by the builder "
        "refusing them:\n" + "\n".join(check.failures[:20])
    )

    compared = check.compared
    assert compared, f"{name}: the emitter built no kernel in any config"
    assert not check.disagreements, (
        f"{len(check.disagreements)} of {compared} {name} kernels lower to different "
        "HIP source:\n" + "\n".join(check.disagreements[:20])
    )
    refusals = check.python_refusals
    lowered = compared - len(refusals)
    if name in FAMILIES_WITHOUT_HIP_LOWERING:
        assert not lowered, (
            f"{name} is listed as having no HIP lowering "
            f"({FAMILIES_WITHOUT_HIP_LOWERING[name]}) but {lowered} of its "
            f"{compared} kernels lower; remove it from FAMILIES_WITHOUT_HIP_LOWERING"
        )
        unexpected = [
            refusal
            for refusal in refusals
            if refusal.error_type_name != "NotImplementedError"
        ]
        assert not unexpected, (
            f"{name}: {len(unexpected)} of {compared} kernels are refused other "
            f"than as not lowered, first: {unexpected[0]}"
        )
    else:
        assert lowered, (
            f"{name}: both engines refuse all {compared} kernels, so no HIP source "
            f"was compared; first refusal: {refusals[0]}"
        )
