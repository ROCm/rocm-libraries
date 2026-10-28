#!/usr/bin/env python3
"""Generate the intrinsic availability table (the "arch domain") by probing the
installed toolchain.

# Why this exists

rocke resolved an intrinsic `declare` on ONE axis, the LLVM flavor; the target
arch picked an ISA backend and never reached the decl table, so "is this
intrinsic available on this GPU" was checked nowhere at build time. See
`dsl_docs/development/arch_axis_proposal.md`. This tool measures that axis and
commits the result. The read side is `rocke.core.arch.domain`, which the
lowerer's `_need` chokepoint consults and `tools/check_arch_domain.py` gates.

# Two stages, because the two axes need different tools

Stage A -- the flavor axis -- asks whether this LLVM knows each name. Arch-free,
so all declarations share one `opt -S` module (recursively split only on
failure): LLVM resolves a recognised `llvm.*` declare on parse (attaching
attributes, remangling overloads) while an unrecognised name round-trips
verbatim. See `_name_exists_many`.

Stage B -- the arch axis -- preflights each target once, then compiles AND LINKS
expected-positive calls in one module per architecture. A failed batch is
recursively split until changed cells receive individual diagnostics. Expected
negative cells remain individual. It must link for the same reason
`check_ir_validity.py` does: a declare for a nonexistent intrinsic passes both
`opt -passes=verify` and `clang -S`, since to the backend it is an ordinary
external call. It runs in a subprocess because an unsupported intrinsic reaches
`report_fatal_error`, which kills the process.

    stage A resolves      -> continue to stage B
    stage A verbatim      -> "name_absent"   this SPELLING is not an intrinsic
                                             in this LLVM (flavor axis)
    stage A `opt` failed  -> no-data         the process died or rejected our
                                             text; the question was never
                                             answered, so nothing is recorded
                                             about the name (see NameVerdict)
    link OK, assembles    -> "ok"            available here
    Cannot select / fatal -> "arch_absent"   the name is real, this target
                                             cannot lower it (arch axis)
    link OK, the assembler -> "arch_absent"  the backend emitted an
      rejects the text                       instruction this target does not
                                             have

The last row exists because a link is not proof. The backend can select an
intrinsic for a target that lacks its instruction and write the encoding into
the object without complaint (llvm.amdgcn.cvt.scalef32.pk.f32.fp8 on gfx942
links, and the object holds a gfx950 instruction). The text assembler,
`llvm-mc` for the same target, checks every instruction against the target's
feature set, so each probe that links is also compiled to assembly and
assembled.

Three more buckets exist so that a non-answer is never recorded as a negative:

    invalid target ID     -> "target_unsupported"  this clang cannot target this
                                                   arch at all; it has no opinion
    clang crashed         -> "toolchain_crash"     asking the question killed the
                                                   compiler; we did not get an
                                                   answer, only a bug report
    anything else         -> "probe_error"         OUR module was malformed

Read `name_absent` with care: it means the exact mangled string rocke emits is
not a known intrinsic. A nonexistent operation and a mis-mangled overload suffix
look identical here -- both are decl-table bugs, but different ones.

# Provenance

A result is only meaningful against the toolchain that produced it, and a host
can only validate its own flavor. Every column records the clang identity and
the flavor rocke resolved, and the generator refuses to run when they disagree:
an artifact labelled with the wrong LLVM vintage looks authoritative.

Usage:
  python rocke/platform/tools/gen_arch_domain.py            # write the artifact
  python rocke/platform/tools/gen_arch_domain.py --check    # CI: regen is a no-op
  python rocke/platform/tools/gen_arch_domain.py --only mfma --verbose
  python rocke/platform/tools/gen_arch_domain.py --keep-ir DIR
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import json
import re
import subprocess
import sys
import tempfile
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Callable, Iterator, NamedTuple

from _bounded_run import run_bounded, stop_all
from _hostcaps import available_cpus
from _llvm_identity import flavor_of_clang, tool_identity

HERE = Path(__file__).resolve().parent
ROCKE = HERE.parent  # tools -> rocke/platform


def _bootstrap_sys_path() -> None:
    """Import rocke from a checkout or a relocated provider test artifact.

    Source execution finds ``python/`` beside ``tools/``. Installed provider
    tests stage this script below ``tests/arch_domain`` and the package at the
    artifact root. A standalone install supplies its site-packages directory
    through the CTest environment. Only existing layouts are added, so normal
    package resolution remains authoritative everywhere else.
    """
    candidates = (ROCKE / "python", HERE.parents[1])
    for path in candidates:
        if (path / "rocke").is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


# At import time, not in main(): the status vocabulary and the schema string
# below come from the read side, so `rocke` has to be importable before the
# module body finishes.
_bootstrap_sys_path()

from rocke.core.arch import domain as _domain

SCHEMA = _domain.SCHEMA
# The committed columns are package data, not source-tree data. Resolving them
# through the imported package makes ``--check`` compare the exact payload CI
# tests after relocation while retaining the same path in a source checkout.
DATA_DIR = Path(str(resource_files("rocke.core.arch") / "data"))


def default_out(flavor: str) -> Path:
    """One artifact per flavor, named after it.

    A host can only measure its own LLVM, so a single shared file would be
    overwritten by whichever flavor ran last and `--check` would fail on every
    machine with a different toolchain.
    """
    return DATA_DIR / f"intrinsic_arch_domain.{flavor}.json"


# The status vocabulary is owned by the read side (`rocke.core.arch.domain`)
# and re-exported here, so the writer and the readers cannot drift.
STATUS_OK = _domain.STATUS_OK
STATUS_NAME_ABSENT = _domain.STATUS_NAME_ABSENT
STATUS_ARCH_ABSENT = _domain.STATUS_ARCH_ABSENT
STATUS_TARGET_UNSUPPORTED = _domain.STATUS_TARGET_UNSUPPORTED
STATUS_TOOLCHAIN_CRASH = _domain.STATUS_TOOLCHAIN_CRASH
STATUS_TOOLCHAIN_TIMEOUT = _domain.STATUS_TOOLCHAIN_TIMEOUT
STATUS_PROBE_ERROR = _domain.STATUS_PROBE_ERROR

# Bumped when the probe *semantics* change -- not when this file is merely
# edited -- and recorded into every column, so a committed answer can be told
# apart from one an older generator measured. Example of why: the llvm20 column
# blessed at -O3 records `ok` on `llvm.amdgcn.raw.ptr.buffer.load.async.lds`
# because the call was deleted before it could hang and the empty module
# linked. At -O0 that probe does not terminate at all.
#
#   1 -- initial, probes at -O3
#   2 -- probes at -O0, per-probe timeout, provenance recorded
#   3 -- immarg values swept on a negative, winning value recorded
#   4 -- llvm23 diagnostics understood; the sweep also runs on probe_error
#   5 -- probes built from the declare `opt` resolved, so `immarg` comes from
#        the LLVM being measured instead of from the hand-written decl table
#   6 -- stage A can return no-data. A failed `opt` was recorded as
#        `name_absent` for every arch; it is now the real no-data status and
#        the key is held out of stage B. No cell moves on a healthy toolchain,
#        but the stamp must, because a generator-5 column cannot otherwise be
#        told apart from one that banked a crash as a definite negative.
GENERATOR = 6

# Hoisted out of `_probe` so the artifact records the flags that actually ran.
# `-O0` is the one that decides correctness (see `_probe`), which is why a
# column must carry proof it was used.
PROBE_CFLAGS = ("-O0", "-nogpulib")

# Bounds the sweep, not any one probe: a probe module is one declare and one
# call, so a healthy probe is quick. Without a budget a single non-terminating
# probe hangs the run and `--check` never returns.
PROBE_TIMEOUT_S = 60

# Each worker owns an external compiler process, and successful probes briefly
# spawn the linker or assembler too. Respect the host/cgroup CPU allowance but
# stop at a bounded fan-out so large bare-metal hosts do not turn one CTest into
# an unbounded process burst. ``--jobs`` remains the explicit operator override.
MAX_PROBE_JOBS = 32


# --------------------------------------------------------------------------
# Probe module synthesis
# --------------------------------------------------------------------------

# What a type may begin with, on both the return and the parameter side.
#
# An allowlist of type openers, not a blocklist of attributes: the attribute set
# grows every LLVM release (`noundef`, `range(...)`, LLVM 21 respelling
# `nocapture` as `captures(none)`), while the set of type openers is closed. A
# blocklist leaks the next new attribute into the probe's types and turns the
# whole row into probe_error.
_TYPE_OPENER_RE = re.compile(
    r"^(?:void|i\d+|half|bfloat|float|double|x86_fp80|fp128|ppc_fp128"
    r"|ptr|metadata|token|label|[<{\[])"
)

_BRACKET_OPEN = "<{["
_BRACKET_CLOSE = ">}]"


def _split_params(params: str) -> list[str]:
    """Split a declare's parameter list on top-level commas.

    Aggregate types contain commas of their own (`{ i32, i32 }`), so a plain
    `.split(",")` corrupts them.
    """
    out: list[str] = []
    depth = 0
    cur = ""
    for ch in params:
        if ch in "<{[(":
            depth += 1
        elif ch in ">}])":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _param_type(param: str) -> tuple[str, bool]:
    """The parameter's type and whether it is an ``immarg``.

    A parameter may carry attributes on *both* sides of its type -- `align 8
    ptr nocapture`, `noundef i32`, `ptr addrspace(8) readonly` -- so finding
    the type means finding where it starts and where it ends, not deleting the
    attributes we happen to know the names of. Deleting known names is what
    this used to do, and it leaked every attribute outside that list straight
    into the probe: `i1 immarg zeroext` became the type `i1 zeroext`, which is
    not a type, so the kernel signature and the call built from it were
    malformed and the whole row measured probe_error. `zeroext`, `signext`,
    `noundef`, `nonnull`, `nofpclass(...)` and LLVM 21's `captures(none)` were
    all already outside it.

    So: skip forward to the first token that opens a type (same allowlist the
    return side uses), take the type, and drop whatever follows. An aggregate
    or vector runs to its closing bracket, and `ptr` absorbs a following
    `addrspace(N)` because that is part of the type rather than an attribute.
    A parameter with no recognisable type opener is returned whole, so it
    fails loudly in the probe rather than silently becoming something else.
    """
    toks = param.split()
    # `immarg` is read off the tokens rather than the raw text so that an
    # attribute argument that merely contains the word cannot fake one.
    is_imm = "immarg" in toks
    start = next((i for i, tok in enumerate(toks) if _TYPE_OPENER_RE.match(tok)), None)
    if start is None:
        return " ".join(toks), is_imm

    end = start + 1
    if toks[start][0] in _BRACKET_OPEN:
        depth = 0
        for end in range(start, len(toks)):
            depth += sum(c in _BRACKET_OPEN for c in toks[end])
            depth -= sum(c in _BRACKET_CLOSE for c in toks[end])
            if depth == 0:
                break
        end += 1
    elif end < len(toks) and toks[end].startswith("addrspace("):
        end += 1
    return " ".join(toks[start:end]), is_imm


def _return_type(ret: str) -> str:
    """Drop return attributes, keeping the type.

    Walks tokens left to right and stops at the first one that opens a type;
    everything from there to the end is the type. An unrecognised leading token
    would therefore swallow the whole string rather than a prefix of it, so the
    caller is handed something that fails to parse loudly instead of IR that is
    subtly wrong.
    """
    toks = ret.split()
    for i, tok in enumerate(toks):
        if _TYPE_OPENER_RE.match(tok):
            return " ".join(toks[i:])
    return ret.strip()


_DECL_RE = re.compile(r"^declare\s+(.+?)\s+@([\w.]+)\((.*)\)\s*$")

# `Segmentation fault (core dumped)` vs `Segmentation fault` is a property of
# the host's core-dump settings, not of the crash. See `_crash_reason`.
_CORE_DUMPED_RE = re.compile(r"\s*\(core dumped\)\s*$")
# SelectionDAG diagnostics print allocator-dependent node addresses. They are
# evidence formatting, not part of the backend's answer, and retaining them
# makes two identical probes differ under ``--check``.
_DAG_NODE_ADDRESS_RE = re.compile(
    r"\b0x[0-9a-fA-F]+(?=:"
    r"(?:\d+\b|\s+(?:ch|glue|Untyped|[if]\d+|bf\d+|v\d+(?:[if]|bf)\d+|"
    r"nxv\d+(?:[if]|bf)\d+)\b))"
)
_DAG_BARE_OPERAND_ADDRESS_RE = re.compile(r"\b0x[0-9a-fA-F]+\b(?=\s*(?:,|$))")

# The buffer fat pointer is not a legal kernel-argument type and cannot be
# produced by an addrspacecast, so it is the one operand we still pass poison
# for. That is safe here: the gfx942 control above lowers fine with a poison
# resource as long as the *integer* operands are real values.
_FAT_PTR = "ptr addrspace(8)"

# Intrinsics with integer operands that must be concrete even though LLVM does
# not mark them ``immarg``.  The value is a known-valid transfer size.
_LITERAL_WITNESSES = {
    "raw.ptr.buffer.load.async.lds": 4,
    "raw.ptr.buffer.load.lds": 4,
}

# Intrinsics whose zero immediate is not semantically valid. These witnesses
# are part of the probe specification rather than values discovered by trying
# a generic sweep after a failure.
_IMMARG_WITNESSES = {
    "global.load.lds": 1,
    "raw.ptr.buffer.load.async.lds": 1,
    "raw.ptr.buffer.load.lds": 1,
}


class ProbeSpec(NamedTuple):
    """A declaration and its semantically valid operand witnesses."""

    decl: str
    literal_ints: int | None
    imm_int: int | None


def _probe_spec(key: str, decl: str) -> ProbeSpec:
    return ProbeSpec(
        decl=decl,
        literal_ints=_LITERAL_WITNESSES.get(key),
        imm_int=_IMMARG_WITNESSES.get(key),
    )


def _probe_module(
    decl: str,
    datalayout: str,
    literal_ints: int | None = None,
    imm_int: int | None = None,
    *,
    probe_name: str = "probe",
    metadata_id: int = 0,
) -> tuple[str, str]:
    """Build a minimal module that declares an intrinsic and calls it.

    Returns (module_text, ""), or ("", reason) when the declare cannot be
    parsed -- an unparseable row is a probe_error, never a negative result.

    Every non-constant operand is a real SSA value, not `poison`. `poison` is a
    valid constant of every first-class type and so looks like the obvious
    generic choice, but it makes the probe lie: a poison `i32` operand sends
    `raw.ptr.buffer.load.lds` into "Do not know how to expand this operator's
    operand!" on gfx942, where the same call with concrete integers lowers
    cleanly -- a false arch_absent. Integers, floats, vectors and aggregates
    therefore arrive as kernel arguments, and non-generic pointers as an
    addrspacecast of one.

    `decl` is normally the declare `opt` resolved (see `_name_exists`), not the
    hand-written row it came from, and that is what makes the `immarg` branch
    below trustworthy. rocke's decl table records `immarg` only where an author
    happened to add it, while LLVM checks the real intrinsic signature; passing
    a variable where a constant is required yields a module that is illegal
    regardless of target, and the two vintages disagree about how loudly they
    say so. llvm22 rejects it up front -- "immarg operand has non-immediate
    parameter" -- but llvm20 has no such verifier check and dies at ISel with
    "Cannot select", which is *also* what a genuinely unsupported intrinsic
    looks like. `wmma.i32.16x16x16.iu4`/`.iu8` were recorded `arch_absent` on
    every target at llvm20 for exactly that reason, when in fact both RDNA
    targets lower them; the hand-written row omits `immarg` on three `i1`
    operands and `opt` supplies it.

    `literal_ints`, when set by :class:`ProbeSpec`, replaces every integer
    kernel argument with a known-valid literal. This covers an operand LLVM
    does not mark `immarg` but cannot legalise as a variable either
    (`raw.ptr.buffer.load.lds`'s size operand). Keeping the witness explicit
    avoids speculative retries that could constant-fold an unsupported call.

    `imm_int`, when set, is the value given to every `immarg` operand in place
    of the default 0. A parameter that is already marked `immarg` is never
    touched by `literal_ints` -- it is a constant either way, so the rescue
    above has nothing to substitute -- but the *value* can still be illegal,
    and an illegal immediate is indistinguishable from an unsupported target
    from the diagnostic alone: both say `Cannot select`. `global.load.lds`
    takes a per-lane transfer size whose legal values are 1, 2 and 4, so 0
    recorded every CDNA target as incapable of an instruction they run in
    production. Its :class:`ProbeSpec` therefore starts with value 1.

    All `immarg` operands move together, as `literal_ints` does. A declare
    whose immediates have disjoint legal domains would need a cross product,
    and nothing in this table does; if one appears, it reads as `arch_absent`
    (conservative, and the direction that gets noticed) rather than as a wrong
    `ok`.

    The result is stored `volatile` so the call survives -O3; without it, DCE
    would drop the reference and the link would succeed spuriously.
    """
    m = _DECL_RE.match(decl.strip())
    if not m:
        return "", f"cannot parse declare: {decl!r}"
    ret, name, params = m.groups()
    ret = _return_type(ret)

    kargs: list[str] = []
    prologue: list[str] = []
    args: list[str] = []
    for i, param in enumerate(_split_params(params)):
        ty, is_imm = _param_type(param)
        if ty == "metadata":
            # Not a first-class type. rocke's only metadata operands are the
            # av.* scope lists, which want a real scope node, not an empty one.
            args.append(f"metadata !{metadata_id}")
        elif is_imm:
            imm = 0 if imm_int is None else imm_int
            args.append(f"{ty} {('true' if imm else 'false') if ty == 'i1' else imm}")
        elif ty == _FAT_PTR:
            args.append(f"{ty} poison")
        elif literal_ints is not None and re.fullmatch(r"i\d+", ty):
            lit = literal_ints
            args.append(f"{ty} {('true' if lit else 'false') if ty == 'i1' else lit}")
        elif ty == "ptr":
            args.append("ptr %base")
        elif re.fullmatch(r"ptr addrspace\(\d+\)", ty):
            prologue.append(f"  %p{i} = addrspacecast ptr %base to {ty}")
            args.append(f"{ty} %p{i}")
        else:
            kargs.append(f"{ty} %a{i}")
            args.append(f"{ty} %a{i}")

    call = f"call {ret} @{name}({', '.join(args)})"
    if ret == "void":
        tail = f"  {call}\n"
    else:
        tail = f"  %r = {call}\n  store volatile {ret} %r, ptr addrspace(1) %out\n"

    signature = ", ".join(["ptr addrspace(1) %out", "ptr %base", *kargs])
    text = (
        f'target datalayout = "{datalayout}"\n'
        'target triple = "amdgcn-amd-amdhsa"\n\n'
        f"{decl.strip()}\n\n"
        f"define amdgpu_kernel void @{probe_name}({signature}) {{\n"
        "entry:\n" + "".join(f"{line}\n" for line in prologue) + tail + "  ret void\n"
        "}\n\n"
        f'!{metadata_id} = !{{!"agent"}}\n'
    )
    return text, ""


def _batch_probe_module(specs: list[ProbeSpec], datalayout: str) -> tuple[str, str]:
    """Build one module whose distinct kernels exercise every declaration."""
    headers: list[str] = []
    declarations: list[str] = []
    functions: list[str] = []
    metadata: list[str] = []
    for index, spec in enumerate(specs):
        text, why = _probe_module(
            spec.decl,
            datalayout,
            literal_ints=spec.literal_ints,
            imm_int=spec.imm_int,
            probe_name=f"probe_{index}",
            metadata_id=index,
        )
        if why:
            return "", why
        header, declaration, function, node = text.strip().split("\n\n")
        if not headers:
            headers.append(header)
        if declaration not in declarations:
            declarations.append(declaration)
        functions.append(function)
        metadata.append(node)
    return (
        "\n\n".join(
            [*headers, "\n".join(declarations), *functions, "\n".join(metadata)]
        )
        + "\n",
        "",
    )


class NameVerdict(NamedTuple):
    """What the flavor axis learned about one declare.

    ``exists`` is deliberately tri-state. ``True``/``False`` are answers --
    `opt` ran, parsed the declare, and either resolved it as an intrinsic or
    printed it back verbatim. ``None`` is *not an answer*: the `opt` process
    itself failed, so nothing was measured, and ``status``/``evidence`` carry
    the real diagnostic for the no-data cell.

    Collapsing ``None`` into ``False`` is the defect this type exists to
    prevent. A crashed, OOM-killed or timed-out `opt` would otherwise be
    written as `name_absent` -- a definite negative, for all seven arches,
    with generic evidence -- and `name_absent` is exactly the status the lane
    and the readers treat as authoritative. The arch axis has had
    `toolchain_crash` / `toolchain_timeout` / `probe_error` since the first
    generator for this reason; the flavor axis simply never grew them, and a
    failure to measure recorded as data is worse than no column at all.
    """

    exists: bool | None
    canonical: str
    resolved: str
    status: str = ""
    evidence: str = ""


def _verdict_from_opt_output(decl: str, output_decls: list[str]) -> NameVerdict:
    """Classify one input declaration from a successful ``opt -S`` output.

    Batched discovery can contain aliases for the same declaration and
    overloaded intrinsics that LLVM remangles.  Prefer an exact name, then a
    remangled name with the input name as its prefix.  If neither survives,
    AutoUpgrade consumed the declaration, which is still a positive result.
    """
    original = _DECL_RE.match(decl.strip())
    if not original:
        return NameVerdict(
            None, "", "", STATUS_PROBE_ERROR, f"cannot parse declare: {decl!r}"
        )
    original_name = original.group(2)

    parsed: list[tuple[str, str, bool]] = []
    for line in output_decls:
        attributed = re.search(r"#\d+\s*$", line) is not None
        resolved = re.sub(r"\s*#\d+\s*$", "", line)
        match = _DECL_RE.match(resolved)
        if match:
            parsed.append((match.group(2), resolved, attributed))

    exact = next((item for item in parsed if item[0] == original_name), None)
    if exact is not None:
        canonical, resolved, attributed = exact
        if attributed:
            return NameVerdict(True, canonical, resolved)
        return NameVerdict(False, "", "")

    remangled = [item for item in parsed if item[0].startswith(original_name + ".")]
    if len(remangled) == 1:
        canonical, resolved, _attributed = remangled[0]
        return NameVerdict(True, canonical, resolved)
    if len(remangled) > 1:
        return NameVerdict(
            None,
            "",
            "",
            STATUS_PROBE_ERROR,
            f"opt produced multiple canonical declarations for {original_name}",
        )

    # An unrecognised llvm.* declaration survives verbatim.  A declaration
    # that vanished was consumed by AutoUpgrade and therefore is recognised.
    return NameVerdict(True, "(auto-upgraded)", "")


def _name_exists_many(
    opt: str, items: list[tuple[str, str]], scratch: Path
) -> dict[str, NameVerdict]:
    """Resolve declarations in one ``opt`` process, bisecting failed batches.

    A healthy flavor normally needs one process for the entire declaration
    table.  If one declaration or a transient tool failure poisons that module,
    recursive splitting isolates it without losing answers for its neighbors.
    Singleton leaves retain :func:`_name_exists`'s retry and diagnostics.
    """
    if not items:
        return {}

    # Alias keys can carry the exact same declaration. Repeating it in one
    # module makes opt reject a harmless redefinition and forces recursive
    # splitting. Resolve one representative and fan its verdict back out.
    keys_by_decl: dict[str, list[str]] = {}
    for key, decl in items:
        keys_by_decl.setdefault(decl, []).append(key)
    if len(keys_by_decl) != len(items):
        representatives = [(keys[0], decl) for decl, keys in keys_by_decl.items()]
        unique_verdicts = _name_exists_many(opt, representatives, scratch)
        return {
            key: unique_verdicts[keys[0]]
            for keys in keys_by_decl.values()
            for key in keys
        }

    if len(items) == 1:
        key, decl = items[0]
        return {key: _name_exists(opt, decl, scratch)}

    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", items[0][0])
    src = scratch / f"names.{stem}.{len(items)}.ll"
    src.write_text(
        "\n".join(decl.strip() for _key, decl in items) + "\n", encoding="utf-8"
    )
    try:
        proc = run_bounded(
            [opt, "-S", "-o", "-", str(src)],
            timeout=PROBE_TIMEOUT_S,
            cwd=scratch,
        )
    except subprocess.TimeoutExpired:
        proc = None

    if proc is None or proc.returncode != 0:
        middle = len(items) // 2
        return {
            **_name_exists_many(opt, items[:middle], scratch),
            **_name_exists_many(opt, items[middle:], scratch),
        }

    output_decls = [
        line for line in proc.stdout.splitlines() if line.startswith("declare ")
    ]
    return {key: _verdict_from_opt_output(decl, output_decls) for key, decl in items}


def _name_exists(opt: str, decl: str, scratch: Path) -> NameVerdict:
    """Ask this LLVM whether it recognises the declare's name as an intrinsic.

    Returns a :class:`NameVerdict`; see there for why ``exists`` is tri-state.

    This is the *flavor* axis, and it is worth answering separately because it
    is arch-free: a name either exists in this LLVM or it does not, and asking
    once per key replaces one link probe per arch.

    The oracle is that LLVM resolves a recognised `llvm.*` declare on parse --
    it attaches the intrinsic's attribute set and, for overloaded intrinsics,
    remangles the name (`ds.read.tr16.b64` prints back as `...b64.v4i16`). An
    unrecognised `llvm.*` name is just an external function and round-trips
    verbatim. No codegen is involved, so this also answers the cases where
    codegen crashes outright: an `llvm.*` name with a `metadata` operand that
    LLVM does not know is lowered as an ordinary call, and computing the
    alignment of a metadata argument segfaults the backend.

    The third return value is the load-bearing one for stage B. "It attaches
    the intrinsic's attribute set" includes the `immarg` markers, which is the
    authoritative answer to "which operands must be constants" -- straight from
    the LLVM we are measuring, rather than from whichever ones a rocke author
    happened to write down. `_probe_module` builds from this rather than from
    the hand-written row, so a probe cannot pass a variable where the intrinsic
    demands a constant. See the `wmma.i32.16x16x16.iu8` note there for what
    that used to cost.
    """
    src = scratch / "name.ll"
    src.write_text(decl.strip() + "\n", encoding="utf-8")

    # Two attempts, same budget. A parse verdict is stable under retry; a
    # resource failure -- OOM, fork limit, a transient under -j elsewhere on
    # the box -- often is not, which is the same bargain the arch axis strikes
    # in its serial second pass.
    proc = None
    for attempt in (1, 2):
        try:
            proc = run_bounded(
                [opt, "-S", "-o", "-", str(src)],
                timeout=PROBE_TIMEOUT_S,
                cwd=scratch,
            )
        except subprocess.TimeoutExpired:
            if attempt == 1:
                continue
            return NameVerdict(
                None,
                "",
                "",
                STATUS_TOOLCHAIN_TIMEOUT,
                f"opt did not terminate within {PROBE_TIMEOUT_S:g}s",
            )
        if proc.returncode == 0 or attempt == 2:
            break

    assert proc is not None
    if proc.returncode != 0:
        # `opt` ran and failed. Which no-data status depends on who was at
        # fault, and both are recorded rather than guessed at: a death by
        # signal or an LLVM-side abort is the toolchain's, a parse diagnostic
        # about our own text is the probe's. Neither is evidence that the name
        # is absent -- that question was never reached.
        diag = (proc.stderr or proc.stdout).strip()
        low = diag.lower()
        crashed = (
            proc.returncode < 0
            or "llvm error" in low
            or "stack dump" in low
            or "out of memory" in low
        )
        status = STATUS_TOOLCHAIN_CRASH if crashed else STATUS_PROBE_ERROR
        return NameVerdict(
            None,
            "",
            "",
            status,
            f"opt exited {proc.returncode} on the declare: {_first_error(diag)}",
        )
    output_decls = [
        line for line in proc.stdout.splitlines() if line.startswith("declare ")
    ]
    return _verdict_from_opt_output(decl, output_decls)


# --------------------------------------------------------------------------
# Probe execution + classification
# --------------------------------------------------------------------------


def _classify(rc: int, diag: str) -> tuple[str, str]:
    """Map a clang invocation to (status, evidence).

    Order matters: "invalid target ID" is a frontend rejection that can coexist
    with nothing else, and must be checked before the backend diagnostics.
    """
    if rc == 0:
        return STATUS_OK, ""
    low = diag.lower()
    if "invalid target id" in low:
        return STATUS_TARGET_UNSUPPORTED, "invalid target ID"
    if "undefined symbol" in low:
        return STATUS_NAME_ABSENT, _first_error(diag)
    # Four ways the backend says "this target cannot lower that": no ISel
    # pattern, type legalisation cannot break the operand down, the lowering
    # wants a libcall the device lacks, and (llvm23) it says so outright. All
    # are reached only after the name resolved, so the flavor axis is already
    # settled by _name_exists() and cannot be confused with these.
    if (
        "cannot select" in low
        or "do not know how to expand this operator's operand" in low
        or "no libcall available for" in low
        or "intrinsic not supported on subtarget" in low
    ):
        return STATUS_ARCH_ABSENT, _first_error(diag)
    # The compiler died rather than answered. NOT arch_absent: we learned only
    # that asking crashes LLVM (`permlane64` on wave64 targets segfaults ISel
    # on llvm20). A crash is reproducible, so it stays stable under `--check`.
    if "please submit a bug report" in low or "stack dump:" in low:
        return STATUS_TOOLCHAIN_CRASH, _crash_reason(diag)
    # Unrecognised failure. Attributing it to the arch would be a guess, and a
    # wrong guess here silently bakes a false negative into the artifact.
    return STATUS_PROBE_ERROR, _first_error(diag)


def _crash_reason(diag: str) -> str:
    """Summarise a clang crash without embedding host-specific paths.

    Keep only the pass name and the signal. The dump leads with the full
    `-cc1` command line (temp dirs, resource-dir path), and the driver appends
    `(core dumped)` depending on the host's `ulimit -c`. Either would make two
    machines that saw the identical crash record different evidence, so
    `--check` would report drift where there is none.
    """
    parts = []
    for line in diag.splitlines():
        s = line.strip()
        if s.startswith("Running pass"):
            parts.append(s.rstrip("."))
        elif "unable to execute command:" in s:
            reason = s.split("unable to execute command:")[-1].strip()
            parts.append(_CORE_DUMPED_RE.sub("", reason).strip())
    return "; ".join(parts[-2:])[:200] or "clang crashed"


def _split_dag_opcode(rhs: str) -> tuple[str, str]:
    """Split a SelectionDAG right-hand side into opcode and operands."""
    closing_delimiters = {"<": ">", "(": ")", "[": "]", "{": "}"}
    delimiter_stack = []
    for index, char in enumerate(rhs):
        if char in closing_delimiters:
            delimiter_stack.append(closing_delimiters[char])
        elif delimiter_stack and char == delimiter_stack[-1]:
            delimiter_stack.pop()
        elif char.isspace() and not delimiter_stack:
            return rhs[:index], rhs[index:]
    return rhs, ""


def _normalize_dag_node_addresses(text: str) -> str:
    """Remove node identities from a SelectionDAG ``Cannot select`` dump."""
    marker = "cannot select:"
    marker_at = text.lower().find(marker)
    if marker_at < 0:
        return text
    dag_at = marker_at + len(marker)
    prefix = text[:dag_at]
    dag = _DAG_NODE_ADDRESS_RE.sub("0xADDR", text[dag_at:])

    # Result zero is implicit in SelectionDAG operands, so LLVM prints those
    # node references as bare addresses rather than ``0x...:0``. Restrict this
    # replacement to the operand list so semantic hexadecimal values elsewhere
    # in the diagnostic remain evidence.
    lhs, separator, rhs = dag.partition(" = ")
    if separator:
        opcode, operands = _split_dag_opcode(rhs)
        if operands:
            operands = _DAG_BARE_OPERAND_ADDRESS_RE.sub("0xADDR", operands)
            dag = lhs + separator + opcode + operands

    return prefix + dag


def _first_error(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if "error" in low or "undefined symbol" in low or "cannot select" in low:
            return _normalize_dag_node_addresses(s)[:200]
    first = (text.strip().splitlines() or ["(no diagnostic)"])[0]
    return _normalize_dag_node_addresses(first)[:200]


def _target_preflight_module(datalayout: str) -> str:
    """A target-only module used once per compiler/architecture pair."""
    return (
        f'target datalayout = "{datalayout}"\n'
        'target triple = "amdgcn-amd-amdhsa"\n\n'
        "define amdgpu_kernel void @target_preflight() {\n"
        "entry:\n"
        "  ret void\n"
        "}\n"
    )


def _target_preflight(clang: str, path: Path, arch: str, out: Path) -> tuple[str, str]:
    """Probe target support once; the scheduler owns any serial retry."""
    return _probe(clang, path, arch, out, verify_assembly=False)


def _probe(
    clang: str,
    path: Path,
    arch: str,
    out: Path,
    timeout: float = PROBE_TIMEOUT_S,
    *,
    verify_assembly: bool = True,
) -> tuple[str, str]:
    """Compile AND LINK one probe module for one arch.

    `-nogpulib` is required, not an optimisation: without it clang links the
    ROCm device bitcode, which on a multi-install host can come from a DIFFERENT
    ROCm than the one rocke resolved (observed: a 7.1 bitcode set pulled into a
    probe, with datalayout-mismatch warnings). The probe must measure the
    compiler, not the device libraries.

    `-O0` is also required, and for a subtler reason: at -O3 the IR pipeline can
    delete the very call we are asking about, and a module with nothing left to
    select links happily. llvm.amdgcn.permlane64 on gfx942 is the case that
    found this -- a kernel argument is wave-uniform, permlane64 of a uniform
    value folds to the identity, and LLVM 22's InstCombine folds it away. The
    object contained no permlane64 at all, the link succeeded, and the probe
    reported ok for a target where the instruction does not exist. Nine keys
    were foldable this way when that was measured. -O0 keeps the call alive to
    ISel, which is the only place that can answer the question.

    Keeping the call alive also exposes probes that never terminate -- llvm20
    spins indefinitely on `raw.ptr.buffer.load.async.lds` for every arch that
    can target it -- so the timeout is part of the same bargain rather than a
    defensive extra. A probe that runs out of budget is reported as
    `toolchain_timeout` and never as `arch_absent`: not finishing is not an
    answer about the target.
    """
    argv = [
        clang,
        "-x",
        "ir",
        *PROBE_CFLAGS,
        "-target",
        "amdgcn-amd-amdhsa",
        f"-mcpu={arch}",
        "-o",
        str(out),
        str(path),
    ]
    try:
        proc = run_bounded(
            argv,
            timeout=timeout,
            # Run from the scratch directory, not from wherever the sweep was
            # invoked. Probing is *expected* to crash clang -- that is what
            # `toolchain_crash` records -- and a crash on a host with core dumps
            # enabled drops a multi-megabyte `core` next to the caller. Run the
            # sweep from a checkout and those land in the worktree.
            cwd=out.parent,
        )
    except subprocess.TimeoutExpired:
        return (
            STATUS_TOOLCHAIN_TIMEOUT,
            f"clang did not terminate within {timeout:g}s",
        )
    status, evidence = _classify(proc.returncode, (proc.stderr or proc.stdout).strip())
    if status != STATUS_OK or not verify_assembly:
        return status, evidence
    return _assemble(clang, path, arch, out, timeout)


def _llvm_mc_beside(clang: str) -> Path:
    """The `llvm-mc` of the same install as `clang`: the assembler must judge
    the instructions the measured backend emits, not another LLVM's."""
    return Path(clang).with_name("llvm-mc")


def _assembler_rejection(diag: str) -> str:
    """`llvm-mc`'s rejection without the scratch path: the message and the
    rejected mnemonic (the source line it echoes after the error)."""
    lines = diag.splitlines()
    for i, line in enumerate(lines):
        if "error:" in line:
            message = line.split("error:", 1)[1].strip()
            echoed = next((l.split() for l in lines[i + 1 :] if l.strip()), [])
            mnemonic = f" ({echoed[0]})" if echoed else ""
            return f"assembler: {message}{mnemonic}"
    return "assembler: " + _first_error(diag)


def _assemble(
    clang: str,
    path: Path,
    arch: str,
    out: Path,
    timeout: float = PROBE_TIMEOUT_S,
) -> tuple[str, str]:
    """Compile a probe that linked to assembly text and assemble it for `arch`.

    `ok` only if the target's assembler accepts every instruction. Its
    "instruction not supported on this GPU" is a real arch answer
    (`arch_absent`); any other assembler failure is ours to explain
    (`probe_error`), never a silent `ok`.
    """
    asm = out.with_suffix(".s")
    argv = [
        clang,
        "-x",
        "ir",
        *PROBE_CFLAGS,
        "-target",
        "amdgcn-amd-amdhsa",
        f"-mcpu={arch}",
        "-S",
        "-o",
        str(asm),
        str(path),
    ]
    mc = [
        str(_llvm_mc_beside(clang)),
        "-triple=amdgcn-amd-amdhsa",
        f"-mcpu={arch}",
        "-filetype=null",
        str(asm),
    ]
    for step in (argv, mc):
        try:
            proc = run_bounded(step, timeout=timeout, cwd=out.parent)
        except subprocess.TimeoutExpired:
            return (
                STATUS_TOOLCHAIN_TIMEOUT,
                f"{Path(step[0]).name} did not terminate within {timeout:g}s",
            )
        if proc.returncode:
            diag = (proc.stderr or proc.stdout).strip()
            if step is mc and "instruction not supported on this gpu" in diag.lower():
                return STATUS_ARCH_ABSENT, _assembler_rejection(diag)
            return STATUS_PROBE_ERROR, f"{Path(step[0]).name}: " + _first_error(diag)
    return STATUS_OK, ""


# --------------------------------------------------------------------------
# Inputs: the decl table, the arch list, the toolchain identity
# --------------------------------------------------------------------------


def _decl_table(flavor: str) -> dict[str, str]:
    """The decls a _Lowerer would resolve for this flavor.

    Mirrors `_Lowerer.__init__` exactly -- base table, then the per-flavor
    override dict. Duplicating the merge here rather than importing a private
    helper keeps the tool honest about what it measured, but it is a mirror and
    must be updated if the resolution rule gains a rung.
    """
    from rocke.core import lower_llvm as L

    decls = dict(L._INTRINSIC_DECLS)
    if flavor == L.LLVM_FLAVOR_LLVM22:
        decls.update(L._INTRINSIC_DECLS_LLVM22_OVERRIDES)
    elif flavor == L.LLVM_FLAVOR_LLVM23:
        decls.update(L._INTRINSIC_DECLS_LLVM23_OVERRIDES)
    return decls


def _drift(committed: str, fresh: str) -> str | None:
    """Describe how a committed column differs from a fresh run, or None.

    None means "differs only in which compiler build answered" -- every cell
    agrees, and so does every field describing how the probe was posed. That is
    not staleness and must not fail: `toolchain.clang` records the exact ROCm
    build, so a byte comparison reds on any host whose patch level differs from
    the one that blessed the column, which is most of them. This check exists to
    catch a *wrong measurement*, and a column this host reproduces cell for cell
    is not one.

    Everything else is drift, including the rest of `toolchain`. Those fields
    are our own settings rather than properties of the host: `generator`,
    `probe_cflags` and `probe_timeout_s` say how the question was asked, and a
    change to any of them means the committed answers were obtained by a method
    we no longer use. `probe_cflags` is the concrete example -- the same clang
    reports `ok` or hangs for the same key depending on whether it carries
    `-O0` -- so forgiving a difference there would forgive precisely the defect
    that field was added to make visible.
    """
    try:
        a, b = json.loads(committed), json.loads(fresh)
    except json.JSONDecodeError as exc:
        return f"committed column is not valid JSON: {exc}"

    if a.get("keys") != b.get("keys"):
        changed = [
            f"{k}/{arch}: {a['keys'][k][arch]['status']} -> {cell['status']}"
            for k, row in b.get("keys", {}).items()
            for arch, cell in row.items()
            if arch in a.get("keys", {}).get(k, {})
            and a["keys"][k][arch]["status"] != cell["status"]
        ]
        head = "; ".join(changed[:5]) or "cell evidence or coverage differs"
        more = f" (+{len(changed) - 5} more)" if len(changed) > 5 else ""
        return f"measurements changed -- {head}{more}"

    for doc in (a, b):
        doc.get("toolchain", {}).pop("clang", None)
    if a != b:
        fields = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        return f"differs in {', '.join(fields)}, not in the measurements"
    return None


def _prune() -> int:
    """Drop rows for decl keys that no longer exist, on every committed column.

    The one repair that needs no toolchain, and the one the sweep cannot do.
    A host can only regenerate its own flavor, so when a key is renamed or
    deleted -- a rebase onto a develop that did it is the realistic route, and
    is how `tanh.f32` and the two pre-rename `wmma.scale` keys went stale in
    all three columns at once -- every other column keeps measuring something
    the tree no longer emits, and stays red on every machine until someone
    finds three toolchains. Nobody has three toolchains.

    Deleting is safe in a way that writing never is: the answer being removed
    was measured, is still true of the LLVM that produced it, and is simply no
    longer a question rocke asks. Nothing is invented, so the column keeps its
    provenance and the remaining cells keep their meaning. Adding the *new*
    key still needs that flavor's compiler, and until someone runs it the
    artifact test reports a named skip -- which is the honest state.
    """
    from rocke.core import lower_llvm as L  # noqa: F401  (loads the decl tables)

    changed = 0
    for path in sorted(DATA_DIR.glob("intrinsic_arch_domain.*.json")):
        flavor = path.name.split(".")[1]
        doc = json.loads(path.read_text())
        expected = set(_decl_table(flavor))
        stale = sorted(set(doc.get("keys", {})) - expected)
        if not stale:
            print(f"   {path.name}: nothing to prune")
            continue
        for key in stale:
            doc["keys"].pop(key, None)
            doc.get("canonical", {}).pop(key, None)
        path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        changed += 1
        print(f"   {path.name}: dropped {len(stale)} stale row(s): {', '.join(stale)}")

    if not changed:
        print("\nOK: every column already covers exactly the current decl table.")
        return 0
    print(
        f"\nwrote {changed} column(s). Rows were only ever removed; a key the "
        "decl table gained still needs a sweep on that flavor's toolchain."
    )
    return 0


def _nonnegative_jobs(raw: str) -> int:
    """Parse ``--jobs`` while preserving zero as the automatic sentinel."""
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("--jobs must be zero or greater")
    return value


def _probe_jobs(requested: int) -> int:
    """Resolve the quota-aware default while preserving an explicit override."""
    return requested or min(MAX_PROBE_JOBS, available_cpus())


@contextlib.contextmanager
def _probe_pool(max_workers: int) -> Iterator[cf.ThreadPoolExecutor]:
    """A thread pool for probes that stops promptly on Ctrl-C.

    Ctrl-C reaches only the main thread, and every probe runs in its own
    session, so the terminal's interrupt reaches neither the workers nor their
    compilers. Leaving a plain pool would then wait out each running probe's
    timeout. On `KeyboardInterrupt` this cancels the queued probes and ends the
    running ones before the pool waits for its workers.
    """
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        try:
            yield ex
        except KeyboardInterrupt:
            ex.shutdown(wait=False, cancel_futures=True)
            stop_all()
            raise


def _schedule_probes(
    keys: list[str],
    arches: list[str],
    absent: set[str],
    unmeasured: dict[str, tuple[str, str]],
    unsupported_arches: set[str],
    expected_cells: dict[str, dict[str, dict[str, str]]],
) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    """Partition every measurable cell into a positive batch or singleton."""
    positive_batches: dict[str, list[str]] = {arch: [] for arch in arches}
    individual_work: list[tuple[str, str]] = []
    for key in keys:
        if key in absent or key in unmeasured:
            continue
        for arch in arches:
            if arch in unsupported_arches:
                continue
            if expected_cells.get(key, {}).get(arch, {}).get("status") == STATUS_OK:
                positive_batches[arch].append(key)
            else:
                individual_work.append((key, arch))
    return positive_batches, individual_work


def _localize_expected_positive(
    keys: list[str],
    batch_status: Callable[[list[str]], str],
    single_result: Callable[[str], tuple[str, str, int | None]],
) -> dict[str, tuple[str, str, int | None]]:
    """Prove a positive batch or recursively isolate every changed cell."""
    if not keys:
        return {}
    if len(keys) == 1:
        status, evidence, imm = single_result(keys[0])
        return {keys[0]: (status, evidence, imm)}
    if batch_status(keys) == STATUS_OK:
        return {key: (STATUS_OK, "", None) for key in keys}
    middle = len(keys) // 2
    return {
        **_localize_expected_positive(keys[:middle], batch_status, single_result),
        **_localize_expected_positive(keys[middle:], batch_status, single_result),
    }


def _needs_serial_retry(
    status: str,
    expected_status: str | None,
    *,
    evidence: str = "",
    expected_evidence: str = "",
) -> bool:
    """Whether a result needs a contention-free confirmation attempt.

    Any changed committed verdict is unexpected and gets confirmed. Matching
    crash and timeout statuses are deterministic only when their normalized
    evidence also matches. Without a committed expectation (generation and
    target preflight), only a non-answer is plausibly transient. ``probe_error``
    always retries because it is the generator admitting it did not classify
    the result.
    """
    if status == STATUS_PROBE_ERROR:
        return True
    if expected_status is not None:
        if status != expected_status:
            return True
        if status in (STATUS_TOOLCHAIN_CRASH, STATUS_TOOLCHAIN_TIMEOUT):
            return evidence != expected_evidence
        return False
    return status in (STATUS_TOOLCHAIN_CRASH, STATUS_TOOLCHAIN_TIMEOUT)


def _expected_preflight_statuses(
    arches: list[str], expected_cells: dict[str, dict[str, dict[str, str]]]
) -> dict[str, str]:
    """Derive the committed target-support answer without using it as verdict."""
    expected: dict[str, str] = {}
    for arch in arches:
        statuses = {
            cell["status"]
            for row in expected_cells.values()
            if (cell := row.get(arch)) is not None
        }
        if STATUS_TARGET_UNSUPPORTED in statuses:
            expected[arch] = STATUS_TARGET_UNSUPPORTED
        elif statuses:
            expected[arch] = STATUS_OK
    return expected


def _retry_target_preflights(
    clang: str,
    path: Path,
    ir_dir: Path,
    preflight: dict[str, tuple[str, str]],
    expected_statuses: dict[str, str],
) -> list[str]:
    """Confirm transient or changed preflights after their worker pool drains."""
    retry = [
        arch
        for arch, (status, _evidence) in preflight.items()
        if _needs_serial_retry(status, expected_statuses.get(arch))
    ]
    for arch in retry:
        preflight[arch] = _target_preflight(
            clang, path, arch, ir_dir / f"target_preflight.{arch}.hsaco"
        )
    return retry


def _terminal_cell(
    key: str,
    arch: str,
    flavor: str,
    absent: set[str],
    unmeasured: dict[str, tuple[str, str]],
    unsupported_arches: set[str],
    preflight: dict[str, tuple[str, str]],
) -> tuple[str, str] | None:
    """Return a non-Stage-B verdict, preserving flavor-axis precedence."""
    if key in absent:
        return STATUS_NAME_ABSENT, f"not an intrinsic in {flavor}"
    if key in unmeasured:
        return unmeasured[key]
    if arch in unsupported_arches:
        _status, evidence = preflight[arch]
        return STATUS_TARGET_UNSUPPORTED, evidence
    return None


def _serial_retry_items(
    work: list[tuple[str, str]],
    results: dict[str, dict[str, dict[str, object]]],
    expected_cells: dict[str, dict[str, dict[str, str]]],
) -> list[tuple[str, str]]:
    """Select every transient or expectation-mismatched cell for confirmation."""
    retry: list[tuple[str, str]] = []
    for key, arch in work:
        result = results[key][arch]
        expected = expected_cells.get(key, {}).get(arch, {})
        if _needs_serial_retry(
            str(result["status"]),
            expected.get("status"),
            evidence=str(result.get("evidence", "")),
            expected_evidence=str(expected.get("evidence", "")),
        ):
            retry.append((key, arch))
    return retry


def _unvalidated_exit(requested: int | None, legacy: int) -> int:
    """Map an existing unvalidated result to a scheduler-requested skip code.

    The generator's legacy CLI uses different exits for different conditions.
    The source CTest asks only that those same conditions be distinguishable
    from green; it does not redefine which conditions are unvalidated.
    """
    return requested if requested is not None else legacy


def main() -> int:
    ap = argparse.ArgumentParser(description="rocKE intrinsic arch-domain generator")
    ap.add_argument(
        "--check", action="store_true", help="regenerate and fail on any diff"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="override the artifact path (default: one file per flavor)",
    )
    ap.add_argument("--only", default=None, help="substring filter on decl keys")
    ap.add_argument(
        "--arch", action="append", default=None, help="limit to these arches"
    )
    ap.add_argument(
        "--keep-ir", type=Path, default=None, help="keep probe modules here"
    )
    ap.add_argument(
        "--prune",
        action="store_true",
        help="drop rows whose decl key is gone, on every column; needs no toolchain",
    )
    ap.add_argument("--jobs", type=_nonnegative_jobs, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--unvalidated-exit-code",
        type=int,
        default=None,
        help=(
            "exit with this code when --check cannot validate on this host "
            "(CTest uses 77 to report a skip; defaults preserve legacy exits)"
        ),
    )
    args = ap.parse_args()
    if args.unvalidated_exit_code is not None:
        if not args.check:
            ap.error("--unvalidated-exit-code requires --check")
        if not 1 <= args.unvalidated_exit_code <= 255:
            ap.error("--unvalidated-exit-code must be between 1 and 255")

    sys.path.insert(0, str(HERE))
    from check_ir_validity import _llvm_tool  # same resolution order, one owner
    from rocke.core import lower_llvm as L
    from rocke.core.isa.backend import wired_arches

    # Before any toolchain resolution: pruning is a repair to columns this host
    # cannot measure, so requiring a compiler for it would defeat the purpose.
    if args.prune:
        print("rocKE intrinsic arch-domain generator -- prune")
        return _prune()

    # Before any probing, so a bad invocation fails at once rather than after a
    # full sweep, which is slowest on the oldest toolchain.
    #
    # `--only`/`--arch` select a subset, but the document written here is a
    # whole column. Writing a filtered sweep to the committed path would
    # *delete* the unselected cells, and the loader cannot tell a key that was
    # never measured from one this flavor lacks -- every lookup would silently
    # become no-data. `--check` is refused for the mirror reason: it would diff
    # a subset against the full column and report drift that is only the
    # filter. Both modes therefore demand their own destination.
    if (args.only or args.arch) and args.out is None:
        selectors = " ".join(
            s
            for s in (
                f"--only {args.only}" if args.only else "",
                " ".join(f"--arch {a}" for a in args.arch or ()),
            )
            if s
        )
        print(
            f"\nERROR: {selectors} selects part of the sweep, and "
            f"{'--check' if args.check else 'the default output path'} is the "
            "committed full column.\n"
            "      A partial sweep is not a column: unselected keys/arches "
            "would be missing, not negative,\n"
            "      and later lookups would read that as no-data.\n"
            "      Pass --out PATH to write the partial result somewhere of "
            "your own, or drop the filter to\n"
            "      regenerate the column in full.",
            file=sys.stderr,
        )
        return 2

    flavor = L._resolve_llvm_flavor()
    clang = _llvm_tool("clang")
    opt = _llvm_tool("opt")

    print("rocKE intrinsic arch-domain generator")
    print(f"   flavor : {flavor}")
    print(f"   clang  : {clang or '(not found)'}")
    print(f"   opt    : {opt or '(not found)'}")

    if clang is None:
        print(
            "\nRESULT: UNVALIDATED - no clang found under the resolved ROCm "
            "install or on PATH; nothing was probed. Set ROCKE_LLVM_BIN."
        )
        return _unvalidated_exit(args.unvalidated_exit_code, 0)

    # `opt` is not optional: it owns the flavor axis, and without it nothing
    # separates "this LLVM has no such intrinsic" from "this target cannot
    # lower it" -- an unresolved `llvm.*` name is an external call a link may
    # happen to accept, so keys would be banked as `ok`.
    #
    # Exits 1 where no-clang exits 0, deliberately. No clang is a legitimate
    # state (this host does not probe). Clang without `opt` is not a state any
    # complete LLVM install produces, so it means a partial environment -- a
    # broken invocation, not a quiet no-op.
    if opt is None:
        print(
            "\nFAIL: clang was found but `opt` was not, and the flavor axis is "
            "measured with `opt`; every key would be recorded as present "
            "without being checked. Nothing was probed. Set ROCKE_LLVM_BIN to "
            "a complete LLVM install (the one whose clang is above), or unset "
            "it to fall back to the ROCm that rocke itself loads."
        )
        return 1

    identity = tool_identity(clang)
    print(f"   version: {identity}")

    # A probe result is only meaningful against the flavor it was measured on.
    # Recording llvm20 results in an artifact stamped llvm23 would be actively
    # misleading, so disagreement is fatal rather than a warning.
    clang_flavor = flavor_of_clang(identity)
    if clang_flavor and clang_flavor != flavor:
        print(
            f"\nERROR: clang reports {clang_flavor} but rocke resolved {flavor}. "
            "The probe would be attributed to the wrong LLVM vintage. "
            "Set ROCKE_LLVM_BIN to the matching toolchain, or ROCKE_LLVM_FLAVOR "
            "if the override is intended."
        )
        return _unvalidated_exit(args.unvalidated_exit_code, 2)

    # `opt` and `clang` are found by independent searches, so a prefix that
    # ships clang but not `opt` lets `opt` fall through to another install.
    # Provenance records clang alone, so the column could be stamped llvm22
    # with its flavor axis measured by an llvm20 `opt` -- the same
    # mis-attribution the check above refuses, by another route. An
    # unparseable banner stays silent, per flavor_of_clang.
    opt_identity = tool_identity(opt)
    opt_flavor = flavor_of_clang(opt_identity)
    if opt_flavor and clang_flavor and opt_flavor != clang_flavor:
        print(
            f"\nERROR: opt reports {opt_flavor} but clang reports {clang_flavor}; "
            "they are different LLVM installs. Stage A would measure the flavor "
            "axis with one vintage and stage B the arch axis with another, and "
            "the result would be committed under clang's name alone.\n"
            f"      opt   : {opt} ({opt_identity})\n"
            f"      clang : {clang} ({identity})\n"
            "      Set ROCKE_LLVM_BIN to one complete LLVM install."
        )
        return 2

    # The assembler is not optional either: a probe that links is only `ok`
    # once the target's assembler accepts it (see `_assemble`).
    if not _llvm_mc_beside(clang).is_file():
        print(
            f"\nFAIL: no llvm-mc beside {clang}; every probe that links is "
            "assembled for its target before it is recorded as ok. Nothing was "
            "probed. Set ROCKE_LLVM_BIN to a complete LLVM install."
        )
        return 1

    arches = sorted(args.arch) if args.arch else sorted(wired_arches())
    decls = _decl_table(flavor)
    keys = sorted(k for k in decls if not args.only or args.only in k)
    print(f"   arches : {', '.join(arches)}")
    print(f"   keys   : {len(keys)}")

    out = args.out or default_out(flavor)
    expected_cells: dict[str, dict[str, dict[str, str]]] = {}
    if args.check and out.is_file():
        try:
            expected_cells = json.loads(out.read_text()).get("keys", {})
        except json.JSONDecodeError:
            # The final drift check owns the actionable malformed-JSON error.
            # Until then, schedule conservatively as though no baseline exists.
            pass

    tmp = tempfile.TemporaryDirectory(prefix="rocke_arch_domain_")
    ir_dir = args.keep_ir if args.keep_ir else Path(tmp.name)
    ir_dir.mkdir(parents=True, exist_ok=True)

    datalayout = L._datalayout_for_flavor(flavor)
    jobs = _probe_jobs(args.jobs)

    # Target support is an axis of the compiler/architecture pair, not of an
    # intrinsic.  Ask it once so a bad target ID cannot be rediscovered and
    # recorded independently for every declaration.
    preflight_path = Path(ir_dir) / "target_preflight.ll"
    preflight_path.write_text(_target_preflight_module(datalayout))

    def preflight_one(arch: str) -> tuple[str, tuple[str, str]]:
        return arch, _target_preflight(
            clang,
            preflight_path,
            arch,
            Path(ir_dir) / f"target_preflight.{arch}.hsaco",
        )

    preflight: dict[str, tuple[str, str]] = {}
    with _probe_pool(min(jobs, len(arches))) as ex:
        for arch, answer in ex.map(preflight_one, arches):
            preflight[arch] = answer
    preflight_retry = _retry_target_preflights(
        clang,
        preflight_path,
        Path(ir_dir),
        preflight,
        _expected_preflight_statuses(arches, expected_cells),
    )
    if preflight_retry:
        print(
            f"   retried {len(preflight_retry)} non-definitive target preflight(s) "
            "serially"
        )
    unsupported_arches = {
        arch
        for arch, (status, _evidence) in preflight.items()
        if status == STATUS_TARGET_UNSUPPORTED
    }
    broken_preflight = {
        arch: answer
        for arch, answer in preflight.items()
        if answer[0] not in (STATUS_OK, STATUS_TARGET_UNSUPPORTED)
    }
    if broken_preflight:
        print("\nFAIL: target preflight did not produce a support verdict:")
        for arch, (status, evidence) in broken_preflight.items():
            print(f"  {arch:14s} {status}: {evidence}")
        return 1

    # Stage A -- the flavor axis, once per key. Arch-free, so it costs one
    # `opt` run instead of one link per arch, and it is the only stage that can
    # answer a key whose codegen crashes. It runs first because the probe
    # modules are built from its output: `opt` returns the declare with this
    # LLVM's own attributes attached, `immarg` included.
    canonical: dict[str, str] = {}
    resolved: dict[str, str] = {}
    absent: set[str] = set()
    # Keys whose flavor-axis probe did not produce an answer, with the no-data
    # status and the diagnostic that explains it.
    unmeasured: dict[str, tuple[str, str]] = {}
    name_verdicts = _name_exists_many(
        opt, [(key, decls[key]) for key in keys], Path(ir_dir)
    )
    for key in keys:
        v = name_verdicts[key]
        if v.exists is None:
            unmeasured[key] = (v.status, v.evidence)
        elif v.exists:
            canonical[key] = v.canonical
            if v.resolved:
                resolved[key] = v.resolved
        else:
            absent.add(key)
    print(
        f"   names  : {len(keys) - len(absent) - len(unmeasured)} present, "
        f"{len(absent)} absent in {flavor}"
    )
    if unmeasured:
        # Loud, not a footnote: these keys leave a no-data hole in the
        # column, and an unmentioned hole is how the old silent `name_absent`
        # went unnoticed for two columns.
        print(f"   UNMEASURED: {len(unmeasured)} key(s) whose `opt` probe failed")
        for key, (status, evidence) in sorted(unmeasured.items()):
            print(f"     {key:44s} {status}: {evidence}")

    # Build every probe module; an unparseable declare is a probe_error for
    # every arch rather than a crash mid-sweep.
    #
    # Use `opt`'s resolved declare where stage A produced one; the
    # hand-written row covers the two cases that have none (no `opt` on this
    # host, and a declare AutoUpgrade consumed entirely).
    modules: dict[str, tuple[Path | None, str]] = {}
    specs: dict[str, ProbeSpec] = {}
    for key in keys:
        spec = _probe_spec(key, resolved.get(key, decls[key]))
        specs[key] = spec
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
        text, why = _probe_module(
            spec.decl,
            datalayout,
            literal_ints=spec.literal_ints,
            imm_int=spec.imm_int,
        )
        if why:
            modules[key] = (None, why)
            continue
        path = ir_dir / f"{stem}.ll"
        path.write_text(text)
        modules[key] = (path, "")

    # Stage B -- the arch axis, only for names that exist. A key stage A could
    # not measure is held out too: its probe would have to be built from the
    # hand-written row, which is the generator-4 construction that produced
    # false `arch_absent` cells.
    positive_batches, individual_work = _schedule_probes(
        keys,
        arches,
        absent,
        unmeasured,
        unsupported_arches,
        expected_cells,
    )
    work = [
        *individual_work,
        *(
            (key, arch)
            for arch, batch_keys in positive_batches.items()
            for key in batch_keys
        ),
    ]

    def run(item: tuple[str, str]) -> tuple[str, str, str, str, int | None]:
        key, arch = item
        path, why = modules[key]
        if path is None:
            return key, arch, STATUS_PROBE_ERROR, why, None
        out = Path(ir_dir) / f"{path.stem}.{arch}.hsaco"
        status, evidence = _probe(clang, path, arch, out)
        return (
            key,
            arch,
            status,
            evidence,
            specs[key].imm_int if status == STATUS_OK else None,
        )

    def run_positive_batch(
        item: tuple[str, list[str]],
    ) -> list[tuple[str, str, str, str, int | None]]:
        """Validate expected-positive cells together, splitting on failure."""
        arch, batch_keys = item

        def batch_status(part: list[str]) -> str:
            text, why = _batch_probe_module([specs[key] for key in part], datalayout)
            if why:
                return STATUS_PROBE_ERROR
            first = re.sub(r"[^A-Za-z0-9_.-]", "_", part[0])
            path = Path(ir_dir) / f"batch.{arch}.{first}.{len(part)}.ll"
            path.write_text(text)
            status, _evidence = _probe(clang, path, arch, path.with_suffix(".hsaco"))
            return status

        def single_result(key: str) -> tuple[str, str, int | None]:
            _key, _arch, status, evidence, imm = run((key, arch))
            return status, evidence, imm

        # The committed positives schedule batches but never decide verdicts.
        # A failed batch is split until ordinary single-probe classification
        # identifies every changed cell.
        answers = _localize_expected_positive(batch_keys, batch_status, single_result)
        return [
            (
                key,
                arch,
                status,
                evidence,
                (
                    imm
                    if imm is not None
                    else specs[key].imm_int if status == STATUS_OK else None
                ),
            )
            for key, (status, evidence, imm) in answers.items()
        ]

    # Threads, not processes: each unit of work is already its own subprocess.
    # ``jobs`` is quota-aware and bounded above; successful probes can briefly
    # add a linker or assembler child to each compiler worker.
    results: dict[str, dict[str, dict[str, str]]] = {k: {} for k in keys}

    def record(
        key: str, arch: str, status: str, evidence: str, imm: int | None = None
    ) -> None:
        cell: dict[str, object] = {"status": status, "verified_on": flavor}
        if evidence and status != STATUS_OK:
            cell["evidence"] = evidence
        # Only on a cell whose ProbeSpec declares a non-default immediate.
        # Present, it records the semantically valid witness used to establish
        # the positive result.
        if imm is not None:
            cell["probe_imm"] = imm
        results[key][arch] = cell

    # Stage A answers precede target support: a spelling absent from this LLVM
    # remains name_absent even where the compiler cannot target the requested
    # architecture. Only declarations eligible for Stage B inherit the target
    # preflight verdict. The same precedence preserves an opt failure's real
    # diagnostic instead of laundering it into target_unsupported.
    for key in keys:
        for arch in arches:
            terminal = _terminal_cell(
                key,
                arch,
                flavor,
                absent,
                unmeasured,
                unsupported_arches,
                preflight,
            )
            if terminal is not None:
                status, evidence = terminal
                record(key, arch, status, evidence)

    with _probe_pool(jobs) as ex:
        futures: dict[cf.Future, str] = {}
        for arch, batch in positive_batches.items():
            if batch:
                futures[ex.submit(run_positive_batch, (arch, batch))] = "batch"
        for item in individual_work:
            futures[ex.submit(run, item)] = "single"

        for future in cf.as_completed(futures):
            if futures[future] == "batch":
                for key, arch, status, evidence, imm in future.result():
                    record(key, arch, status, evidence, imm)
                continue
            key, arch, status, evidence, imm = future.result()
            record(key, arch, status, evidence, imm)
            if args.verbose and status != STATUS_OK:
                print(f"     {key:44s} {arch:14s} {status}")

    # Second pass, serial: unclassified, transient, and expectation-mismatched
    # probes get one more chance with no contention. A real result is stable
    # under retry, while a resource failure is not -- llvm.fabs.f32 on
    # gfx11-generic aborted on the process limit once in three runs, while
    # permlane64 on the wave64 targets reproduces serially. The retry separates
    # the two and confirms any changed definitive verdict before reporting drift.
    #
    # A timeout retries on the same budget: the point is to remove contention,
    # not to grant more time to a module this small.
    retry = _serial_retry_items(work, results, expected_cells)
    if retry:
        print(f"   retrying {len(retry)} unexpected/transient probe(s) serially...")
        for key, arch in retry:
            _k, _a, status, evidence, imm = run((key, arch))
            record(key, arch, status, evidence, imm)

    # `probe_error` is the generator reporting itself broken -- an
    # unrecognised diagnostic, or a probe module built wrong -- so it must
    # never be written. The artifact test already forbids it in a committed
    # column; refusing here stops it at the person who can still act on it.
    #
    # The fix is almost always to teach `_classify` a wording a new LLVM
    # introduced (llvm23's "intrinsic not supported on subtarget" was twelve
    # cells), so the evidence is printed rather than summarised.
    stuck = [
        (k, a, results[k][a].get("evidence", ""))
        for k in keys
        for a in arches
        if results[k][a]["status"] == STATUS_PROBE_ERROR
    ]
    if stuck:
        print(f"\n== {len(stuck)} unclassified probe(s) survived the serial retry ==")
        for key, arch, evidence in stuck[:20]:
            print(f"  {key:44s} {arch:14s} {evidence or '(no diagnostic)'}")
        if len(stuck) > 20:
            print(f"  ... and {len(stuck) - 20} more")
        print(
            f"\nFAIL: nothing written. `{STATUS_PROBE_ERROR}` means this tool "
            "did not understand the answer, not that the target cannot lower "
            "the intrinsic; committing it would record our defect as the "
            "toolchain's. Teach `_classify` the wording above, then re-run."
        )
        return 1

    doc = {
        "schema": SCHEMA,
        "_comment": (
            "GENERATED by tools/gen_arch_domain.py -- do not hand-edit. "
            "Which intrinsic declarations link for which gfx target, measured "
            "by compiling AND LINKING a probe module per (key, arch). Only the "
            "flavor named in toolchain.flavor was measured; every other flavor "
            "is unvalidated on this host by construction. See "
            "dsl_docs/development/arch_axis_proposal.md."
        ),
        # Provenance. `clang` identifies the compiler but not how it was
        # asked, and the same build reports `ok` or hangs for the same key
        # depending on the probe's optimisation level. A column missing these
        # fields was written by generator 1 and cannot be trusted on `ok`.
        "toolchain": {
            "flavor": flavor,
            "clang": identity,
            "arches": arches,
            "generator": GENERATOR,
            "probe_cflags": list(PROBE_CFLAGS),
            "probe_timeout_s": PROBE_TIMEOUT_S,
        },
        "keys": results,
        # What this LLVM resolved each surviving declare to. Mostly identical
        # to the declared name; the interesting rows are overloads that
        # remangle (`ds.read.tr16.b64` -> `...b64.v4i16`) and legacy names
        # AutoUpgrade rewrites -- both places where what rocke emits and what
        # the toolchain executes differ.
        "canonical": {k: v for k, v in sorted(canonical.items()) if v},
    }
    text = json.dumps(doc, indent=2, sort_keys=True) + "\n"

    counts: dict[str, int] = {}
    for key in keys:
        for arch in arches:
            s = results[key][arch]["status"]
            counts[s] = counts.get(s, 0) + 1
    print("\n   " + "  ".join(f"{s}={n}" for s, n in sorted(counts.items())))

    if args.check:
        if not out.is_file():
            # Not a failure: a host can only measure its own flavor, so
            # failing here would red every CI machine on a newer ROCm.
            print(
                f"\nUNVALIDATED: no committed column for {flavor} "
                f"({out.name}); nothing to check. Run without --check to add one."
            )
            return _unvalidated_exit(args.unvalidated_exit_code, 0)
        committed = out.read_text()
        if committed == text:
            print(f"\nOK: {out.name} is up to date.")
            return 0
        why = _drift(committed, text)
        if why is None:
            # Reproduced cell for cell by a different build of the same
            # flavor -- a stronger pass than a byte match, since two
            # independent builds agreeing shows the column describes the
            # flavor rather than one machine. See `_drift`.
            print(
                f"\nOK: {out.name} is up to date -- every cell reproduces "
                "here; only the compiler build differs.\n"
                f"      blessed by: {json.loads(committed)['toolchain']['clang']}\n"
                f"      this host : {identity}"
            )
            return 0
        print(
            f"\nFAIL: {out} is stale -- regenerating changed it.\n"
            f"      {why}\n"
            "      Re-run without --check and commit the result."
        )
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
