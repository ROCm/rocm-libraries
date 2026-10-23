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

Stage A -- the flavor axis -- asks whether this LLVM knows the name. Arch-free,
so it runs once per key, and it uses `opt -S`: LLVM resolves a recognised
`llvm.*` declare on parse (attaching attributes, remangling overloads) while an
unrecognised name round-trips verbatim. See `_name_exists`.

Stage B -- the arch axis -- compiles AND LINKS a probe module per (key, arch)
for names that survived stage A. It must link for the same reason
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
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

from _hostcaps import available_cpus
from _llvm_identity import flavor_of_clang, tool_identity

HERE = Path(__file__).resolve().parent
ROCKE = HERE.parent  # tools -> rocke/platform


def _bootstrap_sys_path() -> None:
    """Import rocke from the checkout without an external PYTHONPATH, matching
    tests/conftest.py. Unlike check_ir_validity this needs no library/ reach --
    the decl table lives entirely in platform."""
    path = ROCKE / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# At import time, not in main(): the status vocabulary and the schema string
# below come from the read side, so `rocke` has to be importable before the
# module body finishes.
_bootstrap_sys_path()

from rocke.core.arch import domain as _domain

SCHEMA = _domain.SCHEMA
DATA_DIR = ROCKE / "python" / "rocke" / "core" / "arch" / "data"


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

# Literal values the immediate-operand rescue tries, in order. See
# `_probe_module`'s `literal_ints` and `_wants_literals`.
_LITERAL_PROBE_VALUES = (0, 4)

# Values the `immarg` sweep tries, in order, when the default 0 produced a
# negative. See `_probe_module`'s `imm_int` and the sweep in `run`.
#
# Not exhaustive: these are the byte counts a load-to-LDS transfer size
# accepts, as llvm23's verifier enumerates them. A selector operand (MFMA
# cbsz/blgp) is legal at 0 and never reaches the sweep. Smallest-first, so the
# recorded winner is the least surprising legal value.
_IMMARG_PROBE_VALUES = (1, 2, 4, 12, 16)

# The ways LLVM says "that operand had to be an immediate". Same complaint,
# three vintages: llvm22 rejects it in the verifier, llvm20 has no such check
# and dies during type legalisation, llvm23 added a load-to-LDS size check that
# names the legal values outright.
_IMMEDIATE_OPERAND_DIAGS = (
    "immarg operand has non-immediate parameter",
    "do not know how to expand this operator's operand",
    "invalid data size for load-to-lds intrinsic",
)


def _wants_literals(status: str, evidence: str) -> bool:
    """True when a failure looks like our probe passed a variable where the
    intrinsic demands a constant, rather than a real arch answer."""
    if status not in (STATUS_ARCH_ABSENT, STATUS_PROBE_ERROR):
        return False
    low = evidence.lower()
    return any(d in low for d in _IMMEDIATE_OPERAND_DIAGS)


def _probe_module(
    decl: str,
    datalayout: str,
    literal_ints: int | None = None,
    imm_int: int | None = None,
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

    `literal_ints`, when set, replaces every integer kernel argument with that
    literal value. With the resolved declare in hand this is a backstop rather
    than the main line: it still covers an operand LLVM does not mark `immarg`
    but cannot legalise as a variable either (`raw.ptr.buffer.load.lds`'s size
    operand), and it stays scoped to the diagnostics that say so -- a blanket
    retry could constant-fold a genuinely unsupported intrinsic away and
    manufacture an `ok`, which is the one error direction that would make the
    artifact worse than no artifact.

    `imm_int`, when set, is the value given to every `immarg` operand in place
    of the default 0. A parameter that is already marked `immarg` is never
    touched by `literal_ints` -- it is a constant either way, so the rescue
    above has nothing to substitute -- but the *value* can still be illegal,
    and an illegal immediate is indistinguishable from an unsupported target
    from the diagnostic alone: both say `Cannot select`. `global.load.lds`
    takes a per-lane transfer size whose legal values are 1, 2 and 4, so 0
    recorded every CDNA target as incapable of an instruction they run in
    production. The only way to tell the two apart is to ask again with a
    different value.

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
            args.append("metadata !0")
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
        f"define amdgpu_kernel void @probe({signature}) {{\n"
        "entry:\n" + "".join(f"{line}\n" for line in prologue) + tail + "  ret void\n"
        "}\n\n"
        '!0 = !{!"agent"}\n'
    )
    return text, ""


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
            proc = subprocess.run(
                [opt, "-S", "-o", "-", str(src)],
                capture_output=True,
                text=True,
                check=False,
                timeout=PROBE_TIMEOUT_S,
                cwd=str(scratch),
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
    for line in proc.stdout.splitlines():
        if line.startswith("declare "):
            attributed = re.search(r"#\d+\s*$", line) is not None
            # The trailing `#N` references an attribute group this probe module
            # will not carry, so it is dropped rather than propagated.
            resolved = re.sub(r"\s*#\d+\s*$", "", line)
            m = _DECL_RE.match(resolved)
            canonical = m.group(2) if m else ""
            if attributed:
                return NameVerdict(True, canonical, resolved if m else "")
            # Remangled but attribute-free: still a resolved intrinsic.
            original = _DECL_RE.match(decl.strip())
            if original and canonical and canonical != original.group(2):
                return NameVerdict(True, canonical, resolved if m else "")
            return NameVerdict(False, "", "")
    # The declare did not survive the round-trip at all. An unrecognised
    # `llvm.*` name is an ordinary external function and is printed back
    # verbatim even when unused, so a vanished declare means AutoUpgrade
    # consumed it -- `amdgcn.global.atomic.fadd` becoming a plain `atomicrmw`
    # is the live example. That still links, so the name counts as present.
    # There is no resolved declare to build from; the hand-written row stands.
    return NameVerdict(True, "(auto-upgraded)", "")


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


def _probe(
    clang: str,
    path: Path,
    arch: str,
    out: Path,
    timeout: float = PROBE_TIMEOUT_S,
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
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            # Run from the scratch directory, not from wherever the sweep was
            # invoked. Probing is *expected* to crash clang -- that is what
            # `toolchain_crash` records -- and a crash on a host with core dumps
            # enabled drops a multi-megabyte `core` next to the caller. Run the
            # sweep from a checkout and those land in the worktree.
            cwd=str(out.parent),
        )
    except subprocess.TimeoutExpired:
        return (
            STATUS_TOOLCHAIN_TIMEOUT,
            f"clang did not terminate within {timeout:g}s",
        )
    status, evidence = _classify(proc.returncode, (proc.stderr or proc.stdout).strip())
    if status != STATUS_OK:
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
            proc = subprocess.run(
                step,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
                cwd=str(out.parent),
            )
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
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

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
        return 0

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
        return 2

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

    tmp = tempfile.TemporaryDirectory(prefix="rocke_arch_domain_")
    ir_dir = args.keep_ir if args.keep_ir else Path(tmp.name)
    ir_dir.mkdir(parents=True, exist_ok=True)

    datalayout = L._datalayout_for_flavor(flavor)

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
    for key in keys:
        v = (
            _name_exists(opt, decls[key], Path(ir_dir))
            if opt
            else NameVerdict(True, "", "")
        )
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
    literal_modules: dict[str, list[Path]] = {}
    imm_modules: dict[str, list[tuple[int, Path]]] = {}
    for key in keys:
        decl = resolved.get(key, decls[key])
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
        text, why = _probe_module(decl, datalayout)
        if why:
            modules[key] = (None, why)
            continue
        path = ir_dir / f"{stem}.ll"
        path.write_text(text)
        modules[key] = (path, "")
        # Two literal values, because neither is safe alone: 0 is rejected
        # where the operand is a byte count, and a nonzero value is out of
        # range where it is a small enum (an MFMA cbsz/blgp selector).
        variants = []
        for lit in _LITERAL_PROBE_VALUES:
            lit_text, why_lit = _probe_module(decl, datalayout, literal_ints=lit)
            if not why_lit and lit_text != text:
                lit_path = ir_dir / f"{stem}.lit{lit}.ll"
                lit_path.write_text(lit_text)
                variants.append(lit_path)
        if variants:
            literal_modules[key] = variants
        # The immarg sweep, built the same way. A declare with no integer
        # immarg yields a module identical to the base one, so the `!=` test
        # is what decides sweepability -- no separate signature analysis to
        # keep in step with `_probe_module`.
        imm_variants = []
        for imm in _IMMARG_PROBE_VALUES:
            imm_text, why_imm = _probe_module(decl, datalayout, imm_int=imm)
            if not why_imm and imm_text != text:
                imm_path = ir_dir / f"{stem}.imm{imm}.ll"
                imm_path.write_text(imm_text)
                imm_variants.append((imm, imm_path))
        if imm_variants:
            imm_modules[key] = imm_variants

    # Stage B -- the arch axis, only for names that exist. A key stage A could
    # not measure is held out too: its probe would have to be built from the
    # hand-written row, which is the generator-4 construction that produced
    # false `arch_absent` cells.
    work = [
        (k, a) for k in keys if k not in absent and k not in unmeasured for a in arches
    ]

    def run(item: tuple[str, str]) -> tuple[str, str, str, str, int | None]:
        key, arch = item
        path, why = modules[key]
        if path is None:
            return key, arch, STATUS_PROBE_ERROR, why, None
        out = Path(ir_dir) / f"{path.stem}.{arch}.hsaco"
        status, evidence = _probe(clang, path, arch, out)
        # Rescue a suspected false negative: both diagnostics mean "this
        # operand had to be an immediate", so a literal re-probe that lowers
        # cleanly means our first probe was malformed, not the arch.
        #
        # Scoped to those two diagnostics: literals give the backend more
        # information, so a blanket retry could constant-fold a genuinely
        # unsupported intrinsic away and manufacture an `ok`.
        if _wants_literals(status, evidence) and key in literal_modules:
            # Try every literal before giving up -- a value out of range for
            # one operand kind is in range for another, and stopping at the
            # first non-OK answer would let an unlucky choice pass as the
            # arch's verdict.
            settled: tuple[str, str] | None = None
            last: tuple[str, str] | None = None
            for i, lit_path in enumerate(literal_modules[key]):
                lit_out = Path(ir_dir) / f"{path.stem}.lit{i}.{arch}.hsaco"
                lit_status, lit_evidence = _probe(clang, lit_path, arch, lit_out)
                if lit_status == STATUS_OK:
                    return key, arch, STATUS_OK, "", None
                last = (lit_status, lit_evidence)
                if settled is None and not _wants_literals(lit_status, lit_evidence):
                    settled = last
            # Every variant failed to legalise, and with no variable integer
            # operands left that is the target speaking, not our probe --
            # gfx942 takes this call with a literal size and gfx1201 does not.
            # The verifier's immarg complaint is excluded: surviving the sweep
            # means the non-immediate operand is not an integer, which is our
            # module's fault.
            if settled is None and last and last[0] == STATUS_ARCH_ABSENT:
                settled = last
            status, evidence = settled or (status, evidence)

        # Sweep the immediates. Only on a negative, and only on the ~1 key in
        # 6 whose declare has an integer immarg, so the common case still
        # costs one probe.
        #
        # These two statuses only: `arch_absent` (ISel failed, and the operand
        # value is one reason why) and `probe_error` (llvm23's verifier
        # rejects an illegal transfer size before codegen -- a malformed
        # module is what the sweep repairs). A crash, timeout or unsupported
        # target says nothing about the operand value.
        #
        # Any legal value wins: the artifact answers "can this target lower
        # this intrinsic", not "with the operand our probe happened to pick".
        if status in (STATUS_ARCH_ABSENT, STATUS_PROBE_ERROR) and key in imm_modules:
            settled_imm: tuple[str, str] | None = None
            for imm, imm_path in imm_modules[key]:
                imm_out = Path(ir_dir) / f"{path.stem}.imm{imm}.{arch}.hsaco"
                imm_status, imm_evidence = _probe(clang, imm_path, arch, imm_out)
                if imm_status == STATUS_OK:
                    return key, arch, STATUS_OK, "", imm
                if settled_imm is None and imm_status != STATUS_PROBE_ERROR:
                    settled_imm = (imm_status, imm_evidence)
            # Every candidate failed, so the operand value was not the
            # obstacle. Where we started from `arch_absent` the original
            # evidence already says so and is kept -- the cell should describe
            # the probe as posed by default.
            #
            # Where we started from `probe_error` it does not: a legal value
            # failed too, so leaving it would commit "the generator is broken"
            # for a target that simply cannot lower the intrinsic. The first
            # candidate with a real classification speaks instead.
            if status == STATUS_PROBE_ERROR and settled_imm is not None:
                status, evidence = settled_imm

        return key, arch, status, evidence, None

    # Threads, not processes: each unit of work is already its own subprocess.
    #
    # The cap sits well under the core count: each probe is a clang that
    # spawns ld.lld, so fan-out is ~2x the worker count and saturating a big
    # host hits RLIMIT_NPROC. That surfaces as probe_error, and a probe_error
    # that depends on machine load makes `--check` flaky.
    jobs = args.jobs or min(8, available_cpus())
    results: dict[str, dict[str, dict[str, str]]] = {k: {} for k in keys}

    def record(
        key: str, arch: str, status: str, evidence: str, imm: int | None = None
    ) -> None:
        cell: dict[str, object] = {"status": status, "verified_on": flavor}
        if evidence and status != STATUS_OK:
            cell["evidence"] = evidence
        # Only on a cell the sweep rescued, so its absence means "the default
        # 0 answered" -- the common case, not worth a field on 1100 cells.
        # Present, it names the value that made the intrinsic lower, which is
        # what distinguishes a first-try `ok` from a rescued one.
        if imm is not None:
            cell["probe_imm"] = imm
        results[key][arch] = cell

    for key in absent:
        for arch in arches:
            record(key, arch, STATUS_NAME_ABSENT, f"not an intrinsic in {flavor}")

    # The diagnostic `opt` actually produced, not a generic reason: "not an
    # intrinsic in llvm20" written over an out-of-memory reads exactly like a
    # real negative and cannot be falsified from inside the data.
    for key, (status, evidence) in unmeasured.items():
        for arch in arches:
            record(key, arch, status, evidence)

    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        for key, arch, status, evidence, imm in ex.map(run, work):
            record(key, arch, status, evidence, imm)
            if args.verbose and status != STATUS_OK:
                print(f"     {key:44s} {arch:14s} {status}")

    # Second pass, serial: unclassified and crashed probes get one more chance
    # with no contention. A real result is stable under retry, a resource
    # failure is not -- llvm.fabs.f32 on gfx11-generic aborted on the process
    # limit once in three runs, while permlane64 on the wave64 targets
    # reproduces serially. The retry separates the two.
    #
    # A timeout retries on the same budget: the point is to remove contention,
    # not to grant more time to a module this small.
    unstable = (STATUS_PROBE_ERROR, STATUS_TOOLCHAIN_CRASH, STATUS_TOOLCHAIN_TIMEOUT)
    retry = [(k, a) for k, a in work if results[k][a]["status"] in unstable]
    if retry:
        print(f"   retrying {len(retry)} unclassified/crashed probe(s) serially...")
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

    out = args.out or default_out(flavor)

    if args.check:
        if not out.is_file():
            # Not a failure: a host can only measure its own flavor, so
            # failing here would red every CI machine on a newer ROCm.
            print(
                f"\nUNVALIDATED: no committed column for {flavor} "
                f"({out.name}); nothing to check. Run without --check to add one."
            )
            return 0
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
