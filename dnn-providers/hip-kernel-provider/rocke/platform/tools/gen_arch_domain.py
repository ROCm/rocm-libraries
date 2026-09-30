#!/usr/bin/env python3
"""Generate the intrinsic availability table (the "arch domain") by probing the
installed toolchain.

# Why this exists

rocke resolves an intrinsic `declare` on ONE axis: the LLVM flavor. The target
arch is consumed only to pick an ISA backend (`lower_llvm.py`, `backend_for(arch
or "gfx950")`) and never reaches the decl table -- so "is this intrinsic
available on this GPU" is checked nowhere at build time. See
`dsl_docs/development/arch_axis_proposal.md`.

This tool measures the missing axis instead of hand-maintaining it, and commits
the result as a data file. Nothing consumes the artifact yet; landing the data
first is deliberate (it cannot break anything, and it surfaces the defects that
justify the rest).

# Two stages, because the two axes are answered by different tools

Stage A -- the flavor axis -- asks whether this LLVM knows the name at all. It
is arch-free, so it runs once per key rather than once per (key, arch), and it
uses `opt -S`: LLVM resolves a recognised `llvm.*` declare on parse, attaching
the intrinsic's attributes and remangling overloads, while an unrecognised name
round-trips verbatim as an ordinary external function. See `_name_exists`.

Stage B -- the arch axis -- compiles AND LINKS a probe module per (key, arch),
for names that survived stage A. It is a link for the same reason
`check_ir_validity.py` is: a `declare` for a nonexistent intrinsic is accepted
by `opt -passes=verify` AND by `clang -S` -- to the backend it is an ordinary
external call, emitted as a GOT-relative `s_swappc_b64` -- and only the link
forces the symbol to resolve. It runs in a subprocess because an intrinsic that
exists but is unsupported on the target reaches `report_fatal_error`, which
kills the process.

The split is what makes this cheap: we do not need a hand-written compatibility
matrix, we can read each answer off its own oracle.

    stage A resolves      -> continue to stage B
    stage A verbatim      -> "name_absent"   this SPELLING is not an intrinsic
                                             in this LLVM (flavor axis)
    link OK               -> "ok"            available here
    Cannot select / fatal -> "arch_absent"   the name is real, this target
                                             cannot lower it (arch axis)

Three more buckets exist so that a non-answer is never recorded as a negative:

    invalid target ID     -> "target_unsupported"  this clang cannot target this
                                                   arch at all; it has no opinion
    clang crashed         -> "toolchain_crash"     asking the question killed the
                                                   compiler; we did not get an
                                                   answer, only a bug report
    anything else         -> "probe_error"         OUR module was malformed

`name_absent` deserves care when reading results: it means the exact mangled
string rocke emits is not a known intrinsic. A genuinely nonexistent operation
and a merely mis-mangled overload suffix are indistinguishable here -- both are
bugs in the decl table, but they are different bugs.

# Provenance is not optional

A result is only meaningful against the toolchain that produced it, and a host
can only ever validate its own flavor. The artifact therefore records the clang
identity and the flavor rocke resolved, and the generator refuses to run when
those two disagree -- an artifact labelled with the wrong LLVM vintage is worse
than no artifact, because it looks authoritative.

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

from _hostcaps import available_cpus

HERE = Path(__file__).resolve().parent
ROCKE = HERE.parent  # tools -> rocke/platform

SCHEMA = "rocke.intrinsic_arch_domain/v1"
DATA_DIR = ROCKE / "python" / "rocke" / "core" / "arch" / "data"


def default_out(flavor: str) -> Path:
    """One artifact per flavor, named after it.

    A single shared file would be wrong, not merely inconvenient: a host can
    only ever measure its own LLVM, so whichever flavor ran last would silently
    overwrite the others and `--check` would fail on every machine whose
    toolchain differs from the one that blessed the file. Naming the flavor
    makes each column independently ownable and independently checkable.
    """
    return DATA_DIR / f"intrinsic_arch_domain.{flavor}.json"


STATUS_OK = "ok"
STATUS_NAME_ABSENT = "name_absent"
STATUS_ARCH_ABSENT = "arch_absent"
STATUS_TARGET_UNSUPPORTED = "target_unsupported"
STATUS_TOOLCHAIN_CRASH = "toolchain_crash"
STATUS_TOOLCHAIN_TIMEOUT = "toolchain_timeout"
STATUS_PROBE_ERROR = "probe_error"

# Bumped when the probe *semantics* change -- not when this file is merely
# edited -- and recorded into every column so a committed answer can be told
# apart from one an older generator measured.
#
# The need is not hypothetical. The llvm20 column blessed before `-O0` became
# mandatory records `ok` for six arches on
# `llvm.amdgcn.raw.ptr.buffer.load.async.lds`, and the current generator cannot
# reproduce that on the same toolchain build because the probe does not
# terminate at all: at -O3 the call was deleted before it could hang, and the
# empty module linked. Nothing in the artifact said which generator wrote it,
# so the only way to find that out was to re-measure and notice the
# contradiction.
#
#   1 -- initial, probes at -O3
#   2 -- probes at -O0, per-probe timeout, provenance recorded
#   3 -- immarg values swept on a negative, winning value recorded
#   4 -- llvm23 diagnostics understood; the sweep also runs on probe_error
GENERATOR = 4

# Hoisted out of `_probe` so the artifact can record the flags that actually
# ran rather than a hand-copied list that drifts from them. `-O0` is the one
# that decides correctness (see `_probe`), which is exactly why a column must
# carry proof it was used.
PROBE_CFLAGS = ("-O0", "-nogpulib")

# A probe that has not answered in this long is not going to. A probe module is
# a single declare and a single call, and the normal cost is tens of
# milliseconds, so this is roughly three orders of magnitude of headroom.
#
# The budget exists to bound the *sweep*, not to hurry any one probe. Without
# it a single non-terminating probe hangs the whole run, `--check` never
# returns, and the drift gate's own subprocess timeout fires 30 minutes later
# as an error rather than a verdict -- one unlucky key takes down the gate for
# every host running that flavor.
PROBE_TIMEOUT_S = 60


def _bootstrap_sys_path() -> None:
    """Import rocke from the checkout without an external PYTHONPATH, matching
    tests/conftest.py. Unlike check_ir_validity this needs no library/ reach --
    the decl table lives entirely in platform."""
    path = ROCKE / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# --------------------------------------------------------------------------
# Probe module synthesis
# --------------------------------------------------------------------------

# Parameter attributes that may appear between the type and the comma. They are
# legal on a declare but carry no meaning for us, and `immarg` is the only one
# that changes what we must pass at the call site.
_PARAM_ATTRS = ("nocapture", "readnone", "readonly", "writeonly", "immarg")


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
    """Strip parameter attributes, returning (type, is_immarg)."""
    is_imm = "immarg" in param
    ty = param
    for attr in _PARAM_ATTRS:
        ty = ty.replace(attr, "")
    return " ".join(ty.split()), is_imm


_DECL_RE = re.compile(r"^declare\s+(.+?)\s+@([\w.]+)\((.*)\)\s*$")

# `Segmentation fault (core dumped)` vs `Segmentation fault` is a property of
# the host's core-dump settings, not of the crash. See `_crash_reason`.
_CORE_DUMPED_RE = re.compile(r"\s*\(core dumped\)\s*$")

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
# Chosen to cover the operand kinds that actually appear in this decl table
# rather than to be exhaustive: these are exactly the byte counts a
# load-to-LDS transfer size accepts, as llvm23's verifier enumerates them. A
# selector operand (an MFMA cbsz/blgp, say) is legal at 0 and so never reaches
# the sweep at all. The list is ordered smallest-first so the recorded winner
# is the least surprising legal value rather than whichever we happened to try
# first.
_IMMARG_PROBE_VALUES = (1, 2, 4, 12, 16)

# The ways LLVM reports "that operand had to be an immediate, and not that
# one". llvm22 rejects the call in the verifier; llvm20 has no such check and
# instead dies during type legalisation, which is why the same decl-table gap
# reads as a frontend error on one vintage and a backend error on the other.
#
# llvm23 added the third: a verifier check on the load-to-LDS transfer size
# that names the legal values outright. It is the same complaint as the first
# two -- the operand we synthesised is not an acceptable constant -- and it is
# the only one of the three that says what would have been acceptable.
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

    `literal_ints`, when set, replaces every integer kernel argument with that
    literal value. Some operands must be immediates even though the declare
    does not mark them `immarg`: rocke's decl table is hand-written and records
    `immarg` only where an author happened to add it, while LLVM checks the
    real intrinsic signature. `raw.ptr.buffer.load.lds`'s size operand is the
    example, and the two vintages report the mismatch differently -- llvm20
    fails to legalise (on *every* arch, reading as a universal arch_absent when
    the truth is "CDNA yes, RDNA no"), llvm22 rejects it up front with "immarg
    operand has non-immediate parameter". This variant exists to rescue both,
    and only those two diagnostics.

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
    ret = ret.strip()

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


def _name_exists(opt: str, decl: str, scratch: Path) -> tuple[bool, str]:
    """Ask this LLVM whether it recognises the declare's name as an intrinsic.

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
    """
    src = scratch / "name.ll"
    src.write_text(decl.strip() + "\n", encoding="utf-8")
    proc = subprocess.run(
        [opt, "-S", "-o", "-", str(src)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False, f"opt rejected the declare: {_first_error(proc.stderr)}"
    for line in proc.stdout.splitlines():
        if line.startswith("declare "):
            attributed = re.search(r"#\d+\s*$", line) is not None
            m = _DECL_RE.match(re.sub(r"\s*#\d+\s*$", "", line))
            canonical = m.group(2) if m else ""
            if attributed:
                return True, canonical
            # Remangled but attribute-free: still a resolved intrinsic.
            original = _DECL_RE.match(decl.strip())
            if original and canonical and canonical != original.group(2):
                return True, canonical
            return False, ""
    # The declare did not survive the round-trip at all. An unrecognised
    # `llvm.*` name is an ordinary external function and is printed back
    # verbatim even when unused, so a vanished declare means AutoUpgrade
    # consumed it -- `amdgcn.global.atomic.fadd` becoming a plain `atomicrmw`
    # is the live example. That still links, so the name counts as present.
    return True, "(auto-upgraded)"


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
    # Three ways the backend says "this target cannot lower that": instruction
    # selection has no pattern, type legalisation cannot break the operand
    # down, and the generic lowering wants a runtime libcall the device does
    # not have. All three are reached only *after* the name resolved, so the
    # flavor axis is already settled by _name_exists() and cannot be confused
    # with them here.
    # A fourth, added in llvm23: the backend says so in words rather than by
    # failing. It is the clearest of the four and the only one that cannot be
    # confused with a malformed probe, so it is worth matching on its own
    # rather than waiting for the selection failure it replaces.
    if (
        "cannot select" in low
        or "do not know how to expand this operator's operand" in low
        or "no libcall available for" in low
        or "intrinsic not supported on subtarget" in low
    ):
        return STATUS_ARCH_ABSENT, _first_error(diag)
    # The compiler died rather than answered. This is NOT arch_absent: we did
    # not learn that the target cannot lower the intrinsic, only that asking
    # crashes LLVM. It gets its own status because folding it into either
    # neighbour loses a finding -- `permlane64` on the wave64 targets segfaults
    # instruction selection on llvm20 -- and because a crash is reproducible,
    # so it stays stable under `--check` the way a resource failure would not.
    if "please submit a bug report" in low or "stack dump:" in low:
        return STATUS_TOOLCHAIN_CRASH, _crash_reason(diag)
    # Unrecognised failure. Attributing it to the arch would be a guess, and a
    # wrong guess here silently bakes a false negative into the artifact.
    return STATUS_PROBE_ERROR, _first_error(diag)


def _crash_reason(diag: str) -> str:
    """Summarise a clang crash without embedding host-specific paths.

    The crash dump leads with the full `-cc1` command line, which carries temp
    directories and the resource-dir path. Committing that would make the
    artifact differ between hosts for no informational gain, so keep only the
    pass name and the signal.

    The signal name needs the same treatment for a less obvious reason: the
    driver appends `(core dumped)` only when the kernel actually wrote one,
    which depends on the host's `ulimit -c` and core_pattern rather than on
    anything about the compiler. Leaving it in makes the committed evidence
    differ between two machines that observed the identical crash, so `--check`
    reports drift where there is none -- the one failure mode a drift gate
    cannot afford.
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


def _first_error(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if "error" in low or "undefined symbol" in low or "cannot select" in low:
            return s[:200]
    return (text.strip().splitlines() or ["(no diagnostic)"])[0][:200]


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
    reported ok for a target where the instruction does not exist. Nine of the
    149 keys are foldable this way. -O0 keeps the call alive to ISel, which is
    the only place that can answer the question.

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
    return _classify(proc.returncode, (proc.stderr or proc.stdout).strip())


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


def _clang_identity(clang: str) -> str:
    try:
        proc = subprocess.run(
            [clang, "--version"], capture_output=True, text=True, check=False
        )
        return (proc.stdout or "").strip().splitlines()[0][:200]
    except (OSError, IndexError):
        return "(unknown)"


def _flavor_of_clang(identity: str) -> str | None:
    """Best-effort LLVM major from `clang --version`, as a flavor string."""
    m = re.search(r"clang version (\d+)", identity)
    return f"llvm{m.group(1)}" if m else None


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
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    _bootstrap_sys_path()
    sys.path.insert(0, str(HERE))
    from check_ir_validity import _llvm_tool  # same resolution order, one owner
    from rocke.core import lower_llvm as L
    from rocke.core.isa.backend import wired_arches

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

    identity = _clang_identity(clang)
    print(f"   version: {identity}")

    # A probe result is only meaningful against the flavor it was measured on.
    # Recording llvm20 results in an artifact stamped llvm23 would be actively
    # misleading, so disagreement is fatal rather than a warning.
    clang_flavor = _flavor_of_clang(identity)
    if clang_flavor and clang_flavor != flavor:
        print(
            f"\nERROR: clang reports {clang_flavor} but rocke resolved {flavor}. "
            "The probe would be attributed to the wrong LLVM vintage. "
            "Set ROCKE_LLVM_BIN to the matching toolchain, or ROCKE_LLVM_FLAVOR "
            "if the override is intended."
        )
        return 2

    arches = sorted(args.arch) if args.arch else sorted(wired_arches())
    decls = _decl_table(flavor)
    keys = sorted(k for k in decls if not args.only or args.only in k)
    print(f"   arches : {', '.join(arches)}")
    print(f"   keys   : {len(keys)}")

    tmp = tempfile.TemporaryDirectory(prefix="rocke_arch_domain_")
    ir_dir = args.keep_ir if args.keep_ir else Path(tmp.name)
    ir_dir.mkdir(parents=True, exist_ok=True)

    datalayout = L._datalayout_for_flavor(flavor)

    # Build every probe module first; an unparseable declare is a probe_error
    # for every arch rather than a crash mid-sweep.
    modules: dict[str, tuple[Path | None, str]] = {}
    literal_modules: dict[str, list[Path]] = {}
    imm_modules: dict[str, list[tuple[int, Path]]] = {}
    for key in keys:
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
        text, why = _probe_module(decls[key], datalayout)
        if why:
            modules[key] = (None, why)
            continue
        path = ir_dir / f"{stem}.ll"
        path.write_text(text)
        modules[key] = (path, "")
        # Two literal values, tried in order. Neither is safe alone: 0 is
        # rejected where the operand is a byte count, and a nonzero value is
        # out of range where the operand is a small enum (an MFMA cbsz/blgp
        # selector, say). Trying both keeps the rescue from depending on which
        # kind of operand we happened to hit.
        variants = []
        for lit in _LITERAL_PROBE_VALUES:
            lit_text, why_lit = _probe_module(decls[key], datalayout, literal_ints=lit)
            if not why_lit and lit_text != text:
                lit_path = ir_dir / f"{stem}.lit{lit}.ll"
                lit_path.write_text(lit_text)
                variants.append(lit_path)
        if variants:
            literal_modules[key] = variants
        # The immarg sweep, built the same way. A declare with no integer
        # immarg produces a module identical to the base one, so the `!=`
        # test below is what decides which keys are sweepable -- no separate
        # signature analysis to keep in step with `_probe_module`.
        imm_variants = []
        for imm in _IMMARG_PROBE_VALUES:
            imm_text, why_imm = _probe_module(decls[key], datalayout, imm_int=imm)
            if not why_imm and imm_text != text:
                imm_path = ir_dir / f"{stem}.imm{imm}.ll"
                imm_path.write_text(imm_text)
                imm_variants.append((imm, imm_path))
        if imm_variants:
            imm_modules[key] = imm_variants

    # Stage A -- the flavor axis, once per key. Arch-free, so it costs one
    # `opt` run instead of one link per arch, and it is the only stage that can
    # answer a key whose codegen crashes.
    canonical: dict[str, str] = {}
    absent: set[str] = set()
    for key in keys:
        exists, note = (
            _name_exists(opt, decls[key], Path(ir_dir)) if opt else (True, "")
        )
        if exists:
            canonical[key] = note
        else:
            absent.add(key)
    print(
        f"   names  : {len(keys) - len(absent)} present, {len(absent)} absent in {flavor}"
    )

    # Stage B -- the arch axis, only for names that exist.
    work = [(k, a) for k in keys if k not in absent for a in arches]

    def run(item: tuple[str, str]) -> tuple[str, str, str, str, int | None]:
        key, arch = item
        path, why = modules[key]
        if path is None:
            return key, arch, STATUS_PROBE_ERROR, why, None
        out = Path(ir_dir) / f"{path.stem}.{arch}.hsaco"
        status, evidence = _probe(clang, path, arch, out)
        # Rescue a suspected false negative. Both diagnostics below are the
        # signature of "this operand had to be an immediate" -- llvm22 says so
        # outright, llvm20 only fails to legalise -- so a literal-operand
        # re-probe that lowers cleanly means the arch does support the
        # intrinsic and our first probe was simply malformed.
        #
        # Scoped to those two diagnostics on purpose: literals give the backend
        # strictly more information, so a blanket retry could constant-fold a
        # genuinely unsupported intrinsic away and manufacture an `ok`.
        if _wants_literals(status, evidence) and key in literal_modules:
            # Try every literal value before giving up: a value that is out of
            # range for one operand kind is in range for another, and stopping
            # at the first non-OK answer would let an unlucky first choice
            # masquerade as the arch's verdict.
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
            # Every variant still failed to legalise -- but with no variable
            # integer operands left, that is no longer something our probe can
            # fix, so it is the target speaking. gfx942 takes this call with a
            # literal size and gfx1201 does not, on both llvm20 and llvm22.
            # The verifier's immarg complaint is excluded on purpose: surviving
            # the literal sweep means the non-immediate operand is not an
            # integer, which really is our module's fault.
            if settled is None and last and last[0] == STATUS_ARCH_ABSENT:
                settled = last
            status, evidence = settled or (status, evidence)

        # Sweep the immediates. Runs only on a negative, so the common case
        # still costs one probe -- and only on the ~1 key in 6 whose declare
        # has an integer immarg at all.
        #
        # Two statuses, and only these two. `arch_absent` is the one the sweep
        # was built for: the backend could not select, and the operand value is
        # one reason why. `probe_error` is the same defect caught earlier --
        # llvm23's verifier rejects an illegal transfer size before codegen
        # runs, and "our module was malformed" is exactly what the sweep
        # repairs. A crash, a timeout or an unsupported target say nothing
        # about the operand value, and re-asking would just pay for the same
        # answer five more times.
        #
        # Any legal value wins, because the question the artifact answers is
        # "can this target lower this intrinsic", not "can it lower it with
        # the operand our probe happened to pick". A 0 that is out of range is
        # our defect, not the target's limit.
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
            # Where we started from `probe_error` it does not. "Our module was
            # malformed" was true of the default probe and is now known not to
            # be the whole story, because a legal value fails too. Leaving it
            # would commit a `probe_error` -- the one status that means the
            # generator is broken -- for a target that simply cannot lower the
            # intrinsic. So the first candidate that produced a real
            # classification speaks instead.
            if status == STATUS_PROBE_ERROR and settled_imm is not None:
                status, evidence = settled_imm

        return key, arch, status, evidence, None

    # Threads, not processes: each unit of work is already its own subprocess.
    #
    # The cap is deliberately well under the core count. Each probe is a clang
    # that itself spawns ld.lld, so the process fan-out is ~2x the worker count
    # and saturating a big host hits RLIMIT_NPROC -- observed as `posix_spawn
    # failed: Resource temporarily unavailable` and as std::system_error from
    # clang's own thread pool. Those surface as probe_error, and a probe_error
    # that depends on machine load would make the artifact nondeterministic and
    # `--check` flaky. Staying cheap is worth more here than being fast.
    jobs = args.jobs or min(8, available_cpus())
    results: dict[str, dict[str, dict[str, str]]] = {k: {} for k in keys}

    def record(
        key: str, arch: str, status: str, evidence: str, imm: int | None = None
    ) -> None:
        cell: dict[str, object] = {"status": status, "verified_on": flavor}
        if evidence and status != STATUS_OK:
            cell["evidence"] = evidence
        # Only on a cell the sweep rescued. Its absence therefore means "the
        # default 0 answered", which is the common case and should not cost a
        # field on 1100 cells. Present, it says which value made the intrinsic
        # lower -- without it nobody can tell a first-try `ok` from one that
        # needed the fourth candidate, and re-deriving that by hand is how the
        # defect it exists to prevent stayed invisible for two columns.
        if imm is not None:
            cell["probe_imm"] = imm
        results[key][arch] = cell

    for key in absent:
        for arch in arches:
            record(key, arch, STATUS_NAME_ABSENT, f"not an intrinsic in {flavor}")

    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        for key, arch, status, evidence, imm in ex.map(run, work):
            record(key, arch, status, evidence, imm)
            if args.verbose and status != STATUS_OK:
                print(f"     {key:44s} {arch:14s} {status}")

    # Second pass, serial: anything that failed in a way we could not classify,
    # and anything that took the toolchain down with it, gets one more chance
    # with no contention. A real result is stable under retry; a resource
    # failure is not, and under -j the probes fork enough linkers to hit the
    # process limit occasionally -- llvm.fabs.f32 on gfx11-generic aborted that
    # way once in three runs and would otherwise have been recorded as a
    # permanent compiler crash. A genuine crash (permlane64 on the wave64
    # targets) reproduces serially, so the retry separates the two. Cells that
    # survive stay as they were, which is the honest answer.
    #
    # A timeout retries on the same budget rather than a larger one. The retry
    # is there to remove contention, not to grant more time: if 60 uncontended
    # seconds are not enough for a module this small, more seconds will not
    # change the verdict, and raising the budget only to watch a known hang
    # spin costs the whole sweep. A probe that was merely starved by a loaded
    # host clears the same bar easily once it has the machine to itself.
    unstable = (STATUS_PROBE_ERROR, STATUS_TOOLCHAIN_CRASH, STATUS_TOOLCHAIN_TIMEOUT)
    retry = [(k, a) for k, a in work if results[k][a]["status"] in unstable]
    if retry:
        print(f"   retrying {len(retry)} unclassified/crashed probe(s) serially...")
        for key, arch in retry:
            _k, _a, status, evidence, imm = run((key, arch))
            record(key, arch, status, evidence, imm)

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
        # Provenance. `clang` alone identifies the compiler but not how it was
        # asked, and the two questions have different answers: the same
        # toolchain build reports `ok` or hangs for the same key depending on
        # the optimisation level the probe used. A column that does not carry
        # these fields was written by generator 1 and cannot be trusted where
        # it says `ok`.
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
        # to the declared name; the interesting rows are the overloaded
        # intrinsics that remangle (`ds.read.tr16.b64` -> `...b64.v4i16`) and
        # the legacy ones AutoUpgrade rewrites, because both are places where
        # what rocke emits and what the toolchain executes differ.
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
            # Not a failure. A host can only measure its own flavor, so a
            # flavor nobody has blessed yet simply has no column to check --
            # failing here would red every CI machine running a newer ROCm.
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
            # flavor. That is a pass, and a stronger one than a byte match:
            # two independent compiler builds agreeing is evidence the column
            # describes the flavor rather than one machine. See `_drift`.
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
