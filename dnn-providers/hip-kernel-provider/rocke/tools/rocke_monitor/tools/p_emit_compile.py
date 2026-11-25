# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Generic: emit every config of the given parity emitters (python <family>_emit.py <idx> ll), compile each emitted .ll
# with ROCm clang 24 (-O3) for its target, and report compile failures, spills and VGPRs.
#
# Target: the IR's "target-cpu" attribute or, when the IR has none, the arch the emitter's run_emit() resolves for the
# config (a shim reports it). A config whose target is still unknown is counted and not compiled; it is never guessed.
# End of configs: the emitter's last error line says "unknown config" (or is an IndexError). Only that last line
# counts, because tracebacks quote source lines such as usage strings ("<config_index 0..10>").
# Rejections: with --base-root (the same tree at develop), a config the PR rejects but develop builds is a NEW
# rejection and fails like a compile failure. A config develop rejects too is reported, but is not the PR's doing.
# A family the PR adds has no develop emitter, so its rejections have no develop baseline.
# Output that is not LLVM IR (conv_abi prints ABI lists) is counted and not compiled. One emitted file may hold
# several modules, and each is compiled on its own (gfx950_attention_tiled_3d prints a segment and a reduce kernel).
#
# Usage: python p_emit_compile.py --root <rocke root> [--base-root <rocke root at develop>] <parity dir>:<family> ...
import argparse, json, os, re, subprocess, sys, tempfile
from pathlib import Path


def _configured(key):
    """A path from the monitor's configuration (rkmon_lib.paths), for a run by hand outside a job."""
    sys.dont_write_bytecode = True  # the toolkit's folder is read-only at run time
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rkmon_lib import config, paths

    try:
        return paths.resolve(config.load(check=False)).get(key) or ""
    except config.ConfigError:
        return ""


# The job exports the LLVM tools' folder (rkmon_lib.config.env); by hand it comes from the configuration. Scratch files go
# to RKMON_TMP, else TMPDIR (in a PR runner's slot, the slot's own), else the configured temporary folder.
BIN = os.environ.get("RKMON_LLVM_BIN") or _configured("rocm_llvm_bin")
TMP = (
    os.environ.get("RKMON_TMP")
    or os.environ.get("TMPDIR")
    or _configured("tmp")
    or tempfile.gettempdir()
)
# A safety cap, not the end of configs: a family ends at its emitter's "unknown config" line, and reaching the cap is
# reported (a fixed range would skip configs past its end in silence).
MAX_CONFIGS = int(os.environ.get("RKMON_MAX_CONFIGS", "5000"))

# Runs the emitter as __main__, after wrapping _emit_common.run_emit so it reports the arch it resolves. Emitters that
# lower directly (conv_direct_grouped_emit.py calls _lower_kernel_to_llvm_python(kernel, arch=...)) report the arch
# they lower for through wrapped lowering functions: their IR names no target, and they would go uncompiled as
# "unknown target".
# Each kernel built or lowered also reports rocKE's own codegen options for it (the tree's
# helpers.compile._comgr_options_for_kernel, as RKMON_OPTS=<json list>): compiled plain -O3, AGPR bodies, which rocKE
# compiles with -mllvm -amdgpu-mfma-vgpr-form=0, read as false spills. The lowering functions are
# wrapped before _emit_common is imported, so a name it imports from them is the wrapped one.
SHIM = r"""
import inspect, json, os, runpy, sys
path = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
_seen = set()
def _report_opts(k):
    if not hasattr(k, "attrs") or id(k) in _seen:
        return
    _seen.add(id(k))
    try:
        from rocke.helpers.compile import _comgr_options_for_kernel as _opts
        o = [str(x) for x in _opts(k) if x != "-O3"]
    except Exception:
        return
    sys.stderr.write("RKMON_OPTS=%s\n" % json.dumps(o))
def _reporting(fn):
    def lower(kernel, *a, **kw):
        arch = kw.get("arch") or (a[0] if a and isinstance(a[0], str) else None)
        if arch:
            sys.stderr.write("RKMON_ARCH=%s\n" % arch)
        _report_opts(kernel)
        return fn(kernel, *a, **kw)
    return lower
for _mod, _name in (("rocke.core.lower_llvm", "_lower_kernel_to_llvm_python"), ("rocke", "lower_kernel_to_llvm")):
    try:
        import importlib
        _m = importlib.import_module(_mod)
        if hasattr(_m, _name):
            setattr(_m, _name, _reporting(getattr(_m, _name)))
    except Exception:
        pass
try:
    import _emit_common as E
except Exception:
    E = None
if E is not None and hasattr(E, "run_emit"):
    _orig = E.run_emit
    try:
        _default = inspect.signature(_orig).parameters["arch"].default
    except Exception:
        _default = None
    def run_emit(spec_fn, build_fn, *a, **kw):
        try:
            sel = spec_fn(int(sys.argv[1]))
            arch = sel[1] if isinstance(sel, tuple) else kw.get("arch", _default)
            if arch:
                sys.stderr.write("RKMON_ARCH=%s\n" % arch)
        except BaseException:
            pass
        def build(*ba, **bkw):
            out = build_fn(*ba, **bkw)
            for k in (out if isinstance(out, (list, tuple)) else (out,)):
                _report_opts(k)
            return out
        return _orig(spec_fn, build, *a, **kw)
    E.run_emit = run_emit
runpy.run_path(path, run_name="__main__")
"""

OPTS = re.compile(r"^RKMON_OPTS=(.*)$", re.M)


def module_options(err: str, n: int) -> list[list[str]]:
    """rocKE's codegen options for each of the n modules a config emits, from the shim's RKMON_OPTS lines: one set for
    every module when the kernels agree, else one per module in emission order, and none when the counts differ.
    """
    sets = []
    for m in OPTS.finditer(err):
        try:
            v = json.loads(m.group(1))
        except ValueError:
            continue
        if isinstance(v, list):
            sets.append([str(x) for x in v])
    if sets and all(s == sets[0] for s in sets):
        return [sets[0]] * n
    return sets if len(sets) == n else [[]] * n


def emit(
    root: Path, pdir: str, fam: str, idx: int, flavor: str | None = None
) -> tuple[int, str, str]:
    d = root / pdir
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [str(root / "platform/python"), str(root / "library"), str(d)]
        ),
    )
    if flavor:
        env["ROCKE_LLVM_FLAVOR"] = flavor
    p = subprocess.run(
        [sys.executable, "-B", "-c", SHIM, str(d / f"{fam}_emit.py"), str(idx), "ll"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(d),
        timeout=600,
    )
    return p.returncode, p.stdout, p.stderr


FLAVOR_NEEDED = re.compile(r"requires LLVM flavor (llvm\d+)")


def emit_any(root: Path, pdir: str, fam: str, idx: int) -> tuple[int, str, str]:
    """Emit at the ambient flavor, and retry a config that needs another one (gfx1250 tensor loads need llvm23) at that
    flavor. Otherwise every config of such a family counts as rejected, on develop too, and the family passes with
    nothing compiled."""
    rc, ll, err = emit(root, pdir, fam, idx)
    if (rc != 0 or not ll.strip()) and not is_end(err):
        m = FLAVOR_NEEDED.search(err)
        if m and m.group(1) != os.environ.get("ROCKE_LLVM_FLAVOR"):
            rc, ll, err = emit(root, pdir, fam, idx, m.group(1))
    return rc, ll, err


def last_line(err: str) -> str:
    lines = [
        ln.strip()
        for ln in err.splitlines()
        if ln.strip() and not ln.startswith(("RKMON_ARCH=", "RKMON_OPTS="))
    ]
    return lines[-1] if lines else ""


def is_end(err: str) -> bool:
    ln = last_line(err)
    return "unknown config" in ln.lower() or bool(
        re.match(r"IndexError: .*index out of range", ln)
    )


def target_of(ll: str, err: str) -> str | None:
    # Generic targets have hyphens (gfx11-generic, gfx10-3-generic).
    m = re.search(r'"target-cpu"="(gfx[0-9a-z-]+)"', ll) or re.search(
        r"^RKMON_ARCH=(gfx[0-9a-z-]+)$", err, re.M
    )
    return m.group(1) if m else None


HEADER = re.compile(r"(; ModuleID|source_filename|target datalayout|target triple)")


def modules(ll: str) -> list[str]:
    """Split emitted text into modules: a header line after a module's body starts the next one. clang reads one
    module per file, and a family that emits two kernels per config (a reduce kernel after the main one) would fail
    "expected top-level entity" at the second `target datalayout`."""
    out, cur, body = [], [], False
    for ln in ll.splitlines(keepends=True):
        if HEADER.match(ln):
            if body:
                out.append("".join(cur))
                cur, body = [], False
        elif ln.strip() and not ln.lstrip().startswith(";"):
            body = True
        cur.append(ln)
    if cur:
        out.append("".join(cur))
    return out


def compile_all(ll: str, cpu: str, err: str) -> list[subprocess.CompletedProcess]:
    """Compile each module of a config's IR on its own, with rocKE's codegen options for its kernel."""
    mods = modules(ll)
    return [compile_ll(m, cpu, o) for m, o in zip(mods, module_options(err, len(mods)))]


def compile_ll(ll: str, cpu: str, opts: list[str] = ()) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory(dir=TMP if os.path.isdir(TMP) else None) as td:
        src, obj = Path(td) / "k.ll", Path(td) / "k.o"
        src.write_text(ll)
        return subprocess.run(
            [
                f"{BIN}/clang",
                "-x",
                "ir",
                "-O3",
                *opts,
                "-nogpulib",
                "-target",
                "amdgcn-amd-amdhsa",
                f"-mcpu={cpu}",
                "-c",
                "-o",
                str(obj),
                str(src),
                "-Rpass-analysis=kernel-resource-usage",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )


def fails_on_base(base: Path, pdir: str, fam: str, idx: int) -> bool:
    """Does develop emit this config and fail to compile it too? Then the failure is not the PR's (a config that needs
    more LDS than the arch has fails on develop too, and must not fail G7 for every PR touching that family).
    """
    if not (base / pdir / f"{fam}_emit.py").exists():
        return False
    rc, ll, err = emit_any(base, pdir, fam, idx)
    if rc != 0 or not re.search(r"^define ", ll, re.M):
        return False
    cpu = target_of(ll, err)
    return bool(cpu) and any(r.returncode != 0 for r in compile_all(ll, cpu, err))


def scratch(r: subprocess.CompletedProcess) -> int:
    return int(
        (re.search(r"ScratchSize \[bytes/lane\]: (\d+)", r.stderr) or [None, -1])[1]
    )


def spills_on_base(base: Path, pdir: str, fam: str, idx: int) -> bool:
    """Does develop emit this config and spill it too? Then the spill is not the PR's."""
    if not (base / pdir / f"{fam}_emit.py").exists():
        return False
    rc, ll, err = emit_any(base, pdir, fam, idx)
    if rc != 0 or not re.search(r"^define ", ll, re.M):
        return False
    cpu = target_of(ll, err)
    res = compile_all(ll, cpu, err) if cpu else []
    return (
        bool(res)
        and all(r.returncode == 0 for r in res)
        and any(scratch(r) > 0 for r in res)
    )


def run_family(root: Path, base: Path | None, pdir: str, fam: str) -> None:
    ok = fail = old_fail = new_rej = old_rej = rej = unknown = noir = spills = 0
    vg, msgs, spilled, old_spill = [], [], [], []
    for idx in range(MAX_CONFIGS):
        rc, ll, err = emit_any(root, pdir, fam, idx)
        if rc != 0 or not ll.strip():
            if is_end(err):
                break
            why = last_line(err)[:110]
            if base is None or not (base / pdir / f"{fam}_emit.py").exists():
                rej += 1  # no develop tree, or a family the PR adds
                continue
            brc, bll, berr = emit_any(base, pdir, fam, idx)
            if brc == 0 and bll.strip():
                new_rej += 1
                msgs.append(f"cfg {idx}: NEW rejection (develop builds it): {why}")
            elif is_end(berr):
                rej += 1  # a config the PR adds: there is no develop baseline for it
            else:
                old_rej += 1
            continue
        if not re.search(r"^define ", ll, re.M):
            noir += 1  # not LLVM IR (an ABI list, for example): no "nothing compiled" failure
            continue
        cpu = target_of(ll, err)
        if not cpu:
            unknown += 1
            continue
        res = compile_all(ll, cpu, err)
        c = next((r for r in res if r.returncode != 0), None)
        if c is not None:
            why = (
                [ln for ln in c.stderr.splitlines() if "error" in ln.lower()] or ["?"]
            )[0][:120]
            if base is not None and fails_on_base(base, pdir, fam, idx):
                old_fail += 1
                msgs.append(f"cfg {idx} [{cpu}]: fails on develop too: {why}")
            else:
                fail += 1
                msgs.append(f"cfg {idx} [{cpu}]: {why}")
            continue
        ok += 1
        g = lambda r, pat: int((re.search(pat, r.stderr) or [None, -1])[1])
        if any(scratch(r) > 0 for r in res):
            spills += 1
            spilled.append(idx)
            if base is not None and spills_on_base(base, pdir, fam, idx):
                old_spill.append(idx)
        vg.append(max(g(r, r"VGPRs: (\d+)") for r in res))
    else:
        msgs.insert(
            0,
            f"stopped at the {MAX_CONFIGS}-config cap: later configs were not checked",
        )
    if noir and not (ok or fail or old_fail or new_rej or rej or unknown):
        # Not "0 compiled": the check script fails a family with nothing compiled, and this one has nothing to compile.
        # A config develop rejects too is no change of the PR's (serialized outputs and refusals are not "nothing
        # compiled").
        also = f"; {old_rej} rejected on develop too" if old_rej else ""
        print(
            f"{fam}: no LLVM IR to compile ({noir} configs emit something else{also})"
        )
        return
    vg.sort()
    parts = [f"{ok} compiled", f"{fail} compile failures"]
    if old_fail:
        parts.append(
            f"{old_fail} also fail to compile on develop"
        )  # not "compile failures": the gate ignores these
    parts += [f"{new_rej} newly rejected", f"{old_rej} rejected on develop too"]
    if rej:
        parts.append(
            f"{rej} rejected ({'no develop baseline' if base else 'not compared with develop'})"
        )
    if unknown:
        parts.append(f"{unknown} unknown target (not compiled)")
    if noir:
        parts.append(f"{noir} not LLVM IR (not compiled)")
    if spilled:
        # Which configs spill, and which of them spill on develop too (a spill develop shares is not the PR's).
        cfgs = ", ".join(map(str, spilled[:8])) + ("…" if len(spilled) > 8 else "")
        new = [i for i in spilled if i not in old_spill]
        also = (
            (
                f"; {len(old_spill)} also spill on develop"
                + (f", new: cfg {', '.join(map(str, new[:8]))}" if new else "")
            )
            if base is not None
            else ""
        )
        parts.append(f"spills {spills} (cfg {cfgs}{also})")
    else:
        parts.append("spills 0")
    if vg:
        parts.append(f"VGPRs min/med/max {vg[0]}/{vg[len(vg) // 2]}/{vg[-1]}")
    print(f"{fam}: " + ", ".join(parts))
    for m in msgs[:4]:
        print("   ", m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", required=True, type=Path, help="rocke root of the tree under test"
    )
    ap.add_argument(
        "--base-root", type=Path, default=None, help="the same rocke root at develop"
    )
    ap.add_argument(
        "families", nargs="+", help="<parity dir relative to the root>:<family name>"
    )
    args = ap.parse_args()
    for spec in args.families:
        pdir, fam = spec.rsplit(":", 1)
        run_family(args.root, args.base_root, pdir, fam)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
