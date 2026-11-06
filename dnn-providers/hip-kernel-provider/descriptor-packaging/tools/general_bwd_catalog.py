# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Write the IngestorGenerator config of the native engine ``hipkernel:AttentionGeneralBwd``.

The engine is slotted (RFC 0021): a single-launch ``prep`` slot and a multi-launch
``main`` slot whose kernels launch ``main``, ``dq``, ``convert_dk`` and ``convert_dv``
(``dq`` is the dQ convert, or the split-dQ kernel in ``main_splitdq_gfx950``). Each
pack serves one arch. The kernels are rocKE's catalog
(``dispatch/attention_bwd/hipdnn_catalog.py``): one main kernel per (class, tuned-table
key, head packing) with the convert / split-dQ kernels its plan launches, and every
distinct prep kernel. Each ``spec`` is the non-default fields of the plan's resolved
spec, rebuilt here through ``hkp_pack.rocke_compile.build_spec`` and the builder's
``supports_*`` predicate before it is written.

Usage::

    general_bwd_catalog.py [--out <IngestorGenerator>/configs/attention_general_bwd.yaml]
                           [--generate] [--verify] [--check]

``--generate`` also runs IngestorGenerator's ``generate.py`` on the config into a
scratch directory (ids reused from the installed bundle) and installs the descriptors
into ``--bundle``; ``--verify`` runs ``verify_variant_sets.py --mode structural`` on the
bundle. ``--check`` writes nothing: it fails when the committed config (and, with
``--generate``, a regenerated bundle) differs. Kernels are always lowered with the
Python backend (``ROCKE_BACKEND=python``).
"""

from __future__ import annotations

import argparse
import dataclasses
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_HKP = Path(__file__).resolve().parents[2]
for _sub in ("rocke/platform/python", "rocke/library", "descriptor-packaging/python"):
    _p = str(_HKP / _sub)
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
os.environ["ROCKE_BACKEND"] = "python"

import yaml  # noqa: E402
from dispatch.attention_bwd import hipdnn as H  # noqa: E402
from dispatch.attention_bwd import hipdnn_catalog as C  # noqa: E402
from hkp_pack.rocke_compile import build_spec  # noqa: E402
from kernels.common import attention_bwd as _main_mod  # noqa: E402
from kernels.common import attention_bwd_aux as _aux_mod  # noqa: E402
from kernels.common import attention_bwd_dq as _dq_mod  # noqa: E402

REPO = _HKP.parents[1]
GENERATOR_DIR = REPO / "projects" / "hipdnn" / "tools" / "IngestorGenerator"
DEFAULT_OUT = GENERATOR_DIR / "configs" / "attention_general_bwd.yaml"
AUTHORED_SUBPATH = "rocKE/attention_general_bwd"
DEFAULT_BUNDLE = (
    _HKP
    / "src"
    / "engines"
    / "kernel_ingestor_engine"
    / "descriptors"
    / AUTHORED_SUBPATH
)
TOOL = "dnn-providers/hip-kernel-provider/descriptor-packaging/tools/general_bwd_catalog.py"
AOT_BUDGET_PER_PACK = 1000

MAIN_SOURCE = "kernels/common/attention_bwd.py"
AUX_SOURCE = "kernels/common/attention_bwd_aux.py"
DQ_SOURCE = "kernels/common/attention_bwd_dq.py"
# builder -> (source, supports predicate, kernel-name stage)
BUILDERS = {
    "build_attn_bwd_main": (MAIN_SOURCE, _main_mod.supports_attn_bwd_main, "main"),
    "build_attn_bwd_prep": (AUX_SOURCE, _aux_mod.supports_attn_bwd_prep, "prep"),
    "build_attn_bwd_convert": (
        AUX_SOURCE,
        _aux_mod.supports_attn_bwd_convert,
        "convert",
    ),
    "build_attn_bwd_dq": (DQ_SOURCE, _dq_mod.supports_attn_bwd_dq, None),
}
MAIN_LAUNCHES = ("main", "dq", "convert_dk", "convert_dv")
# Fields every main and prep spec states, so a launch override stating them agrees.
STATED_FIELDS = ("dtype", "head_size", "seq_mode", "stage_vec")
# The KMD spells dtype as the hipDNN-facing name (the rocKE bundles' convention).
KMD_DTYPES = {"fp16": "FP16", "bf16": "BF16"}
# Spec knobs the engine never ships (no XCD hint reaches a hipDNN plan).
XCD_FIELDS = ("xcd_chunk", "dq_xcd_chunk")

# KMD fields per slot: (name, type, bound spec field or None for matcher-only)
PREP_FIELDS = (
    ("dtype", "string", "dtype"),
    ("head_size", "int", "head_size"),
    ("seq_mode", "string", "seq_mode"),
    ("stage_vec", "int", "stage_vec"),
    ("worklist", "bool", "worklist"),
    # resolved by attn_bwd_aux_rows_per_cta; the spec field stays None
    ("rows_per_cta", "int", None),
    ("narrow_vec", "int", "narrow_vec"),
)
MAIN_FIELDS = (
    ("dtype", "string", "dtype"),
    ("head_size", "int", "head_size"),
    ("seq_mode", "string", "seq_mode"),
    ("stage_vec", "int", "stage_vec"),
    ("dkv_mode", "string", "dkv_mode"),
    ("mask_class", "string", "mask_class"),
    ("dq_mode", "string", "dq_mode"),
    ("head_pack", "bool", "head_pack"),
    ("lb_mirror", "bool", "lb_mirror"),
    ("lb_interleave", "bool", "lb_interleave"),
    ("tuning_key", "string", None),
    ("main_block_n", "int", None),
    ("main_worklist", "bool", None),
    ("main_static_scale", "string", None),
)

HEADER = f"""\
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Generated by {TOOL}; do not edit.
#
# The descriptors of hipkernel:AttentionGeneralBwd under
# dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/{AUTHORED_SUBPATH}/
# are generated from this file. Regenerate both from the repository root with the rocKE
# library on the path:
#     python3 {TOOL} --generate --verify
# which writes this config, runs generate.py on it into a scratch directory (reusing the
# installed bundle's ids) with ROCKE_BACKEND=python, installs the descriptors and checks
# them with tools/verify_variant_sets.py --mode structural. `--check` instead fails when
# either is stale.
"""


class CatalogError(RuntimeError):
    pass


def _default(field: dataclasses.Field):
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return dataclasses.MISSING


def _json_value(value):
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if dataclasses.is_dataclass(value):
        raise CatalogError(f"nested spec dataclass {value!r} is not emitted")
    return value


def spec_dict(spec, stated=()) -> dict:
    """The fields of ``spec`` that differ from the dataclass defaults, plus
    ``stated``, in field order; tuples as lists."""
    out = {}
    for field in dataclasses.fields(spec):
        value = getattr(spec, field.name)
        if field.name in stated or value != _default(field):
            out[field.name] = _json_value(value)
    return out


def _self_check(builder: str, spec, emitted: dict, arch: str, kernel: str) -> None:
    """``emitted`` rebuilds ``spec`` through hkp_pack, the builder supports it on
    ``arch``, and its symbol is ``kernel``."""
    _source, supports, stage = BUILDERS[builder]
    rebuilt = build_spec(type(spec), emitted)
    if rebuilt != spec:
        raise CatalogError(f"{builder} spec does not round-trip for {kernel}")
    ok, reason = supports(rebuilt, arch=arch)
    if not ok:
        raise CatalogError(f"{builder} refuses {kernel} on {arch}: {reason}")
    name = rebuilt.kernel_name() if stage is None else rebuilt.kernel_name(stage)
    if name != kernel:
        raise CatalogError(f"{builder} names {name}, the plan launches {kernel}")


def _check_bindings(slot: str, fields, metadata: dict, specs: dict, where: str):
    """Every bound KMD value is concrete and agrees with every launch spec that
    carries the bound attribute (hkp_pack compares each reporting launch)."""
    for name, _kind, bound in fields:
        if bound is None:
            continue
        value = metadata[name]
        if name == "dtype":
            value = {v: k for k, v in KMD_DTYPES.items()}[value]
        if value is None:
            raise CatalogError(f"{where}: {slot} field {name} is unresolved")
        for launch, spec in specs.items():
            if hasattr(spec, bound) and getattr(spec, bound) != value:
                raise CatalogError(
                    f"{where}: launch {launch} spec {bound}="
                    f"{getattr(spec, bound)!r} != KMD {name}={value!r}; make the "
                    f"field matcher-only under a new name"
                )


def _check_engine_invariants(spec, where: str) -> None:
    if getattr(spec, "xcd_remap", False):
        raise CatalogError(f"{where}: xcd_remap is set")
    for name in XCD_FIELDS:
        if getattr(spec, name, None) not in (None, 0):
            raise CatalogError(f"{where}: {name} is set")


def prep_kernel(row: C.PrepRow) -> dict:
    f = row.fields
    name = (
        f"prep.{row.arch}.{f['dtype']}_d{f['head_size']}_{f['seq_mode']}"
        f"_sv{f['stage_vec']}_wl{int(f['worklist'])}_r{f['rows_per_cta']}"
        f"_nv{f['narrow_vec']}"
    )
    spec = spec_dict(row.spec, STATED_FIELDS)
    _self_check("build_attn_bwd_prep", row.spec, spec, row.arch, row.kernel)
    _check_engine_invariants(row.spec, name)
    metadata = {n: f[n] for n, _k, _b in PREP_FIELDS}
    metadata["dtype"] = KMD_DTYPES[metadata["dtype"]]
    _check_bindings("prep", PREP_FIELDS, metadata, {"": row.spec}, name)
    return {
        "name": name,
        "kernel_source": {
            "kind": "rocke",
            "source": AUX_SOURCE,
            "builder": "build_attn_bwd_prep",
            "spec": spec,
        },
        "metadata": metadata,
    }


def main_kernel(row: C.CatalogRow) -> dict:
    tk = "default" if row.tuning_key_id == 0 else f"k{row.tuning_key_id}"
    name = (
        f"main.{row.arch}.{row.dtype}_d{row.head_size}_{row.seq_mode}"
        f"_sv{row.stage_vec}_{row.dkv_mode}_{row.mask_class}_{row.dq_mode}"
        f"_hp{int(row.head_pack)}_{tk}"
    )
    split = row.dq_mode == "split"
    if split != (row.dq_spec is not None):
        raise CatalogError(f"{name}: dq_mode {row.dq_mode} with dq spec {row.dq_spec}")
    convert = spec_dict(row.convert_spec)
    _self_check(
        "build_attn_bwd_convert",
        row.convert_spec,
        convert,
        row.arch,
        row.kernels["convert_dk"],
    )
    convert_launch = {
        "source": AUX_SOURCE,
        "builder": "build_attn_bwd_convert",
        "spec": convert,
    }
    if split:
        dq = spec_dict(row.dq_spec)
        _self_check("build_attn_bwd_dq", row.dq_spec, dq, row.arch, row.kernels["dq"])
        dq_launch = {"source": DQ_SOURCE, "builder": "build_attn_bwd_dq", "spec": dq}
        launch_specs = {"dq": row.dq_spec}
    else:
        if row.kernels["dq"] != row.kernels["convert_dk"]:
            raise CatalogError(f"{name}: the dQ launch is not the convert kernel")
        dq_launch = dict(convert_launch)
        launch_specs = {"dq": row.convert_spec}
    launch_specs.update(convert_dk=row.convert_spec, convert_dv=row.convert_spec)

    kmd_names = {n for n, _k, _b in MAIN_FIELDS}
    stated = set(STATED_FIELDS)
    for launch in (dq_launch, convert_launch):
        stated |= set(launch["spec"]) & kmd_names
    main = spec_dict(row.main_spec, stated)
    _self_check(
        "build_attn_bwd_main", row.main_spec, main, row.arch, row.kernels["main"]
    )
    for spec in (row.main_spec, row.convert_spec, row.dq_spec):
        if spec is not None:
            _check_engine_invariants(spec, name)
    for launch, launch_spec in (("dq", dq_launch), ("convert", convert_launch)):
        for field in set(launch_spec["spec"]) & kmd_names:
            if launch_spec["spec"][field] != main.get(field):
                raise CatalogError(
                    f"{name}: launch {launch} states {field} differently"
                )

    metadata = dict(row.main_kmd)
    metadata["dtype"] = KMD_DTYPES[metadata["dtype"]]
    _check_bindings(
        "main", MAIN_FIELDS, metadata, {"main": row.main_spec, **launch_specs}, name
    )
    return {
        "name": name,
        "kernel_source": {
            "kind": "rocke",
            "source": MAIN_SOURCE,
            "builder": "build_attn_bwd_main",
            "spec": main,
            "launches": {
                "main": {},
                "dq": dq_launch,
                "convert_dk": dict(convert_launch),
                "convert_dv": dict(convert_launch),
            },
        },
        "metadata": {n: metadata[n] for n, _k, _b in MAIN_FIELDS},
    }


def _slot(name: str, fields) -> dict:
    kmd = []
    for field, kind, _bound in fields:
        entry = {"name": field, "type": kind}
        if field == "tuning_key":
            entry["default_value"] = ""
        kmd.append(entry)
    bound = [(f, b) for f, _k, b in fields if b is not None]
    return {
        "name": name,
        "kmd_fields": kmd,
        "specialization": {
            "metadata_fields": [f for f, _b in bound],
            "matcher_only_fields": [f for f, _k, b in fields if b is None],
            "bindings": {f: {"field": b} for f, b in bound},
            "vocabulary": {"dtype": dict(KMD_DTYPES)},
        },
    }


def _unique(pack: str, kernels: list) -> None:
    names = [k["name"] for k in kernels]
    if len(names) != len(set(names)):
        raise CatalogError(f"{pack}: duplicate kernel names")
    tuples = [tuple(sorted(map(str, k["metadata"].items()))) for k in kernels]
    if len(tuples) != len(set(tuples)):
        raise CatalogError(f"{pack}: two kernels share one KMD record")
    if len(kernels) > AOT_BUDGET_PER_PACK:
        raise CatalogError(f"{pack}: {len(kernels)} kernels > {AOT_BUDGET_PER_PACK}")


def build_config() -> dict:
    packs: dict[str, dict] = {}

    def pack(name: str, slot: str, arch: str) -> list:
        if name not in packs:
            packs[name] = {
                "name": name,
                "slot": slot,
                "arch": [arch],
                "discriminator": name,
                "kernels": [],
            }
        return packs[name]["kernels"]

    for arch in H.ENGINE_ARCHS:
        preps = C.prep_rows(arch)
        layouts = {(p.spec.ws_layout, p.spec.kv_target) for p in preps}
        if len(layouts) != 1:
            raise CatalogError(f"{arch}: prep ws_layout/kv_target vary: {layouts}")
        for prep in preps:
            pack(C.pack_of(arch, "prep"), "prep", arch).append(prep_kernel(prep))
        rows = C.catalog_rows(arch)
        if len({r.main_spec.ws_layout for r in rows}) != 1:
            raise CatalogError(f"{arch}: main ws_layout varies")
        for row in rows:
            pack(row.pack, "main", arch).append(main_kernel(row))
    order = [
        f"{slot}_{arch}" for slot in ("prep", "main") for arch in H.ENGINE_ARCHS
    ] + ["main_splitdq_gfx950"]
    if set(packs) != set(order):
        raise CatalogError(f"unexpected packs {sorted(packs)}")
    for name, p in packs.items():
        _unique(name, p["kernels"])
    return {
        "dialect": "packaged",
        "kernel_source_kind": "rocke",
        "authored_subpath": AUTHORED_SUBPATH,
        "engine": {
            "name": H.ENGINE_NAME,
            "sdk_version": "1.4.0",
            "shared_workspace": "ws.bytes",
        },
        "slots": [_slot("prep", PREP_FIELDS), _slot("main", MAIN_FIELDS)],
        "packs": [packs[name] for name in order],
    }


class _Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def _flow_mapping(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True)


class _Flow(dict):
    pass


_Dumper.add_representer(_Flow, _flow_mapping)


def _flow_leaves(node, key=None):
    """Specs and metadata in flow style (one line each); the rest in block style."""
    if isinstance(node, dict):
        if key in ("spec", "metadata", "bindings"):
            return _Flow(node)
        return {k: _flow_leaves(v, k) for k, v in node.items()}
    if isinstance(node, list):
        if key == "kmd_fields":
            return [_Flow(v) for v in node]
        return [_flow_leaves(v) for v in node]
    return node


def render(config: dict) -> str:
    body = yaml.dump(
        _flow_leaves(config),
        Dumper=_Dumper,
        sort_keys=False,
        width=1 << 16,
        default_flow_style=False,
    )
    return HEADER + "\n" + body


def summary(config: dict) -> dict:
    return {p["name"]: len(p["kernels"]) for p in config["packs"]}


# ---------------------------------------------------------------------------
# descriptors
# ---------------------------------------------------------------------------


def _tool_env() -> dict:
    env = dict(os.environ)
    env["ROCKE_BACKEND"] = "python"
    paths = [
        str(_HKP / "rocke" / "platform" / "python"),
        str(_HKP / "rocke" / "library"),
    ]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def generate(config_path: Path, bundle: Path, scratch: Path) -> Path:
    """Run generate.py on the config into ``scratch``, reusing ``bundle``'s ids
    when it holds one; returns the generated bundle directory."""
    cmd = [
        sys.executable,
        "generate.py",
        "--config",
        str(config_path),
        "--output-dir",
        str(scratch),
        "--strict",
    ]
    if any(bundle.glob("*.ued.json")):
        cmd += ["--ids-from", str(bundle)]
    else:
        cmd += ["--new-ids"]
    r = subprocess.run(
        cmd,
        cwd=GENERATOR_DIR,
        env=_tool_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        raise CatalogError(
            f"generate.py failed ({r.returncode}):\n{r.stdout}\n{r.stderr}"
        )
    out = scratch / "descriptors" / AUTHORED_SUBPATH
    if not out.is_dir():
        raise CatalogError(f"generate.py wrote no bundle at {out}")
    return out


def _installed(bundle: Path) -> dict[str, Path]:
    """The bundle's descriptors by file name; KDPs may sit in the DVC-tracked
    ``artifacts/`` folder."""
    files = {p.name: p for p in bundle.glob("*.json")}
    files.update({p.name: p for p in (bundle / "artifacts").glob("*.kdp.json")})
    return files


def _install_path(bundle: Path, name: str) -> Path:
    artifacts = bundle / "artifacts"
    if name.endswith(".kdp.json") and artifacts.is_dir():
        return artifacts / name
    return bundle / name


def install(generated: Path, bundle: Path) -> None:
    bundle.mkdir(parents=True, exist_ok=True)
    fresh = {p.name for p in generated.glob("*.json")}
    for name, path in _installed(bundle).items():
        if name not in fresh:
            path.unlink()
    for name in sorted(fresh):
        shutil.copyfile(generated / name, _install_path(bundle, name))


def stale_files(generated: Path, bundle: Path) -> list[str]:
    fresh = {p.name: p for p in generated.glob("*.json")}
    installed = _installed(bundle)
    return sorted(
        n
        for n in set(fresh) | set(installed)
        if n not in fresh
        or n not in installed
        or not filecmp.cmp(fresh[n], installed[n], shallow=False)
    )


def verify(bundle: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(GENERATOR_DIR / "tools" / "verify_variant_sets.py"),
            "--mode",
            "structural",
            "attention_general_bwd",
            str(bundle),
        ],
        cwd=GENERATOR_DIR,
        env=_tool_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--summary", action="store_true", help="print kernels per pack")
    args = ap.parse_args(argv)
    try:
        config = build_config()
    except CatalogError as exc:
        print(f"general_bwd_catalog: {exc}", file=sys.stderr)
        return 2
    if args.summary:
        for name, n in summary(config).items():
            print(f"{name}: {n}")
    text = render(config)
    failures = []
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding="utf-8") != text:
            failures.append(f"stale config {args.out}")
    else:
        args.out.write_text(text, encoding="utf-8")
    if args.generate:
        with tempfile.TemporaryDirectory(prefix="general_bwd_catalog_") as tmp:
            config_path = Path(tmp) / args.out.name
            config_path.write_text(text, encoding="utf-8")
            try:
                generated = generate(config_path, args.bundle, Path(tmp) / "out")
            except CatalogError as exc:
                print(f"general_bwd_catalog: {exc}", file=sys.stderr)
                return 2
            if args.check:
                failures += [
                    f"stale descriptor {args.bundle / n}"
                    for n in stale_files(generated, args.bundle)
                ]
            else:
                install(generated, args.bundle)
    if args.verify:
        r = verify(args.bundle)
        if r.returncode != 0:
            failures.append(
                f"verify_variant_sets.py --mode structural failed:\n{r.stdout}{r.stderr}"
            )
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
