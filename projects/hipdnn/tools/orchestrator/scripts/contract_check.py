#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Check one stage's handover contract against the checkout it claims to describe.

Three agents hand work to each other through three JSON files. The orchestrator's own
`result_schema` proves the file parses and has the keys; it cannot prove that what the
keys say is true. This does: every path is opened, every symbol is looked for in the
source that is supposed to define it, and the integration contract is cross-checked
against the authoring contract it consumed.

The check that earns this script on its own is the launch ABI. The integration agent
writes `launch()` and the authoring agent wrote the kernel's parameter list, and **a
disagreement between them is diagnosed nowhere**: hipRTC compiles the kernel, symbol
lookup resolves it, `hipModuleLaunchKernel` reads one pointer per parameter the kernel
declared, and the launch pushes whatever it has into whatever was declared. Wrong arity
or wrong order is a wrong number somewhere else, hours later, in a test that names a
different operation. Both sides state the argument list; this compares them.

The second is `consumed_authoring_sha256`. The integration contract names the digest of
the authoring contract it was written from. An integration that re-derived the ABI by
reading the tree, or that was carried over from a previous round whose kernel has since
changed, fails here rather than in the numbers.

What this does NOT prove: that any of it runs. Every check here is static. Dispatch is
proved by the shared integration suite and by nothing else.

Exit codes: 0 the report was written, whatever it says -- the flow asserts on
`failed_check_count`, so a negative verdict still leaves evidence. 1 usage or I/O
failure. 2 an unknown --contract kind.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

#: The generator writes this into every hook body it cannot write for you.
PLACEHOLDER = "FILL THIS OUT"

ENGINE_NAME = re.compile(r"^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+$")

#: The provider's CMake `project()` name. Every external integration test target is
#: spelled `${PROJECT_NAME}-<tail>` in the source and expands to this prefix in CTest.
PROVIDER_PROJECT = "hip-kernel-provider"


class Checks:
    """A named pass/fail ledger. A check is a name, a verdict and, when it fails, the
    sentence the retrying agent needs -- never a bare False."""

    def __init__(self) -> None:
        self.results: dict[str, int] = {}
        self.notes: list[str] = []

    def add(self, name: str, ok: bool, why: str = "") -> bool:
        self.results[name] = 1 if ok else 0
        if not ok and why:
            self.notes.append(f"[{name}] {why}")
        return bool(ok)

    @property
    def failed(self) -> list[str]:
        return [name for name, value in self.results.items() if not value]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _paths(value: Any) -> list[Path]:
    return [Path(str(item)) for item in (value or []) if str(item).strip()]


def _signature_params(signature: str) -> list[str]:
    """The parameter list of a C declaration, split at top-level commas.

    Top-level because `hipdnn::Vec<float, 4> v` has a comma that is not a separator, and
    a naive split reports an arity that is wrong in the safe-looking direction.
    """
    start = signature.find("(")
    end = signature.rfind(")")
    if start < 0 or end <= start:
        return []
    body = signature[start + 1 : end].strip()
    if not body or body == "void":
        return []
    params, depth, current = [], 0, []
    for char in body:
        if char in "(<[":
            depth += 1
        elif char in ")>]":
            depth -= 1
        if char == "," and depth == 0:
            params.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    params.append("".join(current).strip())
    return [p for p in params if p]


# -- authoring ---------------------------------------------------------------


def check_authoring(args: argparse.Namespace, contract: Any, checks: Checks) -> None:
    bundle = contract.get("bundle") or {}
    sources = _paths(bundle.get("sources"))
    headers = _paths(bundle.get("headers"))
    entry_points = contract.get("entry_points") or []
    required = contract.get("required_defines") or []

    checks.add("sources_declared", bool(sources), "bundle.sources is empty")
    absent = [p for p in sources + headers if not p.is_file()]
    checks.add(
        "bundle_on_disk",
        not absent,
        "these bundle files do not exist: " + ", ".join(p.as_posix() for p in absent),
    )

    # A header the bundle rules cannot load is a build failure at prepare() time, not
    # here, and the rule is not guessable from an error message: headers are `.h`, `.hpp`
    # and `.cuh` only, one level deep. A second .hip is not a header.
    wrong_suffix = [p for p in headers if p.suffix not in (".h", ".hpp", ".cuh")]
    checks.add(
        "header_suffixes",
        not wrong_suffix,
        "bundle.headers may only hold .h, .hpp or .cuh; found "
        + ", ".join(p.name for p in wrong_suffix),
    )

    checks.add("entry_points_declared", bool(entry_points), "entry_points is empty")
    source_names = {p.name for p in sources}
    problems: list[str] = []
    for entry in entry_points:
        name = str(entry.get("name", ""))
        owner = Path(str(entry.get("source_file", "")))
        signature = str(entry.get("signature", ""))
        params = entry.get("params") or []
        if owner.name not in source_names:
            problems.append(
                f"{name}: source_file {owner.name} is not in bundle.sources"
            )
            continue
        if not owner.is_file():
            continue
        body = _text(owner)
        # The declaration, not a mention: `extern "C" __global__` somewhere ahead of the
        # name. Anchored loosely because attribute order and line breaks vary, and
        # tightly enough that a comment naming the kernel does not satisfy it.
        declared = re.search(
            rf'extern\s+"C"[^;{{]{{0,200}}?__global__[^;{{]{{0,200}}?\b{re.escape(name)}\s*\(',
            body,
            re.S,
        )
        if not declared:
            problems.append(
                f'{name}: no `extern "C" __global__ ... {name}(` declaration in '
                f"{owner.name}; a C++-linkage kernel is unreachable by descriptor name"
            )
        declared_arity = len(_signature_params(signature))
        if declared_arity != len(params):
            problems.append(
                f"{name}: signature declares {declared_arity} parameter(s) but the "
                f"params list has {len(params)}; the two are the same statement and a "
                "handler written from the shorter one launches into the longer kernel"
            )
    checks.add("entry_points_consistent", not problems, "\n  ".join(problems))

    # Every required macro must be guarded in a source that uses it. rtc_compile.py
    # proves this by compiling; this catches it in seconds, before a device is involved.
    unguarded = []
    for entry in required:
        name = str(entry.get("name", ""))
        if not name:
            continue
        if not entry.get("legal_values"):
            unguarded.append(f"{name}: no legal_values listed, so nothing can bind it")
            continue
        guarded = any(
            re.search(
                rf"#\s*ifndef\s+{re.escape(name)}\b[^#]{{0,400}}?#\s*error",
                _text(p),
                re.S,
            )
            for p in sources
            if p.is_file()
        )
        if not guarded:
            unguarded.append(
                f"{name}: no `#ifndef {name}` / `#error` pair in any source"
            )
    checks.add("macros_guarded", not unguarded, "\n  ".join(unguarded))

    numerics = contract.get("numerics") or {}
    command = numerics.get("harness_command") or []
    report_path = str(numerics.get("report_path", ""))
    checks.add(
        "harness_declared",
        bool(command) and bool(report_path) and Path(report_path).is_absolute(),
        "numerics.harness_command must be a non-empty argv list and "
        "numerics.report_path an absolute path; the orchestrator runs the harness and "
        "reads that file, so a relative path resolves against a directory neither side "
        "agreed on",
    )
    checks.add(
        "tolerance_provenance",
        bool(str(numerics.get("tolerance_provenance", "")).strip()),
        "numerics.tolerance_provenance is empty; a tolerance with no stated source is a "
        "number chosen until the comparison passed",
    )

    # Every tensor the graph declares has to be classified, and every output has to be
    # written by some launch. An output nobody writes is the failure the sentinel check
    # in the harness is for; naming it here makes it a contract error instead.
    tensors = contract.get("tensors") or []
    unclassified = [
        t.get("uid")
        for t in tensors
        if t.get("role") not in ("input", "output", "virtual")
    ]
    checks.add(
        "tensors_classified",
        bool(tensors) and not unclassified,
        (
            "every tensor needs role input|output|virtual; unclassified uids: "
            + ", ".join(str(u) for u in unclassified)
            if tensors
            else "tensors is empty; the graph's UIDs were never enumerated"
        ),
    )
    outputs = {t.get("uid") for t in tensors if t.get("role") == "output"}
    written = {
        p.get("uid")
        for entry in entry_points
        for p in (entry.get("params") or [])
        if p.get("role") == "tensor_out"
    }
    checks.add(
        "outputs_written",
        outputs.issubset(written),
        "these output UIDs are not a tensor_out parameter of any entry point: "
        + ", ".join(str(u) for u in sorted(outputs - written, key=str)),
    )

    checks.add(
        "abi_stated",
        contract.get("abi") in ("unbound", "bound"),
        "abi must be 'unbound' (the normal answer: the integration writes the handler) "
        "or 'bound' (you are adding to a pack that already ships one)",
    )
    checks.add(
        "limits_stated",
        bool(contract.get("does_not_prove")),
        "does_not_prove is empty; a kernel proved on one shape, one dtype and one device "
        "has limits, and an empty list claims it has none",
    )

    if args.graph:
        graph = _load(Path(args.graph))
        node_count = len(graph.get("nodes") or [])
        checks.add(
            "graph_nodes_covered",
            bool(contract.get("decomposition")),
            f"the graph has {node_count} node(s) and decomposition is empty; the "
            "launch plan is what integration writes prepare() from",
        )


# -- integration -------------------------------------------------------------

#: Hooks that are never optional. A pack without a graph_match empties the engine's
#: catalog; a pack without a dispatch handler has nothing to launch.
MANDATORY_HOOKS = ("graph_match", "dispatch")


def check_integration(args: argparse.Namespace, contract: Any, checks: Checks) -> None:
    identity = _load(Path(args.identity)) if args.identity else {}
    authoring_path = Path(args.authoring) if args.authoring else None

    engine = str(contract.get("engine_name", ""))
    checks.add(
        "engine_name_wellformed",
        bool(ENGINE_NAME.match(engine)),
        f"engine_name {engine!r} is not <namespace>:<local>",
    )
    if identity:
        checks.add(
            "engine_name_matches_identity",
            engine == identity.get("engine_name"),
            f"engine_name {engine!r} is not the identity this run was started with "
            f"({identity.get('engine_name')!r}); the validator, the TOML, the CMake "
            "target and --test-engine are all derived from that one name",
        )

    native = Path(str(contract.get("native_file", "")))
    checks.add("native_file_exists", native.is_file(), f"{native} does not exist")
    native_text = _text(native) if native.is_file() else ""

    checks.add(
        "no_placeholders_left",
        PLACEHOLDER not in native_text,
        f"{native.name} still contains {PLACEHOLDER!r}; a stub hook body is a hook that "
        "is not implemented, whatever the contract says",
    )

    hooks = contract.get("hooks") or []
    by_role = {str(h.get("role")): h for h in hooks}
    missing_roles = [r for r in MANDATORY_HOOKS if r not in by_role]
    checks.add(
        "mandatory_hooks_present",
        not missing_roles,
        "these hooks are not in the contract at all: " + ", ".join(missing_roles),
    )
    undeclared = [
        str(h.get("role"))
        for h in hooks
        if h.get("implemented") and str(h.get("symbol", "")) not in native_text
    ]
    checks.add(
        "hook_symbols_in_source",
        not undeclared,
        "these hooks are reported implemented but their symbol does not appear in "
        + f"{native.name}: "
        + ", ".join(undeclared),
    )
    declined_without_reason = [
        str(h.get("role"))
        for h in hooks
        if not h.get("implemented") and not str(h.get("reason_if_declined", "")).strip()
    ]
    checks.add(
        "declines_explained",
        not declined_without_reason,
        "a hook may be declined, but not silently: "
        + ", ".join(declined_without_reason),
    )

    registration = contract.get("registration") or {}
    register_fn = str(registration.get("symbol_scope_function", ""))
    expected_fn = str(identity.get("register_symbol", register_fn))
    checks.add(
        "registration_function",
        register_fn == expected_fn and register_fn in native_text,
        f"expected {expected_fn}() defined in {native.name}; the contract says "
        f"{register_fn!r}",
    )
    # Both splices, because one without the other is the silent case: a pack declared in
    # the header but absent from the s_packs table links fine into the plugin and
    # vanishes from the statically-linked unit test binary, with no error either way.
    if args.repo:
        repo = Path(args.repo)
        packs_cpp = (
            repo
            / "dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/IngestorPacks.cpp"
        )
        packs_hpp = packs_cpp.with_suffix(".hpp")
        cpp_text = _text(packs_cpp) if packs_cpp.is_file() else ""
        hpp_text = _text(packs_hpp) if packs_hpp.is_file() else ""
        checks.add(
            "packs_table_row",
            engine in cpp_text and register_fn in cpp_text,
            f"IngestorPacks.cpp must hold a row naming {engine!r} and {register_fn}",
        )
        checks.add(
            "packs_header_decl",
            register_fn in hpp_text,
            f"IngestorPacks.hpp must declare {register_fn}",
        )
        cmake = repo / "dnn-providers/hip-kernel-provider/src/CMakeLists.txt"
        cmake_text = _text(cmake) if cmake.is_file() else ""
        target = str(contract.get("external_test_target", ""))
        # The contract names the CTest entry, which is what `ctest -R` and the run report
        # have to agree on. The CMake source spells it `${PROJECT_NAME}-<tail>`, so the
        # expanded name never appears literally in the file. Accept either spelling: the
        # question here is whether an entry was added, and `ctest -N` later answers
        # whether it registered.
        spellings = [target]
        if target.startswith(f"{PROVIDER_PROJECT}-"):
            spellings.append("${PROJECT_NAME}-" + target[len(PROVIDER_PROJECT) + 1 :])
        checks.add(
            "external_target_registered",
            bool(target) and any(s and s in cmake_text for s in spellings),
            f"neither {spellings} appears in {cmake}; without an "
            "add_external_integration_test_target entry the engine is runnable only by "
            "hand, which is not registration",
        )

    toml = Path(str(contract.get("engine_toml", "")))
    checks.add("engine_toml_exists", toml.is_file(), f"{toml} does not exist")

    descriptors = _paths(contract.get("descriptor_files"))
    absent = [p for p in descriptors if not p.is_file()]
    checks.add(
        "descriptors_on_disk",
        bool(descriptors) and not absent,
        "descriptor_files is empty or names files that do not exist: "
        + ", ".join(p.as_posix() for p in absent),
    )

    kind = str(contract.get("kernel_source_kind", ""))
    checks.add(
        "kernel_source_kind_known",
        kind in ("embedded_source", "hiprtc_file"),
        f"kernel_source_kind {kind!r} is not one a direct-load pack can ship; "
        "kpack is the packaged dialect and rocke is stubbed in IngestorKernelCode.hpp",
    )
    if kind == "hiprtc_file":
        # ConvNative.cpp compiles kernel.source.sourceFile directly, which serves
        # embedded_source only, and a hiprtc_file descriptor under it throws at
        # plan-build time however correct the descriptor is.
        checks.add(
            "handler_uses_builder",
            "buildIngestorKernelCode" in native_text,
            f"{native.name} must call buildIngestorKernelCode() in prepare(); a handler "
            "that compiles kernel.source.sourceFile itself serves embedded_source only "
            "and throws at plan-build time for a hiprtc_file descriptor",
        )

    cases = contract.get("bundle_case_ids") or []
    checks.add(
        "bundle_cases_added",
        bool(cases),
        "bundle_case_ids is empty; an integration with no graphs has been verified by "
        "nothing, and a --test-engine run that selects nothing still exits 0",
    )

    # The seam. Both sides state the argument list, and nothing in the toolchain compares
    # them.
    if authoring_path and authoring_path.is_file():
        authoring = _load(authoring_path)
        digest = _sha256(authoring_path)
        checks.add(
            "consumed_authoring_digest",
            str(contract.get("consumed_authoring_sha256", "")) == digest,
            "consumed_authoring_sha256 does not match the authoring contract this run "
            f"produced (expected {digest}). Either the handover was not read, or it has "
            "changed since -- and an ABI copied from an older kernel is the defect this "
            "field exists to catch",
        )
        launched = str(contract.get("entry_point_launched", ""))
        entry = next(
            (
                e
                for e in (authoring.get("entry_points") or [])
                if e.get("name") == launched
            ),
            None,
        )
        if entry is None:
            checks.add(
                "launch_entry_point_known",
                False,
                f"entry_point_launched {launched!r} is not one of the authoring "
                "contract's entry points: "
                + ", ".join(
                    str(e.get("name")) for e in (authoring.get("entry_points") or [])
                ),
            )
        else:
            checks.add("launch_entry_point_known", True)
            kernel_params = [str(p.get("name")) for p in (entry.get("params") or [])]
            launch_order = [str(n) for n in (contract.get("launch_arg_order") or [])]
            checks.add(
                "launch_arity",
                len(launch_order) == len(kernel_params),
                f"launch() passes {len(launch_order)} argument(s) and {launched} declares "
                f"{len(kernel_params)}. hipModuleLaunchKernel reads one pointer per "
                "declared parameter, so the shorter side does not raise -- it reads "
                "whatever is next in memory",
            )
            checks.add(
                "launch_order",
                launch_order == kernel_params,
                f"launch() argument order {launch_order} is not {launched}'s parameter "
                f"order {kernel_params}. Two same-typed pointers swapped is a wrong "
                "answer with no diagnostic anywhere",
            )

    checks.add(
        "changes_reported",
        bool(contract.get("changed_files")),
        "changed_files is empty; nothing reached disk",
    )


# -- ingestor ----------------------------------------------------------------


def check_ingestor(args: argparse.Namespace, contract: Any, checks: Checks) -> None:
    identity = _load(Path(args.identity)) if args.identity else {}
    engine = str(contract.get("engine_name", ""))
    if identity:
        checks.add(
            "engine_name_matches_identity",
            engine == identity.get("engine_name"),
            f"engine_name {engine!r} is not {identity.get('engine_name')!r}",
        )

    prefix = Path(str(contract.get("install_prefix", "")))
    root = Path(str(contract.get("final_descriptor_root", "")))
    checks.add("install_prefix_exists", prefix.is_dir(), f"{prefix} is not a directory")
    checks.add(
        "final_descriptor_root_exists",
        root.is_dir(),
        f"{root} is not a directory; the final gates run against the INSTALLED tree, and "
        "a build-tree path here means they were run against something else",
    )
    if root.is_dir():
        found = list(root.rglob("*.json"))
        checks.add(
            "descriptor_root_populated",
            bool(found),
            f"{root} holds no descriptor JSON at all",
        )

    rocke = contract.get("rocke") or {}
    # rocKE is ON in this flow's build. That makes "was it used" a question with an
    # answer, and an unexamined switch is how a run acquires a capability nobody checked.
    checks.add(
        "rocke_disposition_stated",
        "enabled_in_build" in rocke and str(rocke.get("why", "")).strip() != "",
        "rocke must state enabled_in_build and why it was or was not used. The build "
        "turns HIPKERNELPROVIDER_ENABLE_ROCKE on; a direct-load HIP pack does not lower "
        "through it, and saying so is the difference between a decision and an oversight",
    )

    corpus = contract.get("corpus") or {}
    checks.add(
        "corpus_declared",
        isinstance(corpus.get("inputs"), list) and bool(corpus.get("root")),
        "corpus.root and corpus.inputs are the denominator every coverage number is "
        "quoted against",
    )
    checks.add(
        "candidates_named",
        bool(contract.get("candidates_selected")),
        "candidates_selected is empty; if every pass came from one kernel, the other "
        "variants are unexercised no matter how many cases ran",
    )
    checks.add(
        "limits_stated",
        bool(contract.get("does_not_prove")),
        "does_not_prove is empty",
    )


KINDS: dict[str, Callable[[argparse.Namespace, Any, Checks], None]] = {
    "authoring": check_authoring,
    "integration": check_integration,
    "ingestor": check_ingestor,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, choices=sorted(KINDS))
    parser.add_argument("--file", required=True, help="the contract JSON to check")
    parser.add_argument("--out", required=True, help="where to write the JSON report")
    parser.add_argument("--repo", help="checkout root, for the splice checks")
    parser.add_argument("--identity", help="identity.json from the identity step")
    parser.add_argument("--authoring", help="authoring.json, for the ABI cross-check")
    parser.add_argument("--graph", help="the graph this run was given")
    args = parser.parse_args()

    path = Path(args.file)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checks = Checks()
    try:
        contract = _load(path)
    except (OSError, ValueError) as error:
        checks.add("contract_parses", False, f"could not read {path}: {error}")
        contract = None

    if contract is not None:
        checks.add("contract_parses", True)
        if not isinstance(contract, dict):
            checks.add("contract_is_object", False, "the contract is not a JSON object")
        else:
            checks.add("contract_is_object", True)
            KINDS[args.contract](args, contract, checks)

    report = {
        "kind": args.contract,
        "file": path.as_posix(),
        "checks": checks.results,
        "failed_checks": checks.failed,
        "failed_check_count": len(checks.failed),
        "check_count": len(checks.results),
        "ok": 0 if checks.failed else 1,
        "feedback": (
            ""
            if not checks.failed
            else f"The {args.contract} contract does not describe the checkout it was "
            "written against:\n\n" + "\n".join(checks.notes)
        ),
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"{args.contract} contract: {len(checks.results) - len(checks.failed)}"
        f"/{len(checks.results)} check(s) passed."
    )
    for note in checks.notes:
        print(f"  {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
