# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Portable --specialize fixtures: a fixture baseline build, producer-shaped traces,
route facts, and helpers that run the real CPU-only CLI and read its outputs."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import shutil
from contextlib import ExitStack, contextmanager
from pathlib import Path

import yaml

from geko import cli

DATA = Path(__file__).parent / "test_data" / "specialize_writer"
CALL_DATA = Path(__file__).parent / "test_data" / "specialize_calls"
RECIPES = json.loads((DATA / "recipes.json").read_text())
ORACLE = yaml.safe_load((DATA / "oracle.yaml").read_text())
MANIFEST_COLUMNS = (
    "problem_id tier provenance source_path source_sha256 source_lines model run derivation "
    "call_count transA transB M N K batch lda ldb ldc ldd strides a_type b_type c_type d_type "
    "compute_type alpha beta pointer_mode bias bias_type activation act_args sav recipe_mode "
    "stock_solution_index stock_kernel_name stock_logic_file stock_logic_index refusal_cases "
    "extra_outputs api_path"
).split()
META = {"call_count", "solution_index", "solution_Name", "kernel_name"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hash(path):
    # Same content fingerprint as the trace producer's run.json library_fp.
    digest = hashlib.sha256()
    path = Path(path)
    for directory, dirs, names in os.walk(path):
        dirs.sort()
        for name in sorted(names):
            file = Path(directory) / name
            rel = str(file.relative_to(path))
            if file.is_symlink():
                digest.update(f"{rel}\0L\0{os.readlink(file)}\0".encode())
            else:
                digest.update(f"{rel}\0{file.stat().st_size}\0{sha(file)}\0".encode())
    return digest.hexdigest()


def write_rows(path, rows):
    """Producer-shaped flow records that keep each scalar's logged spelling."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def scalar(value):
        return str(value).lower() if isinstance(value, bool) else str(value)
    path.write_text("".join(
        "- { " + ", ".join(f"{k}: {scalar(v)}" for k, v in row.items()) + " }\n"
        for row in rows
    ))
    return path


def row(recipe="dp", **changes):
    record = dict(yaml.load((CALL_DATA / "producer_default.yaml").read_text(), Loader=yaml.BaseLoader)[0])
    metadata = RECIPES[recipe]
    record.update(solution_index=str(metadata["baseline_index"]),
                  solution_Name=metadata["solution_name"], kernel_name=metadata["kernel_name"],
                  M="32", N="618", K="120", lda="32", ldb="120", ldc="32", ldd="32",
                  a_type="bf16_r", b_type="bf16_r", c_type="bf16_r", d_type="bf16_r",
                  bias_type="bf16_r", aux_type="bf16_r", beta="0.000000", call_count="2")
    if recipe == "f32":
        record.update(M="128", N="128", K="128", lda="128", ldb="128", ldc="128", ldd="128",
                      a_type="f32_r", b_type="f32_r", c_type="f32_r", d_type="f32_r", bias_type="f32_r",
                      aux_type="f32_r", beta="1.000000")
    if recipe in {"sk", "g942"}:
        record.update(transA="T", M="2048", N="576", K="8192", lda="8192", ldb="8192", ldc="2048", ldd="2048")
    if recipe == "f16":
        record.update(a_type="f16_r", b_type="f16_r", c_type="f16_r", d_type="f16_r", bias_type="f16_r", aux_type="f16_r")
    record.update({k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in changes.items()})
    return record


def case_record(kind, case):
    """A host_calls / generation_only record from oracle.yaml: row(recipe) plus its changes."""
    spec = ORACLE[kind][case]
    record = row(spec["recipe"], **spec["changes"])
    for key in spec.get("remove", []):
        del record[key]
    return record


def oracle_logic(name):
    """Expected bespoke logic: stock recipe plus independently recorded constraint changes.
    The base is hash-pinned, so a fixture edit cannot move input and expectation together."""
    spec = ORACLE["oracle"][name]
    base = DATA / "stock" / RECIPES[spec["recipe"]]["logic_file"]
    assert sha(base) == RECIPES[spec["recipe"]]["source_sha256"], f"oracle base {spec['recipe']} changed"
    logic = yaml.safe_load(base.read_text())
    for key, changes in spec.items():
        if key != "recipe":
            (logic["Solutions"][0] if key == "Solution" else logic if key == "top" else logic[key]).update(changes)
    return logic


def identity(record):
    return {k: str(v) for k, v in record.items() if k not in META}


def recipe_key(name="dp"):
    metadata = RECIPES[name]
    return {"source_file": metadata["logic_file"], "sha256": metadata["source_sha256"], "in_file_index": 0}


def route_document(records, cfg="fixture-on", **changes):
    calls = []
    scan = [{"file": "synthetic-api.log", "line_start": 1, "line_end": 1,
             "sha256": "0" * 64, "observation": "synthetic fixture, not live evidence"}]
    witness = [{"file": "synthetic-api.log", "line": 1, "sha256": "0" * 64}]
    for record in records:
        evidence = {"call_route": witness, "bias_axis": witness, "output_amax": scan, "runtime_route": witness,
                    "identity": [{"file": "hipblaslt_101.log", "line": 1, "sha256": "0" * 64}]}
        calls.append({"identity": identity(record), "call_route": "ordinary matmul API", "bias_axis": "M",
                      "output_amax": False, "runtime_route": "hipBLASLt heuristic", "evidence": evidence,
                      "attribution": "synthetic configuration proof, not live evidence", **changes})
    count = sum(int(r["call_count"]) for r in records)
    distinct = len({tuple(sorted(identity(record).items())) for record in records})
    return {"schema": 1, "run_configuration": cfg, "calls": calls, "matmul_count": count,
            "mask128_call_count": count, "per_process": {"101": {"rocblaslt_create": 1, "rocblaslt_matmul": count}},
            "call_count": count, "thin": distinct < 2, "distinct_identities": distinct,
            "reconciliation": {"matched": True, "mask16_matmul_count": count, "mask128_call_count": count, "evidence": scan},
            "stops": [], "zero_verified": count == 0, "scan": scan,
            "runtime_route": {"proven": True, "evidence": scan, "reason": None},
            "run_json": {"file": "synthetic-api-run.json", "sha256": "0" * 64},
            "producer": {"class": "synthetic fixture, not trace evidence"}}


def build_fixture(root):
    root = Path(root)
    source = root / "source-logic"
    shutil.copytree(DATA / "stock", source)
    # Second recipe identities built from the dp file: a byte copy at another path, and
    # the same stock file with another of its solutions (see stock.source.json).
    dp = (source / RECIPES["dp"]["logic_file"]).read_text()
    head, rest = dp.split("Solutions:\n", 1)
    variants = {"duplicate": dp,
                "dp_other": head + "Solutions:\n" + (DATA / "solution98_Solutions.yaml").read_text()
                + "IndexOrder:\n" + rest.split("IndexOrder:\n", 1)[1]}
    for name, text in variants.items():
        (source / RECIPES[name]["logic_file"]).write_text(text)
    for name, metadata in RECIPES.items():
        assert sha(source / metadata["logic_file"]) == metadata["source_sha256"], f"stock fixture {name} changed"
    table = yaml.safe_load((DATA / "MatchTable.yaml").read_text())
    table = {index: [path.replace("@LOGIC@", str(source)), local] for index, (path, local) in table.items()}
    (root / "device-library").mkdir()
    (root / "device-library" / "MatchTable.yaml").write_text(yaml.safe_dump(table))
    (root / "params.txt").write_text(f"src=fixture logic={source} build={root} jobs=1\nfixture-build\n")
    for arch in ("gfx950", "gfx942"):
        library = root / "Tensile" / "library" / arch
        library.mkdir(parents=True)
        # A content-tree fixture, not a loadable GPU library.
        (library / f"TensileLibrary_lazy_{arch}.dat").write_bytes(("fixture-" + arch).encode())
    return root


def input_fixture(root, records, cfg="fixture-on", route_changes=None, run=True):
    root = Path(root)
    logdir = root / "traces" / cfg
    logdir.mkdir(parents=True, exist_ok=True)
    write_rows(logdir / "hipblaslt_101.log", records)
    factsdir = root / "route_facts"
    factsdir.mkdir(exist_ok=True)
    factsfile = factsdir / f"{cfg}.json"
    factsfile.write_text(json.dumps(route_document(records, cfg, **(route_changes or {})), indent=2))
    if run:
        library = root / "baseline/Tensile/library/gfx950"
        (logdir / "run.json").write_text(json.dumps({"status": "complete", "mode": "analysis",
            "model": {"slug": "fixture-model"}, "run_configuration": cfg,
            "library": str(library), "library_fp": tree_hash(library),
            "env_values": {}, "process_inventory": [{"pid": 101, "role": "engine"}]}))
    return logdir, factsfile


def configured(root, records, cfg="fixture-on", run=True):
    build_fixture(root / "baseline")
    return input_fixture(root, records, cfg=cfg, run=run)


def command(root, logs, facts, out, arch="gfx950", extra=()):
    return ["--specialize", "--workload-log", str(logs), "--route-facts", str(facts),
            "--hipblaslt", str(root / "baseline"), "--arch", arch, "--workdir", str(out), *extra]


def invoke(argv, monkeypatch):
    # Generation is CPU-only. A wrong legacy dispatch must fail before touching a GPU.
    def gpu_forbidden(*args, **kwargs):
        raise AssertionError("--specialize attempted GPU enumeration")
    monkeypatch.setattr(cli.device, "main", gpu_forbidden)
    return cli.main(argv)


def generated(root, logs, facts, monkeypatch, extra=(), arch="gfx950", out=None):
    """Run --specialize on a configured root (trace/route directories) and read outputs."""
    out = out or root / "out"
    return outputs(out, invoke(command(root, logs.parent, facts.parent, out, arch, extra), monkeypatch))


def specialize(root, records, monkeypatch, *, arch="gfx950", extra=(), route_changes=None, run=True):
    root = Path(root)
    build = build_fixture(root / "baseline")
    logs, facts = input_fixture(root, records, route_changes=route_changes, run=run)
    if arch != "gfx950" and run:
        meta = json.loads((logs / "run.json").read_text())
        meta["library_fp"] = tree_hash(build / "Tensile/library" / arch)
        (logs / "run.json").write_text(json.dumps(meta))
    return generated(root, logs, facts, monkeypatch, extra, arch)


def outputs(work, code):
    work = Path(work)
    manifest = json.loads((work / "specialize_manifest.json").read_text()) if (work / "specialize_manifest.json").is_file() else {}
    refusals = json.loads((work / "refusals.json").read_text()) if (work / "refusals.json").is_file() else []
    if isinstance(refusals, dict):
        refusals = refusals["refusals"]
    rows = []
    if (work / "calls.csv").is_file():
        with (work / "calls.csv").open() as stream:
            rows = list(csv.DictReader(stream))
    logic = {p.name: yaml.safe_load(p.read_text()) for p in sorted((work / "logic").rglob("*.yaml"))}
    return {"code": code, "manifest": manifest, "calls": rows, "refusals": refusals, "logic": logic, "work": work}


def one_solution(result):
    assert len(result["logic"]) == 1, "expected one generated bespoke logic file"
    logic = next(iter(result["logic"].values()))
    assert len(logic["Solutions"]) == 1
    return logic


def effective_solution(logic):
    return {**logic.get("DefaultSolution", {}), **logic["Solutions"][0]}


def refusal(result, count, reason):
    assert result["code"] != 0, "refused input must return nonzero"
    assert len(result["refusals"]) == 1
    entry = result["refusals"][0]
    assert entry["call_count"] == count
    assert reason.lower() in entry["reason"].lower()
    return entry


@contextmanager
def held_directories(paths):
    # Open handles stop inode reuse from disguising a delete-and-recreate.
    with ExitStack() as stack:
        handles = []
        for path in paths:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            stack.callback(os.close, fd)
            handles.append((path, fd))
        yield handles


def assert_directories_preserved(handles):
    for path, fd in handles:
        assert path.is_dir(), f"unowned directory was removed: {path}"
        assert os.path.samestat(path.stat(), os.fstat(fd)), f"unowned directory was deleted and recreated: {path}"


def readme_fenced_blocks(text):
    """Matched fenced blocks in the specialize section, including deeper headings."""
    blocks = []
    section_depth = None
    fence = None
    language = ""
    body = []
    for line in text.splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence is not None:
            if (marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence)
                    and not marker[2].strip()):
                if section_depth is not None:
                    blocks.append((language, "\n".join(body)))
                fence = None
            elif section_depth is not None:
                body.append(line)
            continue
        heading = re.match(r"^(#{1,6})[ \t]+(.*)$", line)
        if heading:
            depth = len(heading[1])
            if section_depth is not None and depth <= section_depth:
                break
            if section_depth is None and re.search(r"\bspecialize\b", heading[2], re.I):
                section_depth = depth
        if marker:
            fence = marker[1]
            info = marker[2].strip().split()
            language = info[0] if info else ""
            body = []
    return blocks


def readme_specialize_examples(text):
    examples = []
    for language, block in readme_fenced_blocks(text):
        if language not in ("", "bash", "sh", "shell"):
            continue
        for line in block.replace("\\\n", " ").splitlines():
            candidate = line.lstrip().removeprefix("$ ")
            first = candidate.split(maxsplit=1)
            if not first:
                continue
            executable = Path(first[0]).name
            if executable != "geko" and not re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", executable):
                continue
            tokens = shlex.split(candidate, comments=True)
            if "--specialize" in tokens:
                examples.append(tokens[tokens.index("--specialize"):])
    return examples
