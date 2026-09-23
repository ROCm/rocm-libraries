#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""E2E checks for a source-built hipblaslt-bench; never builds or installs anything."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rows(stdout):
    lines = stdout.splitlines()
    found = []
    for index, line in enumerate(lines[:-1]):
        if "norm_error" not in line or ",atol" not in line or ",rtol" not in line:
            continue
        header = re.sub(r"^\s*\[\d+\]:", "", line).strip()
        fields = next(csv.reader([header]))
        # Complex alpha/beta are printed as (real,imag) by the benchmark.
        row = re.sub(r"\(([^()]*)\)", r'"\1"', lines[index + 1].strip())
        values = next(csv.reader([row]))
        require(len(fields) == len(values), "Malformed benchmark result row")
        found.append(dict(zip(fields, values)))
    require(found, "No benchmark correctness CSV row found")
    return found


def check_numerics(stdout):
    result = rows(stdout)
    # Match clients/common/include/norm.hpp: include both input types even when
    # C/D are FP32. MX and other low-precision inputs use that ordinary policy.
    tolerances = {
        "f32_r": 1e-5,
        "f64_r": 1e-12,
        "f32_c": 1e-5,
        "f64_c": 1e-12,
        "f16_r": 0.01,
        "bf16_r": 0.1,
        "f8_r": 0.125,
        "bf8_r": 0.25,
        "f8_fnuz_r": 0.125,
        "bf8_fnuz_r": 0.25,
        "i32_r": 1e-4,
        "i8_r": 0.01,
        "f4_r": 0.3,
        "f6_r": 0.5,
        "bf6_r": 0.5,
    }
    for row in result:
        bound = max(tolerances[row[field]] for field in ("a_type", "b_type", "d_type"))
        require(
            all(value.strip().lower() != "failed" for value in row.values()),
            "Benchmark reported failed allclose",
        )
        for field in ("norm_error", "atol", "rtol", "us"):
            value = float(row[field])
            require(math.isfinite(value) and value >= 0, f"Invalid {field}: {value}")
        require(
            float(row["norm_error"]) <= bound,
            f'Norm error {row["norm_error"]} exceeds {bound}',
        )
        require(
            float(row["atol"]) < 1 and float(row["rtol"]) < 1,
            "Allclose tolerances were not resolved",
        )
    return result


def check_provenance(stderr, stdout, case, artifact_root, architecture):
    import yaml  # Use the existing configured venv (already needed by Tensile).

    def reported(label):
        matches = re.findall(r"^" + re.escape(label) + r": (.+)$", stderr, re.MULTILINE)
        require(len(matches) == 1, f"Expected one {label}, got {len(matches)}")
        path = Path(matches[0]).resolve(strict=True)
        require(
            path.is_relative_to(artifact_root), f"Artifact escaped test output: {path}"
        )
        return path

    config = reported("JIT recipe")
    manifest_path = reported("JIT manifest")
    require(
        "JIT recipe:" not in stdout and "JIT manifest:" not in stdout,
        "JIT diagnostics polluted benchmark stdout",
    )
    manifests = list(artifact_root.rglob("manifest.json"))
    require(manifests == [manifest_path], f"Expected one generated bundle: {manifests}")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["counts"]["solutions"] == 1, "Generated more than one solution")
    require(
        manifest["counts"]["main_kernels"] == 1, "Generated more than one main kernel"
    )
    require(
        manifest["main_kernel"]["name"] in stdout,
        "Reported kernel differs from artifact",
    )
    require(
        manifest["architecture"]["resolved"].split(":")[0] == architecture,
        "Wrong device ISA",
    )
    prediction = manifest["jit_prediction"]
    model = prediction["model"]
    require(
        model == "origami.gemm.estimation", "Prediction substituted an unranked recipe"
    )
    ranked = prediction["ranked_candidates"]
    require(ranked, "No candidates were recorded")
    for candidate in ranked:
        score = candidate["predicted_cycles"]
        require(math.isfinite(score) and 0 < score < 1e300, "Invalid Origami score")
        parameters = candidate["parameters"]
        mi = parameters["MatrixInstruction"]
        require(
            len(mi) == 9 and all(isinstance(value, int) and value > 0 for value in mi),
            f"Malformed matrix instruction: {mi}",
        )
        if architecture in ("gfx90a", "gfx1250"):
            require(
                parameters["NonTemporalA"] == parameters["NonTemporalB"] == 0,
                f"{architecture} recipe uses unsupported cache hints",
            )
    # Origami can reorder near-equal scores using tie-breaking. Preserve its rank order.
    rejected = {item["candidate_id"] for item in prediction["rejections"]}
    accepted = next(
        candidate for candidate in ranked if candidate["id"] not in rejected
    )
    require(
        prediction["candidate_id"] == accepted["id"],
        "Not the first valid ranked candidate",
    )
    require(
        prediction["predicted_cycles"] == accepted["predicted_cycles"], "Score changed"
    )
    require(
        prediction["selected_parameters"] == accepted["parameters"], "Candidate changed"
    )
    require(prediction["defaults_source"], "No default provenance")
    problem = prediction["problem"]
    for key, expected in case["problem"].items():
        require(
            problem[key] == expected, f"Problem {key}: {problem[key]} != {expected}"
        )
    for tensor in "abcd":
        for option, index in ((f"--ld{tensor}", 1), (f"--stride_{tensor}", 2)):
            if option in case["args"]:
                expected = int(case["args"][case["args"].index(option) + 1])
                require(
                    problem[f"strides_{tensor}"][index] == expected,
                    f"Explicit {option} was not preserved in the generated problem",
                )
    document = yaml.safe_load(config.read_text())
    problem_type = document["BenchmarkProblems"][0][0]
    for key, expected in case.get("problem_type", {}).items():
        require(
            problem_type[key] == expected,
            f"ProblemType {key}: {problem_type[key]} != {expected}",
        )
    fork = document["BenchmarkProblems"][0][1]["ForkParameters"]
    parameters = {key: values[0] for item in fork for key, values in item.items()}
    for key, value in accepted["parameters"].items():
        require(
            parameters[key] == value, f"Resolved YAML does not preserve predicted {key}"
        )
    request_path = config.with_suffix(".request.json")
    request = json.loads(request_path.read_text())
    require(
        request["candidates"] == ranked,
        "Ranking changed between C++ request and manifest",
    )
    for relative in manifest["code_objects"] + [manifest["library"]["path"]]:
        artifact = (manifest_path.parent / relative).resolve(strict=True)
        require(
            artifact.is_relative_to(manifest_path.parent), "Nonlocal generated artifact"
        )
        require(artifact.stat().st_size > 0, "Empty generated artifact")
    return {
        "recipe": str(config),
        "manifest": str(manifest_path),
        "candidate": prediction["candidate_id"],
        "ranked_count": len(ranked),
        "architecture": architecture,
    }


def make_case(
    name,
    api,
    dtype,
    m,
    n,
    k,
    extra=(),
    batch=1,
    ta=False,
    tb=False,
    amax=False,
    problem_type=None,
):
    problem = dict(
        m=m,
        n=n,
        k=0 if name == "float-alpha-zero" else k,
        batch=batch,
        transpose_a=ta,
        transpose_b=tb,
    )
    type_id = {"s": 0, "h": 4, "b": 7, "f8": 15, "b8": 16, "i8": 8, "c": 2, "f4": 21}[
        dtype
    ]
    expected = dict(
        DataTypeA=type_id,
        DataTypeB=type_id,
        DestDataType=0 if name == "half-c-default" else type_id,
        ComputeDataType={"i8": 6, "c": 2}.get(dtype, 0),
        OutputAmaxD=amax,
    )
    expected.update(problem_type or {})
    args = ["--api_method", api, "-m", str(m), "-n", str(n), "-k", str(k)]
    # Omitting precision verifies the JIT-specific FP16 input / FP32 output defaults.
    if name != "half-c-default":
        precision = {
            "h": "f16_r",
            "s": "f32_r",
            "b": "bf16_r",
            "f8": "f8_r",
            "b8": "bf8_r",
            "i8": "i8_r",
            "c": "f32_c",
            "f4": "f4_r",
        }[dtype]
        args += [
            "-r",
            precision,
            "--compute_type",
            "i32_r" if dtype == "i8" else "f32_r",
        ]
    return dict(
        name=name,
        args=args + list(extra),
        problem=problem,
        dtype=dtype,
        problem_type=expected,
    )


CASES = [
    make_case("half-c-default", "c", "h", 128, 128, 128),
    make_case("half-explicit-precision", "c", "h", 128, 128, 128),
    make_case(
        "half-float-output",
        "mix",
        "h",
        128,
        96,
        128,
        ["--c_type", "f32_r", "--d_type", "f32_r"],
        problem_type={"DestDataType": 0},
    ),
    make_case("bfloat16", "cpp", "b", 128, 96, 128),
    make_case(
        "float8-float-output",
        "c",
        "f8",
        128,
        96,
        128,
        ["--c_type", "f32_r", "--d_type", "f32_r"],
        problem_type={"DestDataType": 0},
    ),
    make_case(
        "mixed-float8-float-output",
        "mix",
        "f8",
        128,
        96,
        128,
        ["--b_type", "bf8_r", "--c_type", "f32_r", "--d_type", "f32_r"],
        problem_type={"DataTypeB": 16, "DestDataType": 0},
    ),
    make_case(
        "int8-int32-output",
        "cpp",
        "i8",
        128,
        96,
        128,
        ["--c_type", "i32_r", "--d_type", "i32_r"],
        problem_type={"DestDataType": 6},
    ),
    make_case("complex-nn", "c", "c", 64, 48, 64),
    make_case(
        "complex-conjugate",
        "c",
        "c",
        64,
        48,
        64,
        ["--transA", "C"],
        ta=True,
        problem_type={"ComplexConjugateA": True, "ComplexConjugateB": False},
    ),
    make_case(
        "mxfp4-float-output",
        "c",
        "f4",
        128,
        128,
        256,
        [
            "--transA",
            "T",
            "--c_type",
            "f32_r",
            "--d_type",
            "f32_r",
            "--scaleA",
            "3",
            "--scaleB",
            "3",
            "--initialization",
            "uniform_low_precision",
        ],
        ta=True,
        problem_type={
            "DestDataType": 0,
            "MXBlockA": 32,
            "MXBlockB": 32,
            "DataTypeMXSA": 22,
            "DataTypeMXSB": 22,
        },
    ),
    make_case(
        "mxfp8-float-output",
        "mix",
        "f8",
        128,
        128,
        256,
        [
            "--transA",
            "T",
            "--c_type",
            "f32_r",
            "--d_type",
            "f32_r",
            "--scaleA",
            "3",
            "--scaleB",
            "3",
            "--initialization",
            "uniform_low_precision",
        ],
        ta=True,
        problem_type={
            "DestDataType": 0,
            "MXBlockA": 32,
            "MXBlockB": 32,
            "DataTypeMXSA": 22,
            "DataTypeMXSB": 22,
        },
    ),
    make_case("float-zero-m", "c", "s", 0, 96, 64),
    make_case("float-zero-n", "cpp", "s", 128, 0, 64),
    make_case("float-zero-k", "c", "s", 128, 96, 0),
    make_case("float-alpha-zero", "cpp", "s", 128, 96, 64),
    make_case(
        "half-bias",
        "c",
        "h",
        128,
        96,
        128,
        ["--bias_vector"],
        problem_type={"UseBias": 1},
    ),
    make_case(
        "half-relu",
        "mix",
        "h",
        128,
        96,
        128,
        ["--activation_type", "relu"],
        problem_type={"Activation": True, "ActivationType": "hipblaslt_all"},
    ),
    make_case(
        "half-gelu-aux",
        "cpp",
        "h",
        128,
        96,
        128,
        ["--activation_type", "gelu", "--use_e"],
        problem_type={
            "Activation": True,
            "ActivationType": "hipblaslt_all",
            "UseE": True,
        },
    ),
    make_case(
        "half-bias-relu-scaled-amax",
        "c",
        "h",
        128,
        96,
        128,
        [
            "--bias_vector",
            "--activation_type",
            "relu",
            "--scaleC",
            "1",
            "--scaleD",
            "1",
            "--amaxD",
        ],
        amax=True,
        problem_type={
            "UseBias": 1,
            "Activation": True,
            "ActivationType": "hipblaslt_all",
            "UseScaleCD": True,
        },
    ),
    make_case(
        "float-scale-ab",
        "c",
        "s",
        128,
        96,
        128,
        ["--scaleA", "1", "--scaleB", "1"],
        problem_type={"UseScaleAB": "Scalar"},
    ),
    make_case(
        "half-scale-alpha",
        "mix",
        "h",
        128,
        96,
        128,
        ["--scaleAlpha_vector"],
        problem_type={"UseScaleAlphaVec": 1},
    ),
    make_case(
        "float-scale-cd-amax",
        "c",
        "s",
        128,
        96,
        64,
        ["--scaleC", "1", "--scaleD", "1", "--amaxD"],
        amax=True,
        problem_type={"UseScaleCD": True},
    ),
    make_case("float-c", "c", "s", 128, 96, 64),
    make_case("half-mix", "mix", "h", 256, 128, 512),
    make_case("float-cpp", "cpp", "s", 128, 128, 64),
    make_case("half-amax-c", "c", "h", 128, 96, 64, ["--amaxD"], amax=True),
    make_case("float-amax-cpp", "cpp", "s", 128, 96, 64, ["--amaxD"], amax=True),
    make_case(
        "half-amax-odd",
        "mix",
        "h",
        129,
        97,
        65,
        ["--amaxD", "--lda", "137", "--ldb", "72", "--ldc", "135", "--ldd", "139"],
        amax=True,
    ),
    make_case(
        "half-odd",
        "c",
        "h",
        129,
        97,
        65,
        ["--lda", "137", "--ldb", "72", "--ldc", "135", "--ldd", "139"],
    ),
    make_case(
        "half-nt", "mix", "h", 80, 65, 48, ["--transB", "T", "--ldb", "73"], tb=True
    ),
    make_case(
        "float-tn", "cpp", "s", 96, 81, 63, ["--transA", "T", "--lda", "70"], ta=True
    ),
    make_case(
        "half-tt",
        "c",
        "h",
        65,
        79,
        96,
        ["--transA", "T", "--transB", "T", "--lda", "104", "--ldb", "87"],
        ta=True,
        tb=True,
    ),
    make_case(
        "float-batch",
        "mix",
        "s",
        64,
        96,
        32,
        [
            "--batch_count",
            "3",
            "--lda",
            "70",
            "--ldb",
            "40",
            "--ldc",
            "72",
            "--ldd",
            "76",
            "--stride_a",
            "2256",
            "--stride_b",
            "3856",
            "--stride_c",
            "6928",
            "--stride_d",
            "7312",
        ],
        batch=3,
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument(
        "--python", type=Path, required=True, help="Configured existing venv Python"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New scratch result directory"
    )
    parser.add_argument(
        "--case", action="append", choices=[case["name"] for case in CASES]
    )
    parser.add_argument("--feature-off", action="store_true")
    parser.add_argument("--negative-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--architecture",
        choices=("gfx90a", "gfx942", "gfx950", "gfx1250"),
        default="gfx950",
        help="Expected real GPU architecture; does not override the runtime device",
    )
    args = parser.parse_args()
    bench, build = args.bench.resolve(strict=True), args.build_root.resolve(strict=True)
    require(
        bench.is_relative_to(build), "Benchmark must come from the existing local build"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    empty_library = output / "empty-device-library"
    empty_library.mkdir()
    env = dict(os.environ)
    for name in list(env):
        if name.startswith("HIPBLASLT_JIT_"):
            del env[name]
    for name in ("HIPBLASLT_TUNING_FILE", "HIPBLASLT_TUNING_USER_MAX_WORKSPACE"):
        env.pop(name, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TENSILE_DISABLE_HELPER_CACHE"] = "1"
    # Explicit override disables lookup of installed project device libraries.
    env["HIPBLASLT_TENSILE_LIBPATH"] = str(empty_library)
    linkage = subprocess.run(
        ["ldd", str(bench)], env=env, text=True, capture_output=True, check=True
    )
    (output / "ldd.txt").write_text(linkage.stdout)
    libraries = re.findall(
        r"(lib(?:hipblaslt|tensilelite)[^\s]*) => (\S+)", linkage.stdout, re.I
    )
    require(
        any("hipblaslt" in name for name, _ in libraries),
        "No dynamic local hipBLASLt linkage",
    )
    for name, path in libraries:
        require(
            Path(path).resolve().is_relative_to(build),
            f"Installed project library: {name}: {path}",
        )

    wrapper = output / "trace-python"
    # The wrapper records every generator invocation and can inject a timing sentinel.
    wrapper.write_text(
        "#!"
        + str(args.python.absolute())
        + "\n"
        + """import json, os, subprocess, sys, time
path = os.environ['HIPBLASLT_JIT_TRACE']
start = time.monotonic_ns()
time.sleep(float(os.environ.get('HIPBLASLT_JIT_TEST_DELAY', '0')))
status = subprocess.call([os.environ['HIPBLASLT_JIT_REAL_PYTHON'], *sys.argv[1:]])
with open(path, 'a') as stream:
    stream.write(json.dumps(dict(args=sys.argv[1:], start_ns=start,
                                 end_ns=time.monotonic_ns(), status=status)) + '\\n')
sys.exit(status)
"""
    )
    wrapper.chmod(0o700)
    env["HIPBLASLT_JIT_PYTHON"] = str(wrapper)
    env["HIPBLASLT_JIT_REAL_PYTHON"] = str(args.python.absolute())
    report = []

    def run(name, options, extra_env=None):
        case_dir = output / name
        case_dir.mkdir()
        case_env = dict(env, HIPBLASLT_JIT_TRACE=str(case_dir / "generator.jsonl"))
        case_env.update(extra_env or {})
        command = [str(bench), *options]
        (case_dir / "command.json").write_text(json.dumps(command, indent=2))
        started = time.monotonic()
        proc = subprocess.run(
            command, env=case_env, text=True, capture_output=True, timeout=args.timeout
        )
        (case_dir / "stdout.txt").write_text(proc.stdout)
        (case_dir / "stderr.txt").write_text(proc.stderr)
        return proc, case_dir, time.monotonic() - started

    if args.feature_off:
        proc, path, _ = run("feature-off", ["--jit-gemm"])
        require(
            proc.returncode != 0 and "HIPBLASLT_ENABLE_JIT=ON" in proc.stderr,
            "Feature-off build did not reject JIT explicitly",
        )
        require(
            not (path / "generator.jsonl").exists(), "Feature-off invoked generation"
        )
        print("PASS feature-off rejection")
        return

    for name, options, diagnostic in [
        ("all", ["--algo_method", "all"], "one generated solution"),
        ("index", ["--algo_method", "index"], "one generated solution"),
        ("solution-index", ["--solution_index", "0"], "one generated solution"),
        ("multiple", ["--requested_solution", "2"], "one generated solution"),
        ("splitk", ["--api_method", "mix", "--splitk", "2"], "tuning"),
        ("wgm", ["--api_method", "mix", "--wgm", "2"], "tuning"),
        ("grouped", ["--grouped_gemm"], "non-grouped strided"),
        ("pointer-array", ["--batch_mode", "1"], "non-grouped strided"),
    ]:
        proc, path, _ = run("negative-" + name, ["--jit-gemm", *options])
        require(
            proc.returncode != 0 and diagnostic in proc.stderr,
            f"Conflict not rejected: {name}",
        )
        require(
            not (path / "generator.jsonl").exists(),
            f"Conflict generated kernels: {name}",
        )
        report.append({"case": name, "pass": True})

    tuning = output / "must-not-create-tuning.txt"
    proc, path, _ = run(
        "negative-tuning", ["--jit-gemm"], {"HIPBLASLT_TUNING_FILE": str(tuning)}
    )
    require(
        proc.returncode != 0 and "HIPBLASLT_TUNING_FILE" in proc.stderr,
        "Tuning conflict ignored",
    )
    require(not tuning.exists(), "Rejected JIT invocation wrote the tuning file")
    require(
        not (path / "generator.jsonl").exists(), "Tuning conflict invoked generation"
    )
    report.append({"case": "tuning-no-file-write", "pass": True})

    proc, path, _ = run(
        "negative-output-without-jit", ["--jit-output-dir", str(output / "unused")]
    )
    require(
        proc.returncode != 0 and "requires --jit-gemm" in proc.stderr,
        "Output option was silently ignored without JIT",
    )
    require(
        not (output / "unused").exists(), "Invalid invocation created JIT artifacts"
    )
    report.append({"case": "output-without-jit", "pass": True})

    if args.architecture == "gfx950" and not args.negative_only:
        natural = next(case for case in CASES if case["name"] == "mxfp4-float-output")
        natural_root = output / "natural-mx-artifacts"
        proc, path, _ = run(
            "negative-natural-mx-layout",
            ["--jit-gemm", "--jit-output-dir", str(natural_root), *natural["args"]],
        )
        require(
            proc.returncode != 0
            and "all supplied recipes were rejected" in proc.stderr
            and "gfx950 MX requires UseSubtileImpl" in proc.stderr,
            "Natural gfx950 MX request did not report the missing modeled subtile parameters",
        )
        trace = [
            json.loads(line)
            for line in (path / "generator.jsonl").read_text().splitlines()
        ]
        require(
            len(trace) == 1 and trace[0]["status"] != 0,
            "Expected provider validation failure",
        )
        require(
            not list(natural_root.glob("**/manifest.json")),
            "Unsupported MX layout built a bundle",
        )
        (request_file,) = natural_root.rglob("*.request.json")
        request = json.loads(request_file.read_text())
        require(
            all(
                request["problem"][f"scale_mode_{tensor}"] == "Block_32_UE8M0"
                for tensor in "ab"
            ),
            "Natural scale descriptors changed before validation",
        )
        report.append({"case": "natural-mx-no-supported-recipe", "pass": True})

    if not args.negative_only:
        for case in CASES:
            if args.case and case["name"] not in args.case:
                continue
            if case["problem_type"].get("MXBlockA") and args.architecture not in (
                "gfx950",
                "gfx1250",
            ):
                continue  # MX instructions require these architectures.
            if case["dtype"] in ("f8", "b8") and args.architecture == "gfx90a":
                continue  # gfx90a has no FP8 instructions.
            case = dict(
                case, args=list(case["args"]), problem_type=dict(case["problem_type"])
            )
            if case["problem_type"].get("MXBlockA") and args.architecture == "gfx950":
                # Tensile gfx950 consumes the public pre-swizzled scale layout.
                for option in ("--scaleA", "--scaleB"):
                    case["args"][case["args"].index(option) + 1] = "1001"
            if args.architecture == "gfx942":
                for key in ("DataTypeA", "DataTypeB"):
                    case["problem_type"][key] = {15: 11, 16: 12}.get(
                        case["problem_type"][key], case["problem_type"][key]
                    )
            artifact_root = output / (case["name"] + "-artifacts")
            sentinel = case["name"] == "half-c-default"
            options = [
                "--jit-gemm",
                "--jit-output-dir",
                str(artifact_root),
                *case["args"],
                "--alpha",
                (
                    "0"
                    if case["name"] == "float-alpha-zero"
                    else "2" if case["dtype"] == "i8" else "1.25"
                ),
                "--beta",
                "1" if case["dtype"] == "i8" else "0.5",
                "--verify",
                "--iters",
                "3",
                "--cold_iters",
                "1",
                "--print_kernel_info",
            ]
            if not sentinel:
                options.append("--use_gpu_timer")
            proc, path, elapsed = run(
                case["name"],
                options,
                {"HIPBLASLT_JIT_TEST_DELAY": "2" if sentinel else "0"},
            )
            no_ranking = case["name"] in {
                "complex-nn",
                "complex-conjugate",
                "float-zero-k",
                "float-alpha-zero",
                "mixed-float8-float-output",
            }
            all_invalid = args.architecture == "gfx950" and case["name"] in {
                "mxfp4-float-output",
                "mxfp8-float-output",
            }
            if no_ranking or all_invalid:
                require(
                    proc.returncode != 0,
                    f'{case["name"]} silently substituted a recipe',
                )
                diagnostic = (
                    "No Origami ranking: no finite positive-latency candidates"
                    if no_ranking
                    else "all supplied recipes were rejected"
                )
                require(
                    diagnostic in proc.stderr,
                    f'{case["name"]}: wrong failure; see {path}',
                )
                require(
                    not list(artifact_root.rglob("manifest.json")),
                    "Rejected request built a bundle",
                )
                require(
                    not list(artifact_root.rglob("solution.yaml")),
                    "Rejected request selected a recipe",
                )
                require(
                    "norm_error" not in proc.stdout,
                    "Rejected request entered timed execution",
                )
                if no_ranking:
                    require(
                        not (path / "generator.jsonl").exists(),
                        "Unranked request invoked generation",
                    )
                    require(
                        not list(artifact_root.rglob("*.request.json")),
                        "Unranked request emitted candidates",
                    )
                else:
                    trace = [
                        json.loads(line)
                        for line in (path / "generator.jsonl").read_text().splitlines()
                    ]
                    require(
                        len(trace) == 1 and trace[0]["status"] != 0,
                        "Expected one failed validation",
                    )
                    (request_file,) = artifact_root.rglob("*.request.json")
                    request = json.loads(request_file.read_text())
                    require(
                        request["model"] == "origami.gemm.estimation"
                        and request["candidates"],
                        "Failure lost the genuine Origami ranking",
                    )
                    generator_log = request_file.with_name("solution.log").read_text()
                    require(
                        all(
                            f"{candidate['id']}:" in generator_log
                            for candidate in request["candidates"]
                        ),
                        "Failure did not report every supplied candidate rejection",
                    )
                report.append(
                    dict(
                        case=case["name"],
                        pass_=True,
                        diagnostic=diagnostic,
                        elapsed_seconds=elapsed,
                    )
                )
                print(f'PASS {case["name"]}: {diagnostic}', flush=True)
                continue
            require(proc.returncode == 0, f'{case["name"]} failed; see {path}')
            if not case["problem"]["m"] or not case["problem"]["n"]:
                require(
                    "empty output; no kernel generation or launch" in proc.stderr,
                    f"Missing empty-output result: {path}",
                )
                require(
                    not (path / "generator.jsonl").exists(),
                    "Empty output invoked generation",
                )
                require(
                    not artifact_root.exists(),
                    "Empty output created generation artifacts",
                )
                report.append(
                    dict(case=case["name"], pass_=True, elapsed_seconds=elapsed)
                )
                print(f'PASS {case["name"]}: no generation or launch', flush=True)
                continue
            numeric = check_numerics(proc.stdout)
            provenance = check_provenance(
                proc.stderr, proc.stdout, case, artifact_root, args.architecture
            )
            trace = [
                json.loads(line)
                for line in (path / "generator.jsonl").read_text().splitlines()
            ]
            require(
                len(trace) == 1,
                f"Expected one generation across warmup/timing: {trace}",
            )
            require(
                trace[0]["args"][:2] == ["-m", "Tensile.JitGemm"],
                "Wrong generator entry point",
            )
            require(trace[0]["status"] == 0, "Generator failed")
            if sentinel:
                require(
                    (trace[0]["end_ns"] - trace[0]["start_ns"]) >= 2e9,
                    "Timing sentinel missing",
                )
                # A 2s compile delay must not appear in the three measured CPU-timed calls.
                require(
                    max(float(row["us"]) for row in numeric) * 3 < 500_000,
                    "Generation delay leaked into normal benchmark timing",
                )
            report.append(
                dict(
                    case=case["name"],
                    pass_=True,
                    elapsed_seconds=elapsed,
                    rows=numeric,
                    **provenance,
                )
            )
            print(
                f'PASS {case["name"]}: generation=1, finite verified result, local ranked artifact',
                flush=True,
            )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS {len(report)} checks; evidence in {output}")


if __name__ == "__main__":
    main()
