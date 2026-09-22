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
        if 'norm_error' not in line or ',atol' not in line or ',rtol' not in line:
            continue
        header = re.sub(r'^\s*\[\d+\]:', '', line).strip()
        fields = next(csv.reader([header]))
        values = next(csv.reader([lines[index + 1].strip()]))
        require(len(fields) == len(values), 'Malformed benchmark result row')
        found.append(dict(zip(fields, values)))
    require(found, 'No benchmark correctness CSV row found')
    return found


def check_numerics(stdout, dtype):
    result = rows(stdout)
    bound = 5e-3 if dtype == 'h' else 1e-5
    for row in result:
        require(all(value.strip().lower() != 'failed' for value in row.values()),
                'Benchmark reported failed allclose')
        for field in ('norm_error', 'atol', 'rtol', 'us'):
            value = float(row[field])
            require(math.isfinite(value) and value >= 0, f'Invalid {field}: {value}')
        require(float(row['norm_error']) <= bound,
                f'Norm error {row["norm_error"]} exceeds {bound}')
        require(float(row['atol']) < 1 and float(row['rtol']) < 1,
                'Allclose tolerances were not resolved')
    return result


def check_provenance(stderr, stdout, case, artifact_root, architecture):
    import yaml  # Use the existing configured venv (already needed by Tensile).

    def reported(label):
        matches = re.findall(r'^' + re.escape(label) + r': (.+)$', stderr, re.MULTILINE)
        require(len(matches) == 1, f'Expected one {label}, got {len(matches)}')
        path = Path(matches[0]).resolve(strict=True)
        require(path.is_relative_to(artifact_root), f'Artifact escaped test output: {path}')
        return path

    config = reported('JIT recipe')
    manifest_path = reported('JIT manifest')
    require('JIT recipe:' not in stdout and 'JIT manifest:' not in stdout,
            'JIT diagnostics polluted benchmark stdout')
    manifests = list(artifact_root.rglob('manifest.json'))
    require(manifests == [manifest_path], f'Expected one generated bundle: {manifests}')
    manifest = json.loads(manifest_path.read_text())
    require(manifest['counts']['solutions'] == 1, 'Generated more than one solution')
    require(manifest['counts']['main_kernels'] == 1, 'Generated more than one main kernel')
    require(manifest['main_kernel']['name'] in stdout, 'Reported kernel differs from artifact')
    require(manifest['architecture']['resolved'].split(':')[0] == architecture, 'Wrong device ISA')
    prediction = manifest['jit_prediction']
    require(prediction['model'] == 'origami.gemm.estimation', 'Missing Origami model provenance')
    ranked = prediction['ranked_candidates']
    require(len(ranked) > 1, 'No actual candidate ranking was recorded')
    for candidate in ranked:
        score = candidate['predicted_cycles']
        require(math.isfinite(score) and 0 < score < 1e300, 'Invalid Origami score')
        parameters = candidate['parameters']
        mi = parameters['MatrixInstruction']
        permitted_k = {4} if case['dtype'] == 's' else ({16, 32} if architecture == 'gfx950' else {16})
        require(mi[:2] == [16, 16] and mi[2] in permitted_k,
                f'Unsupported {architecture} matrix instruction: {mi}')
        if architecture == 'gfx90a':
            require(parameters['NonTemporalA'] == parameters['NonTemporalB'] == 0,
                    'gfx90a recipe uses unsupported cache hints')
    # Origami can reorder near-equal scores using tie-breaking. Preserve its rank order.
    rejected = {item['candidate_id'] for item in prediction['rejections']}
    accepted = next(candidate for candidate in ranked if candidate['id'] not in rejected)
    require(prediction['candidate_id'] == accepted['id'], 'Not the first valid ranked candidate')
    require(prediction['predicted_cycles'] == accepted['predicted_cycles'], 'Score changed')
    require(prediction['selected_parameters'] == accepted['parameters'], 'Candidate changed')
    require(prediction['defaults_source'], 'No default provenance')
    problem = prediction['problem']
    for key, expected in case['problem'].items():
        require(problem[key] == expected, f'Problem {key}: {problem[key]} != {expected}')
    for tensor in 'abcd':
        for option, index in ((f'--ld{tensor}', 1), (f'--stride_{tensor}', 2)):
            if option in case['args']:
                expected = int(case['args'][case['args'].index(option) + 1])
                require(problem[f'strides_{tensor}'][index] == expected,
                        f'Explicit {option} was not preserved in the generated problem')
    document = yaml.safe_load(config.read_text())
    fork = document['BenchmarkProblems'][0][1]['ForkParameters']
    parameters = {key: values[0] for item in fork for key, values in item.items()}
    for key, value in accepted['parameters'].items():
        require(parameters[key] == value, f'Resolved YAML does not preserve predicted {key}')
    request_path = config.with_suffix('.request.json')
    request = json.loads(request_path.read_text())
    require(request['candidates'] == ranked, 'Ranking changed between C++ request and manifest')
    for relative in manifest['code_objects'] + [manifest['library']['path']]:
        artifact = (manifest_path.parent / relative).resolve(strict=True)
        require(artifact.is_relative_to(manifest_path.parent), 'Nonlocal generated artifact')
        require(artifact.stat().st_size > 0, 'Empty generated artifact')
    return {'recipe': str(config), 'manifest': str(manifest_path),
            'candidate': prediction['candidate_id'], 'ranked_count': len(ranked),
            'architecture': architecture}


def make_case(name, api, dtype, m, n, k, extra=(), batch=1, ta=False, tb=False, amax=False):
    problem = dict(m=m, n=n, k=k, batch=batch, transpose_a=ta, transpose_b=tb,
                   data_type=dtype, high_precision_accumulate=(dtype == 'h'), output_amax_d=amax)
    args = ['--api_method', api, '-m', str(m), '-n', str(n), '-k', str(k)]
    # The first case deliberately uses the bench's FP16 / FP32-compute defaults.
    if name != 'half-c-default':
        args += ['-r', 'f16_r' if dtype == 'h' else 'f32_r', '--compute_type', 'f32_r']
    return dict(name=name, args=args + list(extra), problem=problem, dtype=dtype)


CASES = [
    make_case('half-c-default', 'c', 'h', 128, 128, 128),
    make_case('float-c', 'c', 's', 128, 96, 64),
    make_case('half-mix', 'mix', 'h', 256, 128, 512),
    make_case('float-cpp', 'cpp', 's', 128, 128, 64),
    make_case('half-amax-c', 'c', 'h', 128, 96, 64, ['--amaxD'], amax=True),
    make_case('float-amax-cpp', 'cpp', 's', 128, 96, 64, ['--amaxD'], amax=True),
    make_case('half-amax-odd', 'mix', 'h', 129, 97, 65,
              ['--amaxD', '--lda', '137', '--ldb', '72', '--ldc', '135', '--ldd', '139'],
              amax=True),
    make_case('half-odd', 'c', 'h', 129, 97, 65,
              ['--lda', '137', '--ldb', '72', '--ldc', '135', '--ldd', '139']),
    make_case('half-nt', 'mix', 'h', 80, 65, 48, ['--transB', 'T', '--ldb', '73'], tb=True),
    make_case('float-tn', 'cpp', 's', 96, 81, 63, ['--transA', 'T', '--lda', '70'], ta=True),
    make_case('half-tt', 'c', 'h', 65, 79, 96,
              ['--transA', 'T', '--transB', 'T', '--lda', '104', '--ldb', '87'], ta=True, tb=True),
    make_case('float-batch', 'mix', 's', 64, 96, 32,
              ['--batch_count', '3', '--lda', '70', '--ldb', '40', '--ldc', '72', '--ldd', '76',
               '--stride_a', '2256', '--stride_b', '3856', '--stride_c', '6928', '--stride_d', '7312'],
              batch=3),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench', type=Path, required=True)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True, help='Configured existing venv Python')
    parser.add_argument('--output', type=Path, required=True, help='New scratch result directory')
    parser.add_argument('--case', action='append', choices=[case['name'] for case in CASES])
    parser.add_argument('--feature-off', action='store_true')
    parser.add_argument('--negative-only', action='store_true')
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--architecture', choices=('gfx90a', 'gfx942', 'gfx950'), default='gfx950',
                        help='Expected real GPU architecture; does not override the runtime device')
    args = parser.parse_args()
    bench, build = args.bench.resolve(strict=True), args.build_root.resolve(strict=True)
    require(bench.is_relative_to(build), 'Benchmark must come from the existing local build')
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    empty_library = output / 'empty-device-library'
    empty_library.mkdir()
    env = dict(os.environ)
    for name in list(env):
        if name.startswith('HIPBLASLT_JIT_'):
            del env[name]
    for name in ('HIPBLASLT_TUNING_FILE', 'HIPBLASLT_TUNING_USER_MAX_WORKSPACE'):
        env.pop(name, None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['TENSILE_DISABLE_HELPER_CACHE'] = '1'
    # Explicit override disables lookup of installed project device libraries.
    env['HIPBLASLT_TENSILE_LIBPATH'] = str(empty_library)
    linkage = subprocess.run(['ldd', str(bench)], env=env, text=True, capture_output=True, check=True)
    (output / 'ldd.txt').write_text(linkage.stdout)
    libraries = re.findall(r'(lib(?:hipblaslt|tensilelite)[^\s]*) => (\S+)', linkage.stdout, re.I)
    require(any('hipblaslt' in name for name, _ in libraries), 'No dynamic local hipBLASLt linkage')
    for name, path in libraries:
        require(Path(path).resolve().is_relative_to(build), f'Installed project library: {name}: {path}')

    wrapper = output / 'trace-python'
    # The wrapper records every generator invocation and can inject a timing sentinel.
    wrapper.write_text('#!' + str(args.python.absolute()) + '\n' + '''import json, os, subprocess, sys, time
path = os.environ['HIPBLASLT_JIT_TRACE']
start = time.monotonic_ns()
time.sleep(float(os.environ.get('HIPBLASLT_JIT_TEST_DELAY', '0')))
status = subprocess.call([os.environ['HIPBLASLT_JIT_REAL_PYTHON'], *sys.argv[1:]])
with open(path, 'a') as stream:
    stream.write(json.dumps(dict(args=sys.argv[1:], start_ns=start,
                                 end_ns=time.monotonic_ns(), status=status)) + '\\n')
sys.exit(status)
''')
    wrapper.chmod(0o700)
    env['HIPBLASLT_JIT_PYTHON'] = str(wrapper)
    env['HIPBLASLT_JIT_REAL_PYTHON'] = str(args.python.absolute())
    report = []

    def run(name, options, extra_env=None):
        case_dir = output / name
        case_dir.mkdir()
        case_env = dict(env, HIPBLASLT_JIT_TRACE=str(case_dir / 'generator.jsonl'))
        case_env.update(extra_env or {})
        command = [str(bench), *options]
        (case_dir / 'command.json').write_text(json.dumps(command, indent=2))
        started = time.monotonic()
        proc = subprocess.run(command, env=case_env, text=True, capture_output=True, timeout=args.timeout)
        (case_dir / 'stdout.txt').write_text(proc.stdout)
        (case_dir / 'stderr.txt').write_text(proc.stderr)
        return proc, case_dir, time.monotonic() - started

    if args.feature_off:
        proc, path, _ = run('feature-off', ['--jit-gemm'])
        require(proc.returncode != 0 and 'HIPBLASLT_ENABLE_JIT_GEMM=ON' in proc.stderr,
                'Feature-off build did not reject JIT explicitly')
        require(not (path / 'generator.jsonl').exists(), 'Feature-off invoked generation')
        print('PASS feature-off rejection')
        return

    for name, options, diagnostic in [
        ('all', ['--algo_method', 'all'], 'one generated solution'),
        ('index', ['--algo_method', 'index'], 'one generated solution'),
        ('solution-index', ['--solution_index', '0'], 'one generated solution'),
        ('multiple', ['--requested_solution', '2'], 'one generated solution'),
        ('splitk', ['--api_method', 'mix', '--splitk', '2'], 'tuning'),
        ('wgm', ['--api_method', 'mix', '--wgm', '2'], 'tuning'),
        ('grouped', ['--grouped_gemm'], 'non-grouped strided'),
        ('pointer-array', ['--batch_mode', '1'], 'non-grouped strided'),
    ]:
        proc, path, _ = run('negative-' + name, ['--jit-gemm', *options])
        require(proc.returncode != 0 and diagnostic in proc.stderr, f'Conflict not rejected: {name}')
        require(not (path / 'generator.jsonl').exists(), f'Conflict generated kernels: {name}')
        report.append({'case': name, 'pass': True})

    tuning = output / 'must-not-create-tuning.txt'
    proc, path, _ = run('negative-tuning', ['--jit-gemm'], {'HIPBLASLT_TUNING_FILE': str(tuning)})
    require(proc.returncode != 0 and 'HIPBLASLT_TUNING_FILE' in proc.stderr, 'Tuning conflict ignored')
    require(not tuning.exists(), 'Rejected JIT invocation wrote the tuning file')
    require(not (path / 'generator.jsonl').exists(), 'Tuning conflict invoked generation')
    report.append({'case': 'tuning-no-file-write', 'pass': True})

    proc, path, _ = run('negative-output-without-jit', ['--jit-output-dir', str(output / 'unused')])
    require(proc.returncode != 0 and 'requires --jit-gemm' in proc.stderr,
            'Output option was silently ignored without JIT')
    require(not (output / 'unused').exists(), 'Invalid invocation created JIT artifacts')
    report.append({'case': 'output-without-jit', 'pass': True})

    if not args.negative_only:
        for name, options in [('activation', ['--activation_type', 'relu']),
                              ('dtype', ['-r', 'bf16_r']),
                              ('scale-cd', ['--amaxD', '--scaleC', '1', '--scaleD', '1'])]:
            proc, path, _ = run('negative-' + name, ['--jit-gemm', *options])
            require(proc.returncode != 0 and 'JIT preparation failed' in proc.stderr,
                    f'Unsupported descriptor was not rejected: {name}')
            require(not (path / 'generator.jsonl').exists(),
                    f'Unsupported descriptor invoked generation: {name}')
            report.append({'case': name, 'pass': True})
        for case in CASES:
            if args.case and case['name'] not in args.case:
                continue
            artifact_root = output / (case['name'] + '-artifacts')
            sentinel = case['name'] == 'half-c-default'
            options = ['--jit-gemm', '--jit-output-dir', str(artifact_root), *case['args'],
                       '--alpha', '1.25', '--beta', '0.5', '--verify',
                       '--iters', '3', '--cold_iters', '1', '--print_kernel_info']
            if not sentinel:
                options.append('--use_gpu_timer')
            proc, path, elapsed = run(case['name'], options,
                                      {'HIPBLASLT_JIT_TEST_DELAY': '2' if sentinel else '0'})
            require(proc.returncode == 0, f'{case["name"]} failed; see {path}')
            numeric = check_numerics(proc.stdout, case['dtype'])
            provenance = check_provenance(proc.stderr, proc.stdout, case, artifact_root, args.architecture)
            trace = [json.loads(line) for line in (path / 'generator.jsonl').read_text().splitlines()]
            require(len(trace) == 1, f'Expected one generation across warmup/timing: {trace}')
            require(trace[0]['args'][:2] == ['-m', 'Tensile.JitGemm'], 'Wrong generator entry point')
            require(trace[0]['status'] == 0, 'Generator failed')
            if sentinel:
                require((trace[0]['end_ns'] - trace[0]['start_ns']) >= 2e9, 'Timing sentinel missing')
                # A 2s compile delay must not appear in the three measured CPU-timed calls.
                require(max(float(row['us']) for row in numeric) * 3 < 500_000,
                        'Generation delay leaked into normal benchmark timing')
            report.append(dict(case=case['name'], pass_=True, elapsed_seconds=elapsed,
                               rows=numeric, **provenance))
            print(f'PASS {case["name"]}: generation=1, finite verified result, local ranked artifact', flush=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS {len(report)} checks; evidence in {output}')


if __name__ == '__main__':
    main()
