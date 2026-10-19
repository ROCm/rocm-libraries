#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Compare identical installed gfx950 test selections on native and DBT hardware."""

import csv
import json
import os
import sys
from pathlib import Path


def compare(source: Path, destination: Path) -> str:
    destination.mkdir(parents=True, exist_ok=True)
    expected_path = source / 'pilot-discovery' / 'components.json'
    expected = json.loads(expected_path.read_text()) if expected_path.exists() else []
    measurements = {}
    for directory in source.glob('pilot-*'):
        manifest_path = directory / 'run-config.json'
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        key = (manifest['component'], manifest['mode'], manifest.get('wait_checking_argument', 'native'))
        summary_path = directory / 'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
        measurements[key] = (manifest, summary[0] if summary else None)

    components = sorted({entry['component'] for entry in expected} | {key[0] for key in measurements})
    lines = [
        '# gfx950 native vs RDNA4 binary translation', '',
        'Same gfx950 nightly, exact quick label, and requested test concurrency. Times below are component test wall time; package setup/build and CI queue time are excluded.', '',
        'The on/off flag controls simulator wait diagnostics and does not validate hardware wait behavior. Quantum and CPU dispatch settings are inactive in hardware DBT.', '',
        '| Component | Native gfx950 | RDNA4 wait on | Delta | RDNA4 wait off | Delta |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    if os.environ.get('NATIVE_BASELINE_RUN'):
        baseline = os.environ['NATIVE_BASELINE_RUN']
        lines[2:2] = [f'Native gfx950 measurements reused from [run {baseline}](https://github.com/ROCm/rocm-libraries/actions/runs/{baseline}); native suites were not rerun.', '']
    rows = []

    def cell(result):
        if result is None:
            return 'No result'
        counts = result['counts']
        return f"{result['seconds']:.1f}s; {counts['passed']}/{result['total']} pass, {counts['failed']} fail, {counts['skipped']} skip"

    for component in components:
        native_manifest, native = measurements.get((component, 'native-gfx950', 'native'), (None, None))
        row = {'component': component, 'native_seconds': native['seconds'] if native else None}
        cells = [component, cell(native)]
        for wait in ('on', 'off'):
            translated_manifest, translated = measurements.get((component, 'rdna4-dbt', wait), (None, None))
            delta = None
            if native and translated:
                native_status = {test['name']: test['status'] for test in native['tests']}
                translated_status = {test['name']: test['status'] for test in translated['tests']}
                same_settings = all(native_manifest.get(key) == translated_manifest.get(key)
                                    for key in ('rocm_version', 'label_regex', 'parallel'))
                comparable = (same_settings and native_status == translated_status and
                              native.get('effective_parallel') == translated.get('effective_parallel') and
                              all(status in ('passed', 'skipped') for status in native_status.values()) and
                              any(status == 'passed' for status in native_status.values()))
                if comparable and native['seconds'] > 0:
                    delta = (translated['seconds'] / native['seconds'] - 1) * 100
            cells.extend([cell(translated), f'{delta:+.1f}%' if delta is not None else 'Not comparable'])
            row[f'rdna4_wait_{wait}_seconds'] = translated['seconds'] if translated else None
            row[f'rdna4_wait_{wait}_delta_percent'] = delta
        lines.append('| ' + ' | '.join(cells) + ' |')
        rows.append(row)
    if not components:
        lines.extend(['', 'No component measurements were produced. Inspect discovery/build/preflight logs.'])
    lines.extend(['', 'A percentage is shown only when both legs use the same nightly, label, requested/effective concurrency and test names, and all statuses match as passing/skipped, with at least one passing test. This checks CTest entry outcomes; it does not establish per-case numerical parity.'])
    text = '\n'.join(lines) + '\n'
    (destination / 'comparison.md').write_text(text)
    (destination / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n')
    if rows:
        with (destination / 'comparison.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return text


if __name__ == '__main__':
    report = compare(Path(sys.argv[1]), Path(sys.argv[2]))
    print(report)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as handle:
            handle.write(report)
