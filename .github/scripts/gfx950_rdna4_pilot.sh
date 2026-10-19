#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
set -euo pipefail

mkdir -p parity
case "${1:?install, preflight, or run required}" in
  install)
    root="${RUNNER_TEMP:-/tmp}/rocm/${ROCM_VERSION}"
    mkdir -p "$root"
    base=https://nightly.repo.amd.com/rocm/core/tarball
    df -h | tee parity/disk-before.txt
    for kind in runtime tests; do
      suffix=""
      if [ "$kind" = tests ]; then suffix="-tests"; fi
      url="$base/therock-dist-linux-gfx950-dcgpu${suffix}-${ROCM_VERSION}.tar.gz"
      printf '%s\n' "$url" >> parity/package-urls.txt
      curl -fL --retry 3 "$url" | tar -xzf - -C "$root"
    done
    test -x "$root/bin/rocminfo"
    echo "ROCM_ROOT=$root" >> "$GITHUB_ENV"
    df -h | tee parity/disk-after.txt
    ;;
  preflight)
    export ROCM_PATH="$ROCM_ROOT"
    export PATH="$ROCM_ROOT/bin:$PATH"
    export LD_LIBRARY_PATH="$ROCM_ROOT/lib:$ROCM_ROOT/lib/llvm/lib:$ROCM_ROOT/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    uname -a > parity/kernel.txt
    lscpu > parity/cpu.txt
    ls -l /dev/kfd /dev/dri/ > parity/gpu-devices.txt
    timeout 120 "$ROCM_ROOT/bin/rocminfo" | tee parity/native-rocminfo.txt
    python3 - <<'PY'
import json, os, pathlib, re
out = pathlib.Path('parity')
native_info = (out / 'native-rocminfo.txt').read_text()
names = re.findall(r'^\s*Name:\s+(gfx\w+)\s*$', native_info, re.M)
mode = os.environ['PILOT_MODE']
manifest = {
    'mode': mode, 'component': os.environ['COMPONENT'],
    'rocm_version': os.environ['ROCM_VERSION'],
    'label_regex': os.environ['TEST_LABEL'],
    'parallel': int(os.environ['TEST_PARALLEL']),
    'rocjitsu_source': os.environ['ROCM_SYSTEMS_REF'],
    'workflow_sha': os.environ['GITHUB_SHA'],
    'runner': os.environ.get('RUNNER_NAME'),
    'runner_label': 'linux-gfx120X-gpu-rocm' if mode == 'rdna4-dbt' else 'linux-gfx950-1gpu-ccs-ossci-rocm',
    'native_gpu_names': names,
    'visibility': {k: os.environ.get(k) for k in ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')},
    'execution_backend': 'hardware',
    'guest_isa': 'gfx950' if mode == 'rdna4-dbt' else None,
}
if mode == 'rdna4-dbt':
    hosts = [name for name in names if name in ('gfx1200', 'gfx1201')]
    if not hosts:
        raise SystemExit(f'Expected RDNA4 host, found {names}')
    config = json.loads(pathlib.Path('build-artifact/guest_gfx950_on_gfx1201.json').read_text())
    config['dbt_guest']['host_isa'] = hosts[0]
    config['dbt_guest']['execution_backend'] = 'hardware'
    wait = os.environ['WAIT_CHECKING']
    if wait == 'off':
        config.update(num_threads=8, cpu_dispatch_threads=170, async_helper_threads=0)
    (out / 'guest-gfx950-on-rdna4.json').write_text(json.dumps(config, indent=2) + '\n')
    manifest.update(host_isa=hosts[0], wait_checking_argument=wait,
                    requested_cpu_profile={'functional_quantum': 0, 'cpu_dispatch_threads': 170} if wait == 'off' else 'default',
                    cpu_profile_effective=False,
                    wait_checking_affects_hardware=False,
                    profile_note='CPU scheduling/quantum and simulator wait diagnostics do not apply to hardware DBT. No functional_quantum is added because this config has no simulated compute units.')
elif 'gfx950' not in names:
    raise SystemExit(f'Expected native gfx950, found {names}')
(out / 'run-config.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
    args=(--direct --rocm-root "$ROCM_ROOT" --out-dir parity --component "$COMPONENT" --label-regex "$TEST_LABEL" --list-only)
    python3 .github/scripts/ffm_quick_rocjitsu.py "${args[@]}" > parity/selection.txt
    if [ "$PILOT_MODE" = rdna4-dbt ]; then
      test -x rj-build/tools/rocjitsu/rocjitsu
      test -f rj-build/librocjitsu.so
      test -f rj-build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_hooks.so
      ldd rj-build/tools/rocjitsu/rocjitsu > parity/launcher-ldd.txt
      timeout 120 "$PWD/rj-build/tools/rocjitsu/rocjitsu" \
        --config "$PWD/parity/guest-gfx950-on-rdna4.json" \
        "--wait-checking=$WAIT_CHECKING" -- "$ROCM_ROOT/bin/rocminfo" \
        2>&1 | tee parity/guest-rocminfo.txt
      grep -Eq 'Name:[[:space:]]+gfx950[[:space:]]*$' parity/guest-rocminfo.txt
    fi
    ;;
  run)
    args=(--rocm-root "$ROCM_ROOT" --out-dir parity --component "$COMPONENT"
          --label-regex "$TEST_LABEL" --parallel "$TEST_PARALLEL")
    if [ "$PILOT_MODE" = native-gfx950 ]; then
      args+=(--direct)
    else
      args+=(--rocjitsu "$PWD/rj-build/tools/rocjitsu/rocjitsu"
             --config "$PWD/parity/guest-gfx950-on-rdna4.json"
             "--rocjitsu-arg=--wait-checking=$WAIT_CHECKING")
    fi
    export LD_LIBRARY_PATH="$ROCM_ROOT/lib/llvm/lib:$ROCM_ROOT/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    set +e
    python3 .github/scripts/resource_monitor.py \
      --phase "$PILOT_MODE ${WAIT_CHECKING:-native} / $COMPONENT" \
      --output parity/resources-test.json -- \
      python3 .github/scripts/ffm_quick_rocjitsu.py "${args[@]}"
    status=$?
    if [ -f parity/summary.md ]; then cat parity/summary.md >> "$GITHUB_STEP_SUMMARY"; fi
    exit "$status"
    ;;
  *) echo "Unknown phase: $1" >&2; exit 2 ;;
esac
