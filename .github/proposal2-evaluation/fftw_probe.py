"""Bounded artifact portability experiment; does not qualify full ROCm reuse."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

PRODUCER = "38220501023"
OUTPUT = Path("probe-evidence").resolve()
OUTPUT.mkdir(exist_ok=True)
ROOT = Path(os.environ.get("RUNNER_TEMP", ".")) / "p2-fftw-probe"
ROOT.mkdir(exist_ok=True)
RAW = ROOT / "raw-extracted"
RAW.mkdir(exist_ok=True)
BASE = f"https://therock-ci-artifacts.s3.amazonaws.com/{PRODUCER}-linux"


def run(command, log_name):
    started = time.monotonic()
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (OUTPUT / log_name).write_text(result.stdout, encoding="utf-8")
    print(f"{log_name}: exit={result.returncode} seconds={time.monotonic()-started:.2f}", flush=True)
    print(result.stdout[-2000:], flush=True)
    return result.returncode


downloads = []
for category in ("dev", "lib"):
    name = f"fftw3_{category}_generic.tar.zst"
    url = f"{BASE}/{name}"
    data = urllib.request.urlopen(url).read()
    sidecar = urllib.request.urlopen(url + ".sha256sum").read().decode()
    expected = sidecar.split()[0]
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch for {url}")
    archive = OUTPUT / name
    archive.write_bytes(data)
    (OUTPUT / (name + ".sha256sum")).write_text(sidecar)
    subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(RAW)], check=True)
    downloads.append({"url": url, "bytes": len(data), "sha256": actual})

source = ROOT / "consumer"
source.mkdir(exist_ok=True)
(source / "CMakeLists.txt").write_text(r'''
cmake_minimum_required(VERSION 3.22)
project(fftw_relocation_probe C)
enable_testing()
find_package(Threads REQUIRED)
find_package(FFTW3 CONFIG REQUIRED PATHS "${DOUBLE}/lib/cmake/fftw3" NO_DEFAULT_PATH)
find_package(FFTW3f CONFIG REQUIRED PATHS "${FLOAT}/lib/cmake/fftw3f" NO_DEFAULT_PATH)
add_executable(probe probe.c)
if(MODE STREQUAL "legacy")
  if(NOT EXISTS "${FFTW3_INCLUDE_DIRS}/fftw3.h")
    message(FATAL_ERROR "Nonrelocatable FFTW3_INCLUDE_DIRS=${FFTW3_INCLUDE_DIRS}")
  endif()
  target_include_directories(probe PRIVATE ${FFTW3_INCLUDE_DIRS})
  target_link_directories(probe PRIVATE ${FFTW3_LIBRARY_DIRS})
  target_link_libraries(probe PRIVATE ${FFTW3_LIBRARIES})
elseif(MODE STREQUAL "double")
  target_link_libraries(probe PRIVATE FFTW3::fftw3)
elseif(MODE STREQUAL "float")
  target_compile_definitions(probe PRIVATE FLOAT_PRECISION=1)
  target_link_libraries(probe PRIVATE FFTW3::fftw3f)
elseif(MODE STREQUAL "double_threads")
  target_compile_definitions(probe PRIVATE THREADED=1)
  target_link_libraries(probe PRIVATE FFTW3::fftw3_threads)
elseif(MODE STREQUAL "float_threads")
  target_compile_definitions(probe PRIVATE FLOAT_PRECISION=1 THREADED=1)
  target_link_libraries(probe PRIVATE FFTW3::fftw3f_threads)
else()
  message(FATAL_ERROR "Unknown mode")
endif()
add_test(NAME numerical_round_trip COMMAND probe)
''')
(source / "probe.c").write_text(r'''
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#ifdef FLOAT_PRECISION
#define FN(name) fftwf_##name
#define COMPLEX fftwf_complex
#define PLAN fftwf_plan
#define TOLERANCE 0.0001
#else
#define FN(name) fftw_##name
#define COMPLEX fftw_complex
#define PLAN fftw_plan
#define TOLERANCE 0.000000001
#endif
int main(void) {
  COMPLEX input[8], transformed[8], output[8];
#ifdef THREADED
  if (!FN(init_threads)()) return 10;
  FN(plan_with_nthreads)(2);
#endif
  for (int i=0; i<8; ++i) { input[i][0]=i+1; input[i][1]=0; }
  PLAN forward=FN(plan_dft_1d)(8,input,transformed,FFTW_FORWARD,FFTW_ESTIMATE);
  PLAN backward=FN(plan_dft_1d)(8,transformed,output,FFTW_BACKWARD,FFTW_ESTIMATE);
  if (!forward || !backward) return 11;
  FN(execute)(forward); FN(execute)(backward);
  for (int i=0; i<8; ++i) {
    if (fabs(output[i][0]/8.0-input[i][0])>TOLERANCE || fabs(output[i][1])>TOLERANCE) return 12;
  }
  FN(destroy_plan)(forward); FN(destroy_plan)(backward);
#ifdef THREADED
  FN(cleanup_threads)();
#endif
  puts("PASS: actual FFTW forward/backward transform round trip");
  return 0;
}
''')


def repair_configs(root):
    """Candidate packaging transformation only, never a production source patch."""
    base = root / "third-party/fftw3"
    double = base / "build_fftw3/stage"
    floating = base / "build_fftw3f/stage"
    (floating / "include").mkdir(exist_ok=True)
    shutil.copy2(double / "include/fftw3.h", floating / "include/fftw3.h")
    for suffix, prefix in (("", double), ("f", floating)):
        package = f"FFTW3{suffix}"
        lib = f"fftw3{suffix}"
        config = prefix / f"lib/cmake/fftw3{suffix}/{package}Config.cmake"
        original = config.read_text()
        (OUTPUT / f"original-{package}Config.cmake").write_text(original)
        config.write_text(f'''
get_filename_component(_fftw_prefix "${{CMAKE_CURRENT_LIST_DIR}}/../../.." ABSOLUTE)
set({package}_LIBRARIES FFTW3::{lib})
set({package}_LIBRARY_DIRS "${{_fftw_prefix}}/lib")
set({package}_INCLUDE_DIRS "${{_fftw_prefix}}/include")
include("${{CMAKE_CURRENT_LIST_DIR}}/FFTW3LibraryDepends.cmake")
if(NOT TARGET FFTW3::{lib}_threads)
  find_package(Threads REQUIRED)
  add_library(FFTW3::{lib}_threads SHARED IMPORTED)
  set_target_properties(FFTW3::{lib}_threads PROPERTIES
    IMPORTED_LOCATION "${{_fftw_prefix}}/lib/lib{lib}_threads.so.3"
    INTERFACE_INCLUDE_DIRECTORIES "${{_fftw_prefix}}/include"
    INTERFACE_LINK_LIBRARIES "FFTW3::{lib};Threads::Threads")
endif()
''')
        (OUTPUT / f"candidate-{package}Config.cmake").write_text(config.read_text())


results = []
for variant in ("original", "candidate"):
    seed = ROOT / f"{variant}-initial-root"
    shutil.copytree(RAW, seed, symlinks=True)
    if variant == "candidate":
        repair_configs(seed)
    for location in ("initial", "relocated"):
        if location == "initial":
            tree = seed
        else:
            tree = ROOT / "another-parent" / f"{variant}-different-depth" / "artifacts"
            tree.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(seed, tree, symlinks=True)
            # Delete the first candidate root to ensure it cannot mask bad exports.
            if variant == "candidate":
                shutil.rmtree(seed)
        for mode in ("legacy", "double", "float", "double_threads", "float_threads"):
            label = f"{variant}-{location}-{mode}"
            build = ROOT / "builds" / label
            command = ["cmake", "-S", str(source), "-B", str(build),
                       f"-DDOUBLE={tree}/third-party/fftw3/build_fftw3/stage",
                       f"-DFLOAT={tree}/third-party/fftw3/build_fftw3f/stage", f"-DMODE={mode}"]
            config_rc = run(command, label + "-configure.log")
            build_rc = run(["cmake", "--build", str(build), "--verbose"], label + "-build.log") if config_rc == 0 else None
            test_rc = run(["ctest", "--test-dir", str(build), "--output-on-failure", "-V"], label + "-test.log") if build_rc == 0 else None
            results.append({"variant": variant, "location": location, "mode": mode,
                            "configure_exit": config_rc, "build_exit": build_rc, "test_exit": test_rc,
                            "passed": config_rc == 0 and build_rc == 0 and test_rc == 0})

summary = {"producer_run_id": PRODUCER, "downloads": downloads, "results": results,
           "scope": "Linux FFTW CPU artifact-export portability only; not real rocFFT/hipFFT clients, GPU coverage, full stage or savings validation"}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
if not all(row["passed"] for row in results if row["variant"] == "candidate"):
    raise SystemExit("Candidate failed a bounded FFTW consumer check")
