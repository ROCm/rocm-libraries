# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Wait until this run has uploaded the artifacts a rocjitsu sidecar fetches.

The sidecars only need build outputs, but a caller cannot depend on one job
inside TheRock's reusable Linux workflow. Polling the run's artifact store
lets them start as soon as their own family's outputs exist, including when
a stage was copied from a prebuilt baseline instead of rebuilt.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

EMULATION_ARTIFACTS = (
    "mirage_run_generic",
    "rocjitsu_lib_generic",
    "rocjitsu_run_generic",
    "rocjitsu-hotswap_lib_generic",
)
# A job that has not finished may still upload; anything else cannot.
ACTIVE_JOB_STATES = {"queued", "in_progress", "waiting", "pending", "requested"}
# Linux build jobs are created a little after this job starts.
NO_BUILD_JOB_GRACE_SECONDS = 1800


def math_libs_artifacts(target):
    return (
        f"blas_lib_{target}",
        f"blas_test_{target}",
        "blas_lib_generic",
        "blas_test_generic",
    )


def missing(names, exists):
    # The checksum is written after its archive, so it marks a complete upload.
    return [name for name in names if not exists(f"{name}.tar.zst.sha256sum")]


def wait(
    family,
    target,
    exists,
    prefix_exists,
    linux_jobs_active,
    timeout,
    poll,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Return None when every artifact exists, else the reason to give up."""
    stages = (
        ("math-libs", f"logs/math-libs/{family}/", math_libs_artifacts(target)),
        ("emulation", "logs/emulation/", EMULATION_ARTIFACTS),
    )
    start = clock()
    seen_build_job = False
    absent = {name: list(names) for name, _, names in stages}
    while True:
        waited = clock() - start
        try:
            absent = {name: missing(names, exists) for name, _, names in stages}
            if not any(absent.values()):
                return None
            for name, logs, _ in stages:
                # Stage logs are uploaded even on failure, after the artifacts.
                if (
                    absent[name]
                    and prefix_exists(logs)
                    and missing(absent[name], exists)
                ):
                    return f"{name} stage finished without {', '.join(absent[name])}"
            active = linux_jobs_active()
            seen_build_job = seen_build_job or active
            if not active and (seen_build_job or waited > NO_BUILD_JOB_GRACE_SECONDS):
                # One more look: the last job may have uploaded just before finishing.
                still = [n for names in absent.values() for n in missing(names, exists)]
                if still:
                    return f"no Linux build job is running and {', '.join(still)} never appeared"
                return None
        except OSError as error:
            # Transient S3 or GitHub API failures must not end a long wait.
            print(f"[{int(waited)}s] check failed, retrying: {error}", flush=True)
        if waited > timeout:
            flat = [n for names in absent.values() for n in names]
            return f"timed out after {int(waited)}s waiting for {', '.join(flat)}"
        print(f"[{int(waited)}s] waiting for: {absent}", flush=True)
        sleep(poll)


def s3_checks(base_url):
    def exists(key):
        request = urllib.request.Request(f"{base_url}/{key}", method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=30):
                return True
        except urllib.error.HTTPError as error:
            if error.code in (403, 404):
                return False
            raise

    bucket_url, _, prefix = base_url.partition(".s3.amazonaws.com/")
    bucket_url += ".s3.amazonaws.com"

    def prefix_exists(sub_prefix):
        query = f"?list-type=2&max-keys=1&prefix={prefix}/{sub_prefix}"
        with urllib.request.urlopen(bucket_url + "/" + query, timeout=30) as reply:
            return b"<Key>" in reply.read()

    return exists, prefix_exists


def github_linux_jobs_active(repository, run_id, token):
    def active():
        page = 1
        while True:
            request = urllib.request.Request(
                f"https://api.github.com/repos/{repository}/actions/runs/"
                f"{run_id}/jobs?filter=latest&per_page=100&page={page}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
            )
            with urllib.request.urlopen(request, timeout=30) as reply:
                jobs = json.load(reply)["jobs"]
            if any(
                job["name"].startswith("Linux::") and job["status"] in ACTIVE_JOB_STATES
                for job in jobs
            ):
                return True
            if len(jobs) < 100:
                return False
            page += 1

    return active


def artifact_base_url(therock_dir, repository, run_id):
    sys.path.insert(0, str(therock_dir / "build_tools"))
    from _therock_utils.workflow_outputs import WorkflowOutputRoot

    root = WorkflowOutputRoot.from_workflow_run(
        run_id=run_id, platform="linux", github_repository=repository
    )
    return f"https://{root.bucket}.s3.amazonaws.com/{root.prefix}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, help="e.g. gfx125X-dcgpu")
    parser.add_argument("--target", required=True, help="e.g. gfx1250")
    parser.add_argument("--therock-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=17400)
    parser.add_argument("--poll", type=float, default=60)
    args = parser.parse_args()

    repository = os.environ["GITHUB_REPOSITORY"]
    run_id = os.environ["GITHUB_RUN_ID"]
    base_url = artifact_base_url(args.therock_dir, repository, run_id)
    print(f"Artifact root: {base_url}", flush=True)
    exists, prefix_exists = s3_checks(base_url)
    linux_jobs_active = github_linux_jobs_active(
        repository, run_id, os.environ["GITHUB_TOKEN"]
    )
    reason = wait(
        args.family,
        args.target,
        exists,
        prefix_exists,
        linux_jobs_active,
        args.timeout,
        args.poll,
    )
    if reason:
        print(f"::error::{reason}")
        return 1
    print(f"All {args.family} rocjitsu inputs are uploaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
