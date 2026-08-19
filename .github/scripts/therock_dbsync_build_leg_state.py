# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Helper for the `dbsync-rocjitsu` job in therock-ci.yml. Prints the state of THIS run's MIOpen
# *build* leg for a given GPU family -- i.e. the therock-ci-linux matrix job that actually produces
# the artifact (miopen_gtest + CK plugin + DBs) the dbsync job fetches.
#
# The dbsync job runs on a cheap CPU runner and only `needs: setup`, so it starts before the build
# has uploaded its artifact. Rather than guessing a fixed wait budget, the fetch loop polls this to
# decide whether to keep waiting (build still queued/running), fail fast (build concluded without
# success -> no artifact will ever appear), or expect the artifact imminently (build succeeded).
#
# Uses only the stdlib (urllib) so it needs no extra deps in the no_rocm container. Always exits 0
# and prints a single lowercase token so the caller can `case` on it; API/parse errors print
# "query-error" (treated as "keep waiting", not fatal).
#
# Env: GH_REPO (owner/repo), ARTIFACT_RUN_ID (workflow run id), AMDGPU_FAMILIES (e.g. gfx94X-dcgpu),
#      GITHUB_TOKEN.

import json
import os
import sys
import urllib.error
import urllib.request


def build_leg_state(repo, run_id, family, token):
    # The build leg's job name looks like:
    #   "Linux (hipdnn,...,miopen,... | gfx94X-dcgpu) / Build (gfx94X-dcgpu)"
    # Match the family bundle that contains miopen and is the Build (not Test) sub-job. Substring
    # matches (not regex) so the comma-separated projects_to_test ordering is irrelevant.
    fam_tag = f"| {family})"
    state = "notfound"
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs"
            f"?per_page=100&page={page}"
        )
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(req) as resp:
            data = json.load(resp)
        jobs = data.get("jobs", [])
        if not jobs:
            break
        for job in jobs:
            name = job.get("name", "")
            if fam_tag in name and "miopen" in name and "/ Build (" in name:
                # conclusion is null until the job finishes; fall back to status while it runs.
                state = job.get("conclusion") or job.get("status") or "unknown"
        if len(jobs) < 100:
            break
        page += 1
    return state


def main():
    try:
        state = build_leg_state(
            os.environ["GH_REPO"],
            os.environ["ARTIFACT_RUN_ID"],
            os.environ["AMDGPU_FAMILIES"],
            os.environ["GITHUB_TOKEN"],
        )
    except (urllib.error.URLError, KeyError, ValueError) as exc:
        print("query-error", file=sys.stderr)
        print(f"build-leg query failed: {exc}", file=sys.stderr)
        print("query-error")
        return
    print(state)


if __name__ == "__main__":
    main()
