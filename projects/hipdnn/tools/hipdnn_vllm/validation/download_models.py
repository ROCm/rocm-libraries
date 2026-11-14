# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Download the models.json weights to <root>/<path>.

usage: python download_models.py --root DIR [--models models.json] [name ...]

Network only, so it may run on a login node when compute nodes have no
internet (jobs then use HF_HUB_OFFLINE=1). The Hugging Face API is called once
per repo to list files, retrying on HTTP 429 (a shared login IP exhausts the
API rate limit quickly); files then come from the resolve endpoint, which has a
separate limit. Files already present with the expected size are skipped.
Gated repos are not supported (no token is used): pick ungated copies in
models.json.

Each model is fetched at its models.json ``revision`` (a branch, tag or commit
sha; default ``main``). The listing and every file use the commit sha that
revision resolves to, so one download never mixes two commits, and the sha is
recorded in ``<dest>/hf_revision.json``.
"""

from __future__ import annotations

import argparse
import datetime
import email.utils
import fnmatch
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
INCLUDE = ["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.model", "*.jinja", "*.py", "sentencepiece*"]
EXCLUDE = ["original/*", "metal/*", "onnx/*", "*/onnx/*"]


def _retry_after(value: str | None, default: int = 60) -> int:
    """Seconds to wait for a Retry-After header: delay seconds or an HTTP-date."""
    if not value:
        return default
    value = value.strip()
    if value.isdigit():
        return int(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return default
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.timezone.utc)
    return max(0, int((when - datetime.datetime.now(datetime.timezone.utc)).total_seconds()))


def _get(url: str, attempts: int = 30):
    for i in range(attempts):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hipdnn-vllm"}),
                                          timeout=120)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or i == attempts - 1:
                raise
            wait = _retry_after(e.headers.get("Retry-After")) + 5
            print(f"   HTTP {e.code} on {url}; retry in {wait} s", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as e:
            if i == attempts - 1:
                raise
            print(f"   {e} on {url}; retry in 30 s", flush=True)
            time.sleep(30)
    raise RuntimeError("unreachable")


def resolve_revision(entry: dict) -> str:
    """The commit sha the entry's revision (default main) points at."""
    rev = urllib.parse.quote(entry.get("revision", "main"), safe="")
    with _get(f"https://huggingface.co/api/models/{entry['hf_id']}/revision/{rev}") as r:
        return json.load(r)["sha"]


def files_for(entry: dict, sha: str) -> list[tuple[str, int]]:
    with _get(f"https://huggingface.co/api/models/{entry['hf_id']}/tree/{sha}?recursive=1") as r:
        tree = json.load(r)
    inc = INCLUDE + entry.get("hf_include", [])
    keep = []
    for x in tree:
        path = x["path"]
        if x["type"] != "file" or any(fnmatch.fnmatch(path, p) for p in EXCLUDE):
            continue
        if any(fnmatch.fnmatch(path, p) for p in inc):
            keep.append((path, x.get("lfs", {}).get("size", x.get("size", 0))))
    return keep


def download(entry: dict, dest: str) -> None:
    rev = entry.get("revision", "main")
    sha = resolve_revision(entry)
    files = files_for(entry, sha)
    print(f"== {entry['hf_id']}@{rev} ({sha}) -> {dest}: {len(files)} files, "
          f"{sum(s for _, s in files) / 1e9:.2f} GB", flush=True)
    # A different commit may change files without changing their size, so a
    # recorded sha change makes every file stale. A tree without a record (a
    # download made before records existed) keeps size-matching files; they are
    # listed as unverified in the record.
    record = os.path.join(dest, "hf_revision.json")
    prior = json.load(open(record)).get("sha") if os.path.exists(record) else None
    unverified = []
    for path, size in files:
        out = os.path.join(dest, path)
        if prior in (sha, None) and os.path.exists(out) and os.path.getsize(out) == size:
            if prior is None:
                unverified.append(path)
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        part = out + ".part"
        with _get(f"https://huggingface.co/{entry['hf_id']}/resolve/{sha}/{path}") as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 24)
        if os.path.getsize(part) != size:
            raise RuntimeError(f"{path}: got {os.path.getsize(part)} bytes, expected {size}")
        os.replace(part, out)
        print(f"   {path} ({size / 1e9:.2f} GB)", flush=True)
    os.makedirs(dest, exist_ok=True)
    json.dump({"hf_id": entry["hf_id"], "revision": rev, "sha": sha, "files": [p for p, _ in files],
               "unverified_files": unverified}, open(record, "w"), indent=2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--models", default=os.path.join(HERE, "models.json"))
    p.add_argument("names", nargs="*")
    args = p.parse_args()
    rc = 0
    for entry in json.load(open(args.models))["models"]:
        if args.names and entry["name"] not in args.names:
            continue
        dest = entry["path"] if os.path.isabs(entry["path"]) else os.path.join(args.root, entry["path"])
        try:
            download(entry, dest)
            print(f"DL_OK {entry['name']}", flush=True)
        except Exception as e:  # noqa: BLE001 - report and continue with the next model
            print(f"DL_FAIL {entry['name']}: {e}", flush=True)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
