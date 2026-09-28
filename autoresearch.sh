#!/usr/bin/env bash
set -euo pipefail
# Site-local orchestration is kept outside the repository. The GPU workload is offline.
exec "${PYTHON:-python}" -c '
import json, os, pathlib, runpy, sys
site = pathlib.Path(os.environ.get("ROCKE_SDPA_SITE", pathlib.Path.home() / ".rocke" / "gfx1151-sdpa-site.json"))
config = json.loads(site.read_text(encoding="utf-8"))
runner = pathlib.Path(config["source_root"]) / "library/benchmarks/gfx1151/attention/run_remote.py"
sys.argv = [str(runner), "--site", str(site)]
runpy.run_path(str(runner), run_name="__main__")
'
