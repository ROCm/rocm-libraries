#!/usr/bin/env python3
"""Triage the WRONG rows of a pairwise run: does the failure follow the input domain?

  triage_wrong.py --root <work dirs> --run <dir> --src <worktree> --build <build> --engine <id> \
      --config <toml> [--plugin libhip_kernel_provider.so] [--domain default] [--tag def] \
      [--class-tol class_check.json] [--no-domain]

For every failures/*/row.json of the run it replays the row twice, solo, with `--row`: unchanged (the
control: a WRONG that does not reproduce is batch dependence, policy.md "Proof per kind") and with
`domain` set to --domain (default `default`: the small, well-conditioned inputs). Prints one line per
row and a summary of (control, with-domain) outcomes, and writes <root>/dab_<tag>/summary.json.

Reading it:  control WRONG -> domain PASS  : the failure needs the wide/large inputs: a conditioning or
tolerance-class effect until something shows otherwise (compare against a shipped engine on the same
graphs: lanes/group.md).  control WRONG -> domain WRONG : not the domain; look at scale, lengths, shapes,
shrunk repro, and the kind (BULK / LARGE / NONFINITE in gauntlet/analyze_runs.py) before calling it a class.
With `--class-tol class_check.json` (from class_check.py) every WRONG row is also replayed with the tolerance of its
ENGINE CLASS (a per-case [[tolerance_overrides]] row appended to a copy of --config, matched on the test name
`*_cases.<case>`): control WRONG -> class-tolerance PASS means the row is not wrong under the engine's own contract;
WRONG -> WRONG means it exceeds even that. Rows class_check marked NO CLASS were replayed with the class it assigned
anyway: say so, since the engine's tolerance model does not cover those graphs. `--no-domain` skips the domain replay.
Set HIP_VISIBLE_DEVICES in the environment to pick the GPU. Pairwise rows only (random mode has no row.json).
"""
import argparse, collections, glob, json, os, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--run", action="append", required=True, help="work dir name(s) under --root")
ap.add_argument("--src", required=True, help="worktree (has dnn-providers/integration-tests/gauntlet)")
ap.add_argument("--build", required=True)
ap.add_argument("--engine", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--plugin", default="libhip_kernel_provider.so")
ap.add_argument("--domain", default="default")
ap.add_argument("--tag", default="def")
ap.add_argument("--class-tol", help="class_check.json: also replay each row with its engine class tolerance")
ap.add_argument("--no-domain", action="store_true", help="skip the domain A/B replay")
a = ap.parse_args()
classes = json.load(open(a.class_tol)) if a.class_tol else {}

cwd = os.path.join(a.src, "dnn-providers/integration-tests")


def name_key(run, name):
    return f"{run}/{name}"

res = {}
for run in a.run:
    for fd in sorted(glob.glob(f"{a.root}/{run}/failures/*")):
        rj = os.path.join(fd, "row.json")
        if not os.path.exists(rj):
            continue
        row = json.load(open(rj))
        name = os.path.basename(fd)
        variants = ["orig"] + ([] if a.no_domain else [a.domain]) + (["class"] if name_key(run, name) in classes else [])
        for variant in variants:
            r = json.loads(json.dumps(row))
            config = a.config
            if variant not in ("orig", "class"):
                r["axes"]["domain"] = variant
            work = f"{a.root}/dab_{a.tag}/{name}_{variant}"
            os.makedirs(work, exist_ok=True)
            if variant == "class":
                c = classes[name_key(run, name)]
                config = os.path.join(work, "class_tol.toml")
                with open(config, "w") as f:
                    f.write(open(a.config, encoding="utf8").read())
                    f.write(f'\n[[tolerance_overrides]]\nfilters = ["*_cases.{row["case_id"]}"]\n'
                            f'atol = {c["atol"]}\nrtol = {c["rtol"]}\n')
            rp = os.path.join(work, "row.json")
            json.dump(r, open(rp, "w"))
            subprocess.run(
                ["python3", "gauntlet/run_gauntlet_fuzz.py", "--row", rp,
                 "--binary", f"{a.build}/bin/hipdnn_integration_tests",
                 "--test-article", f"{a.build}/lib/hipdnn_plugins/engines/{a.plugin}",
                 "--test-engine", a.engine, "--test-config", config, "--work-dir", work + "/w"],
                cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                oc = json.loads(open(work + "/w/outcomes.jsonl").readline())["outcome"]
            except Exception:
                oc = "ERR"
            res.setdefault(name, {"axes": row["axes"]})[variant] = oc
        print(name, {k_: v_ for k_, v_ in res[name].items() if k_ != "axes"}, flush=True)
if not a.no_domain:
    print("SUMMARY (control, with %s domain):" % a.domain,
          dict(collections.Counter((v["orig"], v[a.domain]) for v in res.values())))
if classes:
    print("SUMMARY (control, with the engine class tolerance):",
          dict(collections.Counter((v["orig"], v.get("class")) for v in res.values())))
json.dump(res, open(f"{a.root}/dab_{a.tag}/summary.json", "w"), indent=1)
