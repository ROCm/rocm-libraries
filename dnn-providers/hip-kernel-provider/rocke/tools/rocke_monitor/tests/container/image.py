#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
The image's test (QUICK_BUILD_GUIDE.md, step 4, `rkmon image test`): it takes an image built already (--no-build), or
builds one from a build settings file (--settings: `rkmon build`, from a built package folder), then runs its stages
on it, each in a throwaway container with only the compose file's capabilities and process cap
(rkmon_lib/container_layout.py RUN_LIMITS); the deploy-path, no-restart and web-host stages deploy it instead
(tests/container/stage_deploy_path.py, stage_no_restart.py, stage_web_host.py). `rkmon image test --help` says the
network each stage runs on. The code of each stage, and what it calls, says in full what it checks.

The one-time-move, adopt, team-prototype and several-deployments stages deploy it too, each on deployments of its own
as the deploy-path stage makes one (tests/container/stage_one_time_move.py, stage_adopt.py, stage_team_prototype.py,
stage_several_deployments.py).

Usage: tests/container/image.py [--package DIR] [--engine docker|podman] (--settings FILE [--tag TAG] [--keep-image]
       | --no-build --image IMAGE) [--stages STAGE,...] [--names PREFIX] [--without-cap CAPABILITY]
       [--earlier-image IMAGE]
DIR is the built package folder (tools/make_package.py --out writes it; the default is the folder this script is in).
--settings FILE builds <image.name>:TAG (default tag image-test) with DIR's docker/rkmon build, from a private copy
of that build settings file whose image.tag is TAG, tests it, then removes it unless --keep-image; --no-build tests IMAGE, built already (rkmon build),
as it is, and keeps it. --stages runs only the stages it names (default: all, in their order). --names PREFIX begins
the name of each container and volume the test makes (default rkmon-image-test). --without-cap runs every container
without that one of the compose file's capabilities: a stage that fails then is one that needs it. Each container runs
with SMOKE_RUN_OPTS (default: --cpus 4 --memory 8g). --earlier-image IMAGE is an image of the package before this
one: the first-apply stage (tests/container/stage_first_apply.py) deploys it, then moves that deployment in place to
IMAGE; without it, that stage is skipped.
Network: building the image (the base image, its packages, ROCm, the atlassian CLI, gh's release, the SDK).
Exit 0 when every check passed.

The stages, in order (DIR's rkmon_lib/steps.py IMAGE_TEST_STAGES, which `rkmon image test --help` lists too):
"""

import copy
import itertools
import json
import os
import platform
import re
import shlex
import shutil
import sys
import tempfile
import time

import hostrun
import stage_deploy_path
import stage_first_apply
import stage_no_restart
import stage_web_host
import stage_adopt
import stage_one_time_move
import stage_several_deployments
import stage_team_prototype

STOP_GRACE_S = 60  # docker stop's grace in the start stage: a stop that takes half of it was not passed on
TOOLS = r'''
    claude --version; echo "claude=$?"
    /opt/rkmon/venv-claude/bin/python -c "import claude_agent_sdk as s; print(\"sdk\", s.__version__)"; echo "sdk=$?"
    gh --version | head -1; echo "gh=$?"
    atlassian version >/dev/null 2>&1; echo "atlassian=$?"
    git --version; cmake --version | head -1; ninja --version; /opt/rkmon/venv/bin/python -c "import pytest"; echo "venv=$?"
    id -un; echo "home=$HOME container=$RKMON_CONTAINER"'''
# `rk selftest`, the image's own code on the checks venv, in the environment it gives the suite everywhere: the summary
# lines, then whatever names what the tests left in their home folder, their TMPDIR or their monitor home, or added to
# the package's folder (the output's last lines).
UNITS = r'''
    out=$(mktemp) && /opt/rkmon/venv/bin/python -B /opt/rocke_monitor/rkmon.py selftest >"$out" 2>&1
    echo "units=$?"
    awk '/^FAILED: the tests /{left=1} left || /^(Ran |OK|FAILED|FAIL:|ERROR:)/' "$out"'''
# A stage on the deployment the image's stages share (tests/container/stages.py set_up: deploy, basics, misconfigured,
# reviewer, parallel-reviews, parallel-reviews-local and stop), as root, in a home of its own under /tmp, which the
# monitor's user can read. {stage}: its name.
ON_A_DEPLOYMENT = """
    python3 -B /opt/rocke_monitor/tests/container/stages.py {stage} --home /tmp/{stage}-stage"""
# rk doctor's own check (rkmon_lib/doctor.py hip_check: one compile for each of build_plan.HIP_CHECK_TARGETS), on the
# image's ROCm: the LLVM tools its ROCm record names. {record}: the record's path, the layout's ROCM_RECORD
# (rkmon_lib/container_layout.py), quoted for the shell.
HIP = r'''
    record={record}
    if [ ! -f "$record" ]; then echo "hip=none"; exit 0; fi
    cd /opt/rocke_monitor && python3 -B -c "
import json, sys
from rkmon_lib import build_plan, doctor
c = doctor.hip_check(json.load(open(sys.argv[1]))['llvm_bin'])
print(c.status, c.found, *([c.fix] if c.fix else []))
print('targets', ' '.join(build_plan.HIP_CHECK_TARGETS))
sys.exit(c.status != 'PASS')" "$record"; echo "hip=$?"'''
ROCM = '''
    python3 -B /opt/rocke_monitor/tests/container/stages.py rocm --home "$(mktemp -d)/rocm-stage"'''
ATLASSIAN = '''
    python3 -B /opt/rocke_monitor/tests/container/stages.py atlassian --home "$HOME/atlassian-stage"'''
ISOLATION = """
    python3 -B /opt/rocke_monitor/tests/container/stages.py isolation --home /root/isolation-stage"""
CANARY = """
    python3 -B /opt/rocke_monitor/tests/container/stages.py canary --home /root/canary-stage"""
# What a container of other ids says about its users and its volumes, as the monitor's user.
IDS = r"""
    echo "monitor=$(id -u):$(id -g) write=$(id -u rkwrite):$(id -g rkwrite) pr=$(id -u rkpr1):$(id -g rkpr1)"
    echo "work=$(stat -c %u:%g /work) secrets=$(stat -c %u:%g:%a /secrets) write_volume=$(stat -c %u:%g:%a /secrets-write)"
    echo "code=$(stat -c %u:%g:%a /code)"
    echo "groups=$(id -Gn)"
    if [ -f /work/kept ]; then echo "kept=$(stat -c %u:%g /work/kept)"; fi
    touch /work/kept && echo "made=$(stat -c %u:%g /work/kept)"
"""
ROOT_ENV = [
    "-e",
    "RKMON_AS=root",
]  # the entry point maps the ids, then stays root (the root stages' setup)
# Every label an image has begins with one of these.
LABEL_PREFIXES = ("org.opencontainers.", "org.rocke.")
NAMES = "rkmon-image-test"  # the default beginning of the name of each container and volume the test makes
# A container's or a volume's name, as docker takes it.
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def plans(pkg: str):
    """The package's build plan (rkmon_lib/build_plan.py)."""
    sys.path.insert(0, pkg)
    from rkmon_lib import build_plan  # noqa: E402

    return build_plan


def locations(pkg: str) -> tuple:
    """The variables the build's download locations are read from (rkmon_lib/build_plan.py DOWNLOADS: each a build
    secret)."""
    bp = plans(pkg)
    return tuple(bp.secret_variable(d.id) for d in bp.DOWNLOADS)


def layout(pkg: str):
    """The package's container layout (rkmon_lib/container_layout.py)."""
    sys.path.insert(0, pkg)
    from rkmon_lib import container_layout  # noqa: E402

    return container_layout


def stage_list(pkg: str) -> dict:
    """The test's stages, in the order it runs them, each with what it checks: the package's own list
    (rkmon_lib/steps.py IMAGE_TEST_STAGES), which `rkmon image test --help` lists too.
    """
    sys.path.insert(0, pkg)
    from rkmon_lib import steps  # noqa: E402

    return steps.IMAGE_TEST_STAGES


def stage_help(pkg: str) -> str:
    """The stages as --help lists them after the usage, a line each: its name, then what it checks."""
    known = stage_list(pkg)
    width = max(len(name) for name in known)
    return "".join(f"#   {name:<{width}}  {what}\n" for name, what in known.items())


def limits(pkg: str, without: str = "") -> list:
    """The compose file's limits as `docker run` flags (rkmon_lib/container_layout.py RUN_LIMITS, flags and their
    values): only its capabilities, but WITHOUT when one is named, and its cap on the processes. Every container of
    this test runs with them.
    """
    flags = list(layout(pkg).RUN_LIMITS)
    pairs = zip(flags[::2], flags[1::2])
    return [x for pair in pairs if pair != ("--cap-add", without) for x in pair]


def run_words(engine: str, pkg: str, without: str, name: str) -> list:
    """`docker run`'s first words for one throwaway container named NAME: no network, the compose file's limits
    (limits, WITHOUT left out), and a host name of docker's form (12 hex digits), another for each container, that the
    container's hosts file gives the loopback address. With no network, docker gives the host name no address, and
    sudo says on each call that it cannot resolve the host."""
    host = os.urandom(6).hex()
    return [
        *(engine, "run", "--rm", "--network", "none", *limits(pkg, without)),
        *("--name", name, "--hostname", host),
        *("--add-host", f"{host}:{layout(pkg).LOOPBACK}"),
    ]


def start_stage(engine: str, image: str, run: list, run_opts: list, res) -> None:
    """A container started as compose starts it: once its root phase is done, `docker stop` ends it at once, with the
    entry point's own exit, because tini (root, the first process) passes the signal on to the monitor's user.
    """
    rc, cid = hostrun.capture(
        [*[a for a in run if a != "--rm"], "-d", *run_opts, image]
    )
    if rc != 0:
        res.result("FAIL", f"start: the container did not start: {cid}")
        return
    try:
        logs, waited_s = "", 0
        while "not set up yet" not in logs and waited_s < 120:
            time.sleep(2)
            waited_s += 2
            _rc, logs = hostrun.capture([engine, "logs", cid])
        start = time.monotonic()
        hostrun.capture([engine, "stop", "-t", str(STOP_GRACE_S), cid])
        took_s = time.monotonic() - start
        _rc, code = hostrun.capture(
            [engine, "inspect", "--format", "{{.State.ExitCode}}", cid]
        )
        hostrun.indented(f"{logs}\nexit {code} after {took_s:.0f} s")
        res.result(
            (
                "PASS"
                if "not set up yet" in logs
                and code == "0"
                and took_s < STOP_GRACE_S / 2
                else "FAIL"
            ),
            "start: with only the compose file's capabilities, the start's root phase runs, and docker stop ends the "
            "container at once (tini passes the signal on)",
        )
    finally:
        hostrun.capture([engine, "rm", "-f", "-v", cid])


def deployment_stage(
    stage: str, what: str, run, run_opts: list, image: str, res
) -> None:
    """STAGE, a stage on the deployment the image's stages share (ON_A_DEPLOYMENT), in a throwaway container (RUN(stage)
    gives `docker run`'s first words): its output, then its row, PASS when the stage passed, with WHAT, what it checks
    as the package's list of the stages says it."""
    command = ON_A_DEPLOYMENT.format(stage=stage)
    rc, out = hostrun.capture(
        [*run(stage), *ROOT_ENV, *run_opts, image, "shell", "-c", command]
    )
    hostrun.indented(out)
    res.result("PASS" if rc == 0 else "FAIL", f"{stage}: {what}")


def built_name(pkg: str, settings: str) -> str:
    """The image name a build settings file gives its image (image.name, as rkmon_lib/build_plan.py plans it); "" for
    a file this package does not read as one."""
    bp = plans(pkg)
    from rkmon_lib import file_types, settings_file  # noqa: E402

    try:
        doc = settings_file.read(settings, file_types.BUILD)
    except (settings_file.Refused, OSError):
        return ""
    return bp.plan(doc, "", "", platform.machine()).image.name


def build_image(pkg: str, settings: str, tag: str) -> int:
    """DIR's rkmon build of <image.name>:TAG: from a private copy of the build settings file SETTINGS whose image.tag
    is TAG (rkmon build takes the tag from the file alone), in a private folder removed after the build, so that the
    image the test builds, and removes, is its own. rkmon build's exit code."""
    plans(pkg)  # the package's rkmon_lib, first on the path
    from rkmon_lib import file_types, settings_file  # noqa: E402

    values = copy.deepcopy(settings_file.read(settings, file_types.BUILD).values)
    values.setdefault("image", {})["tag"] = tag
    folder = tempfile.mkdtemp(prefix="rkmon-image-test-")
    try:
        copied = hostrun.write_settings(folder, values)
        return hostrun.quiet(
            [sys.executable, "-B", f"{pkg}/docker/rkmon", "build", "--settings", copied]
        )
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def image_labels(engine: str, image: str) -> dict:
    rc, out = hostrun.capture(
        [engine, "image", "inspect", "--format", "{{json .Config.Labels}}", image]
    )
    try:
        v = json.loads(out) if rc == 0 else {}
    except ValueError:
        return {}
    return v if isinstance(v, dict) else {}


def labels_findings(labels: dict, history: str, names: tuple, wanted: tuple) -> list:
    """What is wrong with an image's labels and history: a label of WANTED (the build's: build_plan.label_names) missing
    (one may be blank: a version the build was not given), a label that begins with none of LABEL_PREFIXES, an address
    in a label, a value of a download location (NAMES, each a build secret's variable, which a step reads from its
    secret) in a layer's history.
    """
    found = [f"no label {k}" for k in wanted if k not in labels]
    found += [
        f"label {k} is neither {' nor '.join(p + '*' for p in LABEL_PREFIXES)}"
        for k in labels
        if not k.startswith(LABEL_PREFIXES)
    ]
    found += [f"label {k} has an address" for k, v in labels.items() if "://" in str(v)]
    for name in names:
        if re.search(rf"\b{name}=(?!\$\()\S", history):
            found.append(f"the history has a value of {name}")
    return found


def record_findings(text, labels: dict, bp) -> list:
    """What is wrong with an image's build record (BP, rkmon_lib/build_plan.py: RECORD_PATH), TEXT as the image gives it
    (None: it has none) and LABELS its labels: no JSON, no build record of the layout BP writes, a fingerprint that is
    not the label's (FINGERPRINT_LABEL), a registry, or an address in it."""
    if text is None:
        return [f"no build record at {bp.RECORD_PATH}"]
    try:
        record = json.loads(text)
    except ValueError:
        return ["the build record is no JSON"]
    record = record if isinstance(record, dict) else {}
    want = (bp.RECORD_KIND, bp.RECORD_SCHEMA_VERSION)
    if (record.get("kind"), record.get("schema_version")) != want:
        return [f"the build record is no {want[0]} of layout {want[1]}"]
    found = []
    if record.get("build_settings_fingerprint") != labels.get(bp.FINGERPRINT_LABEL):
        found.append(
            f"the build record's fingerprint is not the label {bp.FINGERPRINT_LABEL}'s"
        )
    if (record.get("image") or {}).get("registry"):
        found.append("the build record names a registry")
    if "://" in text:
        found.append("the build record has an address")
    return found


def hip_result(out: str, bp) -> tuple:
    """The hip stage's result and its line, from OUT, what its container printed (HIP): PASS when rk doctor's hipcc
    check passed, for the GPU targets of BP (rkmon_lib/build_plan.py HIP_CHECK_TARGETS), which the line names; the
    image's own build plan must give the same targets."""
    targets = bp.HIP_CHECK_TARGETS
    what = f"hip: the image's hipcc compiles <hip/hip_runtime.h> for {', '.join(targets)} (rk doctor's check)"
    lines = out.split("\n")
    if "hip=none" in lines:
        return "FAIL", what + ": the image has no ROCm"
    passed = "hip=0" in lines and f"targets {' '.join(targets)}" in lines
    return ("PASS" if passed else "FAIL"), what


def main(args: list) -> int:
    hostrun.setup_output()
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = hostrun.folder(os.path.join(here, "..", ".."))
    engine = os.environ.get("CONTAINER_ENGINE", "")
    settings, tag, keep, build_it = "", "image-test", 0, True
    image, stages, names, without, earlier = "", None, NAMES, "", ""
    run_opts = hostrun.run_opts("--cpus 4 --memory 8g")
    while args:
        a = args[0]
        if a == "--package":
            pkg = hostrun.folder(hostrun.need_value(args, a))
        elif a == "--engine":
            engine = hostrun.need_value(args, a)
        elif a == "--settings":
            settings = hostrun.need_value(args, a)
        elif a == "--tag":
            tag = hostrun.need_value(args, a)
        elif a == "--image":
            image = hostrun.need_value(args, a)
        elif a == "--earlier-image":
            earlier = hostrun.need_value(args, a)
        elif a == "--stages":
            stages = [s for s in hostrun.need_value(args, a).split(",") if s]
        elif a == "--names":
            names = hostrun.need_value(args, a)
            if not NAME.fullmatch(names):
                print(
                    f"--names {names!r}: letters, digits, _ . and -, a letter or a digit first"
                )
                return 2
        elif a == "--without-cap":
            without = hostrun.need_value(args, a).upper()
            kept = layout(pkg).CAPABILITIES
            if without not in kept:
                print(
                    f"--without-cap {without}: the compose file keeps {', '.join(kept)}"
                )
                return 2
        elif a == "--keep-image":
            keep = 1
            args = args[1:]
            continue
        elif a == "--no-build":  # test the image built already, as it is, and keep it
            build_it, keep = False, 1
            args = args[1:]
            continue
        elif a in ("-h", "--help"):
            usage = hostrun.help_lines(__doc__, len(__doc__.splitlines()))
            sys.stdout.write(usage + stage_help(pkg))
            return 0
        else:
            print(f"unknown argument: {a} (see --help)")
            return 2
        args = args[2:]
    known = stage_list(pkg)
    bad = [s for s in stages or () if s not in known]
    if bad:
        print(f"unknown stage: {bad[0]} (the stages: {','.join(known)})")
        return 2
    stages = list(known) if stages is None else stages
    engine = hostrun.pick_engine(engine)
    if not engine:
        return 2
    res = hostrun.Results()

    if build_it and not settings or not build_it and not image:
        print(
            "name --settings FILE (a build settings file) or --no-build --image IMAGE (see --help)"
        )
        return 2
    if build_it:
        name = built_name(pkg, settings)
        if not name:
            print(
                f"FAIL  {settings}: no build settings file this package reads (rkmon check --settings FILE says why)"
            )
            return 2
        image = f"{name}:{tag}"
    print(
        f"== building {image} from {pkg} ({settings})"
        if build_it
        else f"== testing {image} as it was built"
    )
    os.environ["CONTAINER_ENGINE"] = engine  # rkmon build's engine
    if build_it and build_image(pkg, settings, tag) != 0:
        print("FAIL  building the image (rkmon build)")
        return 1
    if not build_it and hostrun.quiet([engine, "image", "inspect", image]) != 0:
        print(f"FAIL  no image {image} to test (--no-build)")
        return 1
    labels = image_labels(engine, image)
    count = itertools.count(1)

    def run(stage: str) -> list:
        """`docker run`'s first words for one throwaway container of STAGE, named after --names (run_words)."""
        name = f"{names}-{stage}-{os.getpid()}-{next(count)}"
        return run_words(engine, pkg, without, name)

    if "labels" in stages:
        bp = plans(pkg)
        _rc, history = hostrun.capture(
            [engine, "history", "--no-trunc", "--format", "{{.CreatedBy}}", image]
        )
        found = labels_findings(labels, history, locations(pkg), bp.label_names())
        rc, record = hostrun.capture(
            [*run("labels"), "--entrypoint", "cat", *run_opts, image, bp.RECORD_PATH],
            both=False,
        )
        found += record_findings(record if rc == 0 else None, labels, bp)
        res.result(
            "FAIL" if found else "PASS",
            "labels: the build's labels and its record are there, the record's fingerprint the label's; each label "
            "org.opencontainers.* or org.rocke.*; no address in a label or the record, and no download location in "
            "the history",
        )
        hostrun.indented(
            "\n".join(found + [f"labels: {json.dumps(labels, sort_keys=True)}"])
        )

    if "tools" in stages:
        _rc, out = hostrun.capture(
            [*run("tools"), *run_opts, image, "shell", "-c", TOOLS]
        )
        s = "PASS"
        wants = ["gh=0", "venv=0", "rkmon", "container=1", "claude=0", "sdk=0"]
        wants += ["atlassian=0"]
        for want in wants:
            if want not in out:
                print(f"   missing: {want}")
                s = "FAIL"
        res.result(
            s,
            "tools: gh, git, cmake, ninja, the checks venv, the non-root user, Claude Code and its SDK, and the "
            "atlassian CLI",
        )
        hostrun.indented(out)

    if "units" in stages:
        _rc, out = hostrun.capture(
            [*run("units"), *run_opts, image, "shell", "-c", UNITS]
        )
        # The checks venv runs them, as it runs rocKE's tests (pytest is in it, not in the system Python).
        res.result(
            "PASS" if "units=0" in out else "FAIL",
            "units: the package's unit tests inside the image (rk selftest, on the checks venv)",
        )
        hostrun.indented(out)

    if "deploy" in stages:
        deployment_stage("deploy", known["deploy"], run, run_opts, image, res)

    if "hip" in stages:
        hip = HIP.format(record=shlex.quote(layout(pkg).ROCM_RECORD))
        _rc, out = hostrun.capture([*run("hip"), *run_opts, image, "shell", "-c", hip])
        hostrun.indented(out)
        res.result(*hip_result(out, plans(pkg)))

    if "rocm" in stages:
        rc, out = hostrun.capture(
            [*run("rocm"), *ROOT_ENV, *run_opts, image, "shell", "-c", ROCM]
        )
        hostrun.indented(out)
        res.result(
            "PASS" if rc == 0 else "FAIL",
            "rocm: install_rocm.sh against a fake installers' folder and a fake download",
        )

    if "atlassian" in stages:
        rc, out = hostrun.capture(
            [*run("atlassian"), *run_opts, image, "shell", "-c", ATLASSIAN]
        )
        hostrun.indented(out)
        res.result(
            "PASS" if rc == 0 else "FAIL",
            "atlassian: install_atlassian.sh against a fake download",
        )

    if "isolation" in stages:
        rc, out = hostrun.capture(
            [*run("isolation"), *ROOT_ENV, *run_opts, image, "shell", "-c", ISOLATION]
        )
        hostrun.indented(out)
        res.result(
            "PASS" if rc == 0 else "FAIL",
            "isolation: the write login, the PR runners and their slots, with fake logins (stage_isolation.py)",
        )

    if "canary" in stages:
        rc, out = hostrun.capture(
            [*run("canary"), *ROOT_ENV, *run_opts, image, "shell", "-c", CANARY]
        )
        hostrun.indented(out)
        res.result(
            "PASS" if rc == 0 else "FAIL",
            "canary: canaries where the private values live: no PR runner finds one, and neither the monitor nor the "
            "session reads the writer's volume (stage_canary.py)",
        )

    if "ids" in stages:
        ids_stage(engine, image, run, run_opts, res, names)

    if "start" in stages:
        start_stage(engine, image, run("start"), run_opts, res)

    if "basics" in stages:
        deployment_stage("basics", known["basics"], run, run_opts, image, res)

    if "misconfigured" in stages:
        deployment_stage(
            "misconfigured", known["misconfigured"], run, run_opts, image, res
        )

    if "reviewer" in stages:
        deployment_stage("reviewer", known["reviewer"], run, run_opts, image, res)

    if "parallel-reviews" in stages:
        deployment_stage(
            "parallel-reviews", known["parallel-reviews"], run, run_opts, image, res
        )

    if "parallel-reviews-local" in stages:
        deployment_stage(
            "parallel-reviews-local",
            known["parallel-reviews-local"],
            run,
            run_opts,
            image,
            res,
        )

    if "stop" in stages:
        deployment_stage("stop", known["stop"], run, run_opts, image, res)

    if "deploy-path" in stages:
        stage_deploy_path.deploy_path_stage(engine, image, pkg, names, run_opts, res)

    if "no-restart" in stages:
        stage_no_restart.no_restart_stage(engine, image, pkg, names, run_opts, res)

    if "web-host" in stages:
        stage_web_host.web_host_stage(engine, image, pkg, names, run_opts, res)

    if "one-time-move" in stages:
        stage_one_time_move.one_time_move_stage(
            engine, image, pkg, names, run_opts, res
        )

    if "adopt" in stages:
        stage_adopt.adopt_stage(engine, image, pkg, names, run_opts, res)

    if "team-prototype" in stages:
        stage_team_prototype.team_prototype_stage(
            engine, image, pkg, names, run_opts, res
        )

    if "several-deployments" in stages:
        stage_several_deployments.several_deployments_stage(
            engine, image, pkg, names, run_opts, res
        )

    if "first-apply" in stages:
        stage_first_apply.first_apply_stage(
            engine, image, pkg, names, run_opts, res, earlier
        )

    if keep != 1:
        hostrun.capture([engine, "rmi", image])
    print("IMAGE TEST: PASS" if res.failed == 0 else "IMAGE TEST: FAIL")
    return res.failed


def ids_stage(
    engine: str, image: str, run, run_opts: list, res, names: str = NAMES
) -> None:
    """The deployment's ids at the start, the volumes that keep up when they change, and again in a new container with
    the same ids (docker gives an empty volume's folder the image's owner in each one), the code volume root's in each,
    a refused set, and the isolation proof under other ids. RUN(stage) gives `docker run`'s first words; NAMES begins
    each volume's name.
    """
    vols = {
        f"{names}-ids-{os.getpid()}-{n}": where
        for n, where in (
            ("work", "/work"),
            ("secrets", "/secrets"),
            ("write", "/secrets-write"),
            ("code", "/code"),
        )
    }
    mounts = [x for v, where in vols.items() for x in ("-v", f"{v}:{where}")]

    def ids(uid, gid) -> list:
        return ["-e", f"RKMON_UID={uid}", "-e", f"RKMON_GID={gid}"]

    a, b = ids(4242, 4343), ids(5252, 5353)
    # Variables that would name the writer's id or a runner's change nothing: those are the package's.
    package = ["-e", "RKMON_WRITE_UID=4244", "-e", "RKMON_PR_UID=4245"]
    code = "code=0:0:755"  # the code volume, root's whatever the ids (rkmon_lib/code_volume.py)
    try:
        _rc, first = hostrun.capture(
            [*run("ids"), *run_opts, *mounts, *a, *package, image, "shell", "-c", IDS]
        )
        hostrun.indented(first)
        want = (
            "monitor=4242:4343",
            "write=1999:4343",
            "pr=1998:1982",
            "work=4242:4343",
            "secrets=4242:4343:700",
            "write_volume=1999:4343:700",
            code,
            "made=4242:4343",
        )
        ok = all(x in first for x in want) and re.search(r"groups=.*\brkpr\b", first)
        res.result(
            "PASS" if ok else "FAIL",
            "ids: the deployment's ids (4242:4343) are the monitor's at the start; the writer and the PR runners keep "
            "the package's (the writer's 1999, the first runner's 1998:1982), whatever the environment says; the "
            "volumes are theirs, their permissions kept, and the code volume root's, mode 755",
        )
        _rc, second = hostrun.capture(
            [*run("ids"), *run_opts, *mounts, *b, image, "shell", "-c", IDS]
        )
        hostrun.indented(second)
        res.result(
            (
                "PASS"
                if all(
                    x in second
                    for x in (
                        "monitor=5252:5353",
                        "kept=5252:5353",
                        "work=5252:5353",
                        "secrets=5252:5353:700",
                        "write_volume=1999:5353:700",
                        code,
                    )
                )
                else "FAIL"
            ),
            "ids: the same volumes under other ids (5252:5353): their files follow them, and the code volume stays "
            "root's",
        )
        _rc, third = hostrun.capture(
            [*run("ids"), *run_opts, *mounts, *b, image, "shell", "-c", IDS]
        )
        hostrun.indented(third)
        res.result(
            (
                "PASS"
                if all(
                    x in third
                    for x in (
                        "secrets=5252:5353:700",
                        "write_volume=1999:5353:700",
                        "kept=5252:5353",
                        code,
                    )
                )
                else "FAIL"
            ),
            "ids: a new container with the same ids: the volumes' folders are still theirs, and the code volume root's",
        )
        rc, bad = hostrun.capture(
            [
                *run("ids"),
                *run_opts,
                "-e",
                "RKMON_UID=1999",
                image,
                "shell",
                "-c",
                "echo started",
            ]
        )
        hostrun.indented(bad)
        res.result(
            (
                "PASS"
                if rc == 2
                and "is the GitHub writer's id" in bad
                and "host.users.monitor_uid" in bad
                and "nothing changed" in bad
                and "started" not in bad
                else "FAIL"
            ),
            "ids: the monitor on the write user's id: refused before anything runs, naming the setting to change",
        )
        rc, out = hostrun.capture(
            [*run("ids"), *ROOT_ENV, *run_opts, *a, image, "shell", "-c", ISOLATION]
        )
        hostrun.indented("\n".join(out.splitlines()[-4:]))
        res.result(
            "PASS" if rc == 0 else "FAIL",
            "ids: the isolation proof (the write login, the PR runners and their slots) under the deployment's ids",
        )
    finally:
        hostrun.capture([engine, "volume", "rm", "-f", *vols])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
