# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The configuration keys no settings file holds any more, each with why: the ones no feature reads and the ones
whose value the package sets. The one home of the lists that the one-time move
(rkmon_lib/one_time_move.py) and the translation of production's settings (tools/translate_production.py) drop and
name. A name in any table is a key, or a section whose keys all go.

- REMOVED_KEYS: name -> why, in words ({program}: the program's name, program.named).
- WHY_DROPPED: name -> the key of layout 1's files that held its value, as those files write it, whose rule says why
  (from_layout1.why_dropped).
- PACKAGE_SET: name -> why, in words: the keys whose value the package sets and no deployment changes, so that no
  settings file holds them.
- NO_HOST_SERVICES: why the keys of the services of an install without a container go, in words: that install
  stays, and runs no services; its older services switch's why (renames.OLD_FEATURES) and the translation's line for
  it are made from it.
- `listed(name) -> str`: the entry of any table that is NAME or holds it; "" for none.
- `why(name) -> str`: why NAME, an entry of any table or a key under one, is dropped, in words; "" for any other name.
"""

from __future__ import annotations

from .container_layout import LOOPBACK

# Why the keys of the services of an install without a container go (the WSL distribution they started, and their
# switches): one why for them all, so that a configuration that has them loses them in one line (one_time_move.split
# names the dropped keys by why).
NO_HOST_SERVICES = "an install without a container runs no services"
REMOVED_KEYS = {
    "paths.rocm_reference_clang": "the gate compiler's commit is a package build default, which rk doctor compares "
    "this machine's clang with",
    "worker.api_key_env": "a key comes from a secret, never from an environment variable",
    "worker.header_env": "a key comes from a secret, never from an environment variable",
    "worker.headers_file": "a binding sends no extra headers",
    "worker.source": "a record of the setup interview's choice",
    "worker.omp_provider": "a record of the setup interview's choice",
    **{
        f"worker.{k}": "the local worker's tuning, and the local worker leaves the package"
        for k in (
            "context",
            "temperature",
            "max_output_tokens",
            "num_batch",
            "keep_alive",
            "max_steps",
            "hang_s",
            "max_hangs",
            "max_resends",
        )
    },
    "mirror": "an unused feature",
    "offload": "an unused feature",
    "models.verify": "the verification pass runs in the Claude session, on its own model",
    "models.quality": "the code quality pass runs in the Claude session, on its own model",
    "models.post": "posting runs in the Claude session",
    "models.tick": "the tick triage's model: a role no deployment uses",
    "models.analytics": "the analytics model: a role no deployment uses",
    "models.planner": "the planner's model: a role no deployment uses",
    "models.analyst": "the analyst's model: a role no deployment uses",
    "roles": "bindings for roles no deployment uses",
    "effort.audit": "the offload's effort",
    "effort.deep": "the offload's effort",
    "effort.tick": "the tick triage's effort",
    "admin_dashboard.token_file": "the admin access token is the one secret admin_access_token",
    "admin_dashboard.url": f"the admin dashboard listens on {LOOPBACK} only",
    "atlassian.proxy_url": "no feature reads it",
    "confluence.space": "the monitor never writes to Confluence",
    "wsl_distro": NO_HOST_SERVICES,
    "services": NO_HOST_SERVICES,
    "update.test_timeout_min": "an update runs no tests: the package build runs them",
}
WHY_DROPPED = {
    "update.path": "advanced.update.path",
    "container.mounts": "host.mounts",
    "confluence.site": "integrations.atlassian_cli.confluence_site",
}
# The container's users' ids are the image's (container_layout.WRITER, PR_RUNNER).
_USERS_IDS = "no file outside the container's volumes is this user's, so no deployment changes it"
PACKAGE_SET = {
    "admin_dashboard.publish_host": f"the admin dashboard listens on this host's {LOOPBACK} only, and another "
    "machine reaches it through a tunnel",
    "container.write_uid": _USERS_IDS,
    "container.pr_uid": _USERS_IDS,
}


def listed(name: str) -> str:
    """The entry of REMOVED_KEYS, WHY_DROPPED or PACKAGE_SET that is NAME or holds it; "" for none."""
    for entry in (*REMOVED_KEYS, *WHY_DROPPED, *PACKAGE_SET):
        if name == entry or name.startswith(entry + "."):
            return entry
    return ""


def why(name: str) -> str:
    """Why NAME is dropped: see the module's docstring."""
    entry = listed(name)
    if not entry:
        return ""
    if entry in WHY_DROPPED:
        # a call-time import: from_layout1 is a large module, which the translation's rules read
        from . import from_layout1

        return from_layout1.why_dropped(WHY_DROPPED[entry])
    from . import program

    return program.named({**REMOVED_KEYS, **PACKAGE_SET}[entry])
