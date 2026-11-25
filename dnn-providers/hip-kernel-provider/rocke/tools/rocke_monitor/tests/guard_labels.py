# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The develop guard's labels the tests' configurations name (config `incident_issue`): the CI-red label, its
extra label and the stability label. The code names no label of its own (rkmon_lib/breakage.py), so a test that
files, reads or prints develop issues names them. They are invented, unlike any site's, so a test passes only when
the code takes its labels from the settings."""

RED, EXTRA, STABILITY = "red-develop", "needs-revert", "silent-break"
INCIDENT_ISSUE = {"labels": [RED, EXTRA], "stability_label": STABILITY}
