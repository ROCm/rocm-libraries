# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The labels that name the schema of the pipeline's run records: the audit
manifest, stage05's cells.json and models.pt (and stage04b's decisions.json),
the feature catalog stage05 records, and splits.json."""

import re

import pytest

import stage05_train as s5
from lib import audit as auditlib
from lib import features as fs
from lib import subcells as sc

LABELS = {
    "audit": auditlib.SCHEMA_VERSION,
    "cells": s5.CELLS_SCHEMA_VERSION,
    "features": fs.FEATURES_VERSION,
    "splits": sc.SCHEMA_VERSION,
}


def test_each_record_has_a_label_of_its_own():
    assert s5.CELLS_SCHEMA_VERSION == "pipeline-v3-cells-1"
    assert fs.FEATURES_VERSION == "pipeline-v3-features-1"
    assert sc.SCHEMA_VERSION == "pipeline-v3-splits-1"
    assert len(set(LABELS.values())) == len(LABELS)


@pytest.mark.parametrize("record", sorted(LABELS))
def test_a_label_is_a_name_and_a_version_number(record):
    label = LABELS[record]
    assert re.fullmatch(r"[a-z]+(-[a-z0-9]+)*", label), label
    # no date: a label changes when its record does, not with the calendar
    assert not re.search(r"\d{4}|\d-\d", label), label
