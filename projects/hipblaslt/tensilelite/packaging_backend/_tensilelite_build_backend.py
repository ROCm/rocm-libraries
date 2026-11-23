# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Setuptools build hooks with the editable source-revision policy.

Editable client validation resolves identity from the source tree. A revision
supplied during installation would label metadata without supplying a usable
runtime identity, so reject it before setuptools prepares editable metadata.

Keep backend-path separate from the project root: frontends can search it for
distribution metadata, and the root contains the package's generated egg-info.
"""

import os

from setuptools import build_meta as _backend

build_wheel = _backend.build_wheel
build_sdist = _backend.build_sdist
get_requires_for_build_wheel = _backend.get_requires_for_build_wheel
get_requires_for_build_sdist = _backend.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = _backend.prepare_metadata_for_build_wheel


def _reject_editable_revision():
    if os.environ.get("TENSILELITE_SOURCE_REVISION"):
        raise RuntimeError(
            "TENSILELITE_SOURCE_REVISION is not supported for editable installs. "
            "Unset it and retry. Editable installs resolve source identity from "
            "their source tree when validating the native client."
        )


def get_requires_for_build_editable(config_settings=None):
    _reject_editable_revision()
    return _backend.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _reject_editable_revision()
    return _backend.prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _reject_editable_revision()
    return _backend.build_editable(wheel_directory, config_settings, metadata_directory)
