# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""PEP 517/660 backend extensions for TensileLite source installations."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

from setuptools import build_meta as _setuptools


_CLIENT_PATH_SETTING = "tensilelite.client-path"
_CLIENT_PATH_RECORD = "tensilelite-client-path.json"


def _backend_settings(config_settings):
    settings = dict(config_settings or {})
    value = settings.pop(_CLIENT_PATH_SETTING, None)
    return settings or None, value


def _validated_client_path(value) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{_CLIENT_PATH_SETTING} must be supplied exactly once")

    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{_CLIENT_PATH_SETTING} must be an absolute executable path: {value!r}")
    path = Path(os.path.abspath(path))
    if not path.is_file():
        raise ValueError(f"{_CLIENT_PATH_SETTING} is not a regular file: {path}")
    if os.name != "nt" and not os.access(path, os.X_OK):
        raise ValueError(f"{_CLIENT_PATH_SETTING} is not executable: {path}")
    return path


def _record_row(name: str, data: bytes) -> tuple[str, str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    return name, f"sha256={digest}", str(len(data))


def _add_client_binding(wheel: Path, client: Path) -> None:
    with zipfile.ZipFile(wheel, "r") as source:
        infos = source.infolist()
        record_names = [info.filename for info in infos if info.filename.endswith(".dist-info/RECORD")]
        if len(record_names) != 1:
            raise ValueError(f"expected one wheel RECORD, found {record_names}")
        record_name = record_names[0]
        dist_info = record_name.rsplit("/", 1)[0]
        binding_name = f"{dist_info}/{_CLIENT_PATH_RECORD}"
        entries = [
            (info, source.read(info.filename))
            for info in infos
            if info.filename not in {record_name, binding_name}
        ]
        template = next(info for info, _ in entries if info.filename.endswith(".dist-info/METADATA"))

    binding_data = json.dumps(str(client)).encode("utf-8")
    binding_info = zipfile.ZipInfo(binding_name, date_time=template.date_time)
    binding_info.compress_type = zipfile.ZIP_DEFLATED
    binding_info.external_attr = 0o100644 << 16
    entries.append((binding_info, binding_data))

    rows = [_record_row(info.filename, data) for info, data in entries]
    rows.append((record_name, "", ""))
    record_buffer = io.StringIO(newline="")
    csv.writer(record_buffer, lineterminator="\n").writerows(rows)
    record_data = record_buffer.getvalue().encode("utf-8")
    record_info = zipfile.ZipInfo(record_name, date_time=template.date_time)
    record_info.compress_type = zipfile.ZIP_DEFLATED
    record_info.external_attr = 0o100644 << 16

    with tempfile.NamedTemporaryFile(dir=wheel.parent, suffix=".whl", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w") as destination:
            for info, data in entries:
                destination.writestr(info, data)
            destination.writestr(record_info, record_data)
        os.replace(temporary_path, wheel)
    finally:
        temporary_path.unlink(missing_ok=True)


def _build_with_binding(builder, wheel_directory, config_settings=None, metadata_directory=None):
    forwarded, raw_client = _backend_settings(config_settings)
    client = _validated_client_path(raw_client)
    wheel_name = builder(wheel_directory, forwarded, metadata_directory)
    if client is not None:
        _add_client_binding(Path(wheel_directory) / wheel_name, client)
    return wheel_name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _build_with_binding(
        _setuptools.build_wheel,
        wheel_directory,
        config_settings,
        metadata_directory,
    )


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _build_with_binding(
        _setuptools.build_editable,
        wheel_directory,
        config_settings,
        metadata_directory,
    )


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    forwarded, raw_client = _backend_settings(config_settings)
    client = _validated_client_path(raw_client)
    dist_info = _setuptools.prepare_metadata_for_build_wheel(metadata_directory, forwarded)
    if client is not None:
        (Path(metadata_directory) / dist_info / _CLIENT_PATH_RECORD).write_text(
            json.dumps(str(client)), encoding="utf-8"
        )
    return dist_info


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    forwarded, raw_client = _backend_settings(config_settings)
    client = _validated_client_path(raw_client)
    dist_info = _setuptools.prepare_metadata_for_build_editable(metadata_directory, forwarded)
    if client is not None:
        (Path(metadata_directory) / dist_info / _CLIENT_PATH_RECORD).write_text(
            json.dumps(str(client)), encoding="utf-8"
        )
    return dist_info


def get_requires_for_build_wheel(config_settings=None):
    forwarded, _ = _backend_settings(config_settings)
    return _setuptools.get_requires_for_build_wheel(forwarded)


def get_requires_for_build_editable(config_settings=None):
    forwarded, _ = _backend_settings(config_settings)
    return _setuptools.get_requires_for_build_editable(forwarded)


def get_requires_for_build_sdist(config_settings=None):
    forwarded, _ = _backend_settings(config_settings)
    return _setuptools.get_requires_for_build_sdist(forwarded)


def build_sdist(sdist_directory, config_settings=None):
    forwarded, _ = _backend_settings(config_settings)
    return _setuptools.build_sdist(sdist_directory, forwarded)
