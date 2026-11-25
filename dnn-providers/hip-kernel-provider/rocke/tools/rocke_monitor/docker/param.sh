#!/bin/sh
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# One download location of the image's build (docker/Dockerfile), as a build step reads it: the build secret ID (its
# first line, from /run/secrets/ID). A secret is in no layer's history and no build log, so a location that names an
# internal host stays out of the image. The build is given each location it needs (rkmon build: rkmon_lib/build_plan.py,
# which gives the package's public default for a blank key), so no step here has a default of its own.
#
# Usage (in a build step that mounts the secret: RUN --mount=type=secret,id=ID ...):
#   param.sh ID              prints the value; a build that was given none stops here (exit 2)
#   param.sh --optional ID   prints the value, or nothing when the build was given none
# It prints nothing else, and never a message that names the value. SECRETS_DIR moves /run/secrets (the tests).
set -eu
optional=""
if [ "${1:-}" = --optional ]; then
    optional=1
    shift
fi
id=${1:?usage: param.sh [--optional] ID}
f="${SECRETS_DIR:-/run/secrets}/$id"
value=""
if [ -s "$f" ]; then
    value=$(sed -n '1{s/[[:space:]]*$//;p;}' "$f")
fi
if [ -z "$value" ] && [ -z "$optional" ]; then
    echo "param.sh: the build was given no build secret $id: build the image with rkmon build, which gives each one" >&2
    exit 2
fi
printf '%s\n' "$value"
