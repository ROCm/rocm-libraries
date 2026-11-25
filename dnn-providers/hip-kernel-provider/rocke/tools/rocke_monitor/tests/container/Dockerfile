# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# The released tarball's test (tests/container/tarball.py) on a public Linux base (BASE: Ubuntu LTS by default;
# tarball.py also uses Debian), in a throwaway container:
#   plain    a non-root user and nothing else, python's base
#   python   plain plus Python 3: tests/container/tarball.py mounts the released tarball, which that user unpacks
# The stage runs as the user op. No secret and no internal host is in any image.
ARG BASE=ubuntu:24.04

FROM ${BASE} AS plain
RUN id op >/dev/null 2>&1 || useradd --create-home --shell /bin/sh op
USER op
WORKDIR /home/op
ENV HOME=/home/op LANG=C.UTF-8

FROM plain AS python
USER root
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*
USER op
