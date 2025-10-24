#!/usr/bin/env bash

#===----------------------------------------------------------------------===#
#
# This script is copied from fusilli:
# https://github.com/nod-ai/shark-ai/blob/52e12fe6bd76b8fad83687d48be1ca067120c310/sharkfuser/build_tools/docker/exec_docker_ci.sh
# It's used to run fusilli-plugin CI in a docker container: https://github.com/sjain-stanford/docker/tree/main
#
# Once fusilli-plugin has been moved to TheRock CI this script (and the
# standalone github actions based CI) should be removed.
#
#===----------------------------------------------------------------------===#


# ROCm requires accesses to the host's /dev/kfd and /dev/dri/* device nodes, typically
# owned by the `render` and `video` groups. The groups' GIDs in the container must
# match the host's to access the resources. Sometimes the device nodes may be owned by
# dynamic GIDs (that don't belong to the `render` or `video` groups). So instead of
# adding user to the GIDs of named groups (obtained from `getent group render` or
# `getent group video`), we simply check the owning GID of the device nodes on the host
# and pass it to `docker run` with `--group-add=<GID>`.
for DEV in /dev/kfd /dev/dri/*; do
  # Skip if not a character device
  # /dev/dri/by-path/ symlinks are ignored
  [[ -c "${DEV}" ]] || continue
  DOCKER_RUN_DEVICE_OPTS+=" --device=${DEV} --group-add=$(stat -c '%g' ${DEV})"
done

# Bind mounts for the following:
# - current directory to /workspace in the container
docker run --rm \
           -v "${PWD}":/workspace \
           ${DOCKER_RUN_DEVICE_OPTS} \
           --security-opt seccomp=unconfined \
           ghcr.io/sjain-stanford/compiler-dev-ubuntu-24.04:main@sha256:d52a5eb21ce21509f5fd1064074ba34f7ad8810c5d5c6caff9790149c8e05b3c \
           "$@"
