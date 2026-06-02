# Characterization-test dev env for tensilelite.
#
# Bakes ONLY the Python deps + build tools on top of the ROCm toolchain image.
# rocisa is NOT baked: its CMake build needs the monorepo siblings
# shared/origami and shared/stinkytofu (at rocisa/../../../../shared/), which
# are absent from the base image. Build rocisa once inside a container that
# mounts the worktree (see env/README.md) — that is the only place those
# siblings resolve, and it matches the documented `invoke rocisa` workflow.
#
# Base provides: ROCm/HIP toolchain (amdclang++, hip headers, hip-config.cmake),
# cmake (therock-tools), Python 3.12 (manylinux layout), ROCM_PATH=/opt/rocm.

FROM rocm-libs-bump-e3b:26531351157

ENV ROCM_PATH=/opt/rocm
ENV PIP_NO_CACHE_DIR=1
# rocisa's _rocisa.so links libamdhip64.so.7 (HIP runtime); put it on the
# loader path so `import rocisa` works at runtime.
ENV LD_LIBRARY_PATH=/opt/rocm/lib

# Core deps required to import Tensile + run unit/characterization tests,
# build rocisa, and measure coverage. Kept faithful to requirements.txt.
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
        packaging pyyaml msgpack "joblib>=1.4.0" numpy \
        "pytest>=5.4.1" "pytest-xdist>=1.32.0" pytest-cov "coverage[toml]>=7.0.0" \
        syrupy \
        filelock invoke memory_profiler \
        "scikit-build-core>=0.10" "nanobind>=2.0" cmake setuptools

# Optional deps from requirements.txt that are not needed for unit/char tests
# (client-run / profiling paths). Tolerant so an index hiccup doesn't fail the
# image; record any miss when running the full suite.
RUN python3 -m pip install simplejson ujson orjson yappi tox hip-python || \
    echo "WARN: some optional deps failed to install (ok for unit/char tests)"

WORKDIR /work
