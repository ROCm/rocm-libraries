# Troubleshooting

## ROCm PyTorch nightlies (gfx90X-dcgpu staging index)

If the default ROCm index does not have Python 3.12 wheels, install torch from the
staging index explicitly:

```bash
source .venv/bin/activate
python -m pip install --index-url https://rocm.nightlies.amd.com/v2-staging/gfx90X-dcgpu/ \
  --pre --upgrade --force-reinstall torch
```

`torchaudio`/`torchvision` are optional and not required for benchmarking, but can
be installed from the same index if needed.

## Using ROCm libraries from the venv

If you installed ROCm torch from the staging index, prefer the venv ROCm SDK
libraries first to avoid LLVM symbol mismatches:

```bash
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:\
$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx90X_dcgpu/lib:\
$PWD/.venv/lib/python3.12/site-packages/triton/backends/amd/lib:\
$LD_LIBRARY_PATH
```

You can make this venv-agnostic by resolving `site-packages` at runtime:

```bash
VENV_SITE=$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
export LD_LIBRARY_PATH=$VENV_SITE/_rocm_sdk_core/lib:\
$VENV_SITE/_rocm_sdk_libraries_gfx90X_dcgpu/lib:\
$VENV_SITE/triton/backends/amd/lib:\
$LD_LIBRARY_PATH
```
