# Troubleshooting

## Using ROCm libraries from the venv

Prefer the venv ROCm SDK libraries first to avoid LLVM symbol mismatches:

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
