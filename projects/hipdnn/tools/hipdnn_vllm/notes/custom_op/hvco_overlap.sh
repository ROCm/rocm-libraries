#!/bin/bash
# hipdnn_vllm custom_op on a shared allocation (srun --overlap).
#   STEP=setup: clone BRANCH, venv with the pins + wheel (CPU only), in node /tmp/hvco_ov
#   STEP=test:  unit + gpu tests on HIP_VISIBLE_DEVICES (one free GPU)
#   STEP=clean: remove /tmp/hvco_ov
set -uo pipefail
BRANCH=${BRANCH:-users/sareeder/hipdnn-vllm-custom-op}
WHL_DIR=${WHL_DIR:-$HOME/mk/hvco/whl_0398903}
W=/tmp/hvco_ov
T=$W/src/projects/hipdnn/tools/hipdnn_vllm
echo "== $STEP $(date -Is) $(hostname) HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}"
case $STEP in
setup)
  rm -rf $W && mkdir -p $W && cd $W
  git clone -q --depth 1 --branch "$BRANCH" https://github.com/ROCm/rocm-libraries.git src || exit 1
  echo "src $(git -C src log --oneline -1) whl=$(ls $WHL_DIR)"
  set -- $(python3.12 -c "import tomllib;p=tomllib.load(open('$T/BUILD_PINS.toml','rb'))['pins'];print(p['vllm'],p['rocm'],p['vllm_index'])")
  python3.12 -m venv $W/venv && $W/venv/bin/pip install -q --upgrade pip
  # whl-next carries rocm-sdk-devel 10.0.0 but also torchvision 0.29.0, built
  # against another torch; pin the vLLM index's torchvision (matches its torch).
  $W/venv/bin/pip install -q "vllm==$1" "rocm[devel]==$2" "torchvision==0.28.0+rocm10.0.0" pytest $WHL_DIR/*.whl \
    --extra-index-url "$3" --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ || exit 1
  $W/venv/bin/python -c "import json,torch,torchvision,hipdnn_vllm,os;print(torch.__version__,torchvision.__version__,torch.ops.torchvision.nms);d=json.load(open(os.path.join(os.path.dirname(hipdnn_vllm.__file__),'_build_info.json')));print(d['plugin_commit'],d['plugin_tree_dirty_files'])"
  mkdir -p $W/t && cp -r $T/tests $W/t/ && mkdir -p $W/standin && tar xzf $WHL_DIR/standin.tgz -C $W/standin
  ;;
test)
  cd $W/t || exit 1
  echo "arch $(rocminfo | grep -m1 -oE 'gfx[0-9a-f]+')"
  export FLYDSL_RUNTIME_ENABLE_CACHE=0
  P="$W/venv/bin/python -m pytest -p no:cacheprovider -rfs"
  echo "== unit"; $P -q tests/unit 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed" | cut -c1-240
  echo "== custom_op (production catalog)"; $P -v tests/gpu/test_custom_op.py 2>&1 | grep -E "PASS|FAIL|SKIP|passed|failed"
  echo "== custom_op (stand-in catalog)"; HIPDNN_DESCRIPTOR_DIR=$W/standin/integration $P -v tests/gpu/test_custom_op.py -k megakernel 2>&1 | grep -E "PASS|FAIL|SKIP|passed|failed"
  echo "== gpu (not conv)"; timeout 1800 $P -q tests/gpu -m "not conv" 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed|error" | cut -c1-240
  ;;
one)
  cd $W/t || exit 1
  nvidia-smi >/dev/null 2>&1; rocm-smi --showmemuse 2>/dev/null | grep -E "GPU\[" | head -4
  $W/venv/bin/python -m pytest -p no:cacheprovider -x -q $TESTS 2>&1 | grep -E "^E |Error|passed|failed" | head -30
  ;;
cmp)
  # Same failing GPU tests on PR #14255's head wheel+tests, then on ours.
  export FLYDSL_RUNTIME_ENABLE_CACHE=0
  echo "arch $(rocminfo | grep -m1 -oE 'gfx[0-9a-f]+')"
  rm -rf $W/tbase && mkdir -p $W/tbase && tar xzf $HOME/mk/hvco/whl_base/tests.tgz -C $W/tbase
  for side in base ours; do
    if [ $side = base ]; then whl=$HOME/mk/hvco/whl_base/*.whl; dir=$W/tbase; else whl=$HOME/mk/hvco/whl/*.whl; dir=$W/t; fi
    $W/venv/bin/pip install -q --force-reinstall --no-deps $whl
    echo "== $side $($W/venv/bin/python -c "import json,hipdnn_vllm,os;print(json.load(open(os.path.join(os.path.dirname(hipdnn_vllm.__file__),'_build_info.json')))['plugin_commit'])")"
    (cd $dir && $W/venv/bin/python -m pytest -p no:cacheprovider -q -rf $TESTS 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed" | cut -c1-200 | tail -12)
  done
  ;;
clean) rm -rf $W ;;
esac
echo "== end $STEP $(date -Is)"
