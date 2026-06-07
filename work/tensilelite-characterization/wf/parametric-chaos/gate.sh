#!/usr/bin/env bash
# Methodology-A 4-process partition gate (HANDOFF-codegen-coverage.md §3), verbatim command set.
# No-regression GUARD for the parametric-chaos Run-1 add-only tests: expect TOTAL >= 80.60% AND 0 failed.
# Run by the DRIVER (main thread), never by a workflow agent. Never push, never commit here.
set -uo pipefail
CON=tl-char
PROJ=/work/projects/hipblaslt/tensilelite
U=Tensile/Tests/unit

echo "== Part A — bulk, isolating the 3 full-Tensile-flow suites =="
docker exec -e PYTHONPATH=$PROJ -e COVERAGE_FILE=$PROJ/.coverage.g_main -w $PROJ $CON \
  pytest -p no:cacheprovider -m unit --cov=Tensile --cov=rocisa --cov-config=pyproject.toml -n4 -q $U \
  --ignore=$U/test_cpu_only_switch.py --ignore=$U/characterization/ClientPath \
  --ignore=$U/characterization/TensileCreateLibraryRun
A=$?

echo "== Parts B/C/D — each isolated in its own process =="
for p in "g_cpu:$U/test_cpu_only_switch.py" "g_client:$U/characterization/ClientPath" \
         "g_tcl:$U/characterization/TensileCreateLibraryRun"; do
  docker exec -e PYTHONPATH=$PROJ -e COVERAGE_FILE=$PROJ/.coverage.${p%%:*} -w $PROJ $CON \
    pytest -p no:cacheprovider -m unit --cov=Tensile --cov=rocisa --cov-config=pyproject.toml -q ${p#*:}
done

echo "== Combine + report =="
docker exec -e COVERAGE_FILE=$PROJ/.coverage.g_combined -w $PROJ $CON \
  coverage combine --keep $PROJ/.coverage.g_main $PROJ/.coverage.g_cpu $PROJ/.coverage.g_client $PROJ/.coverage.g_tcl
docker exec -e COVERAGE_FILE=$PROJ/.coverage.g_combined -w $PROJ $CON coverage report | tail -1
echo "Part A pytest exit: $A  (expect 0 failed; TOTAL >= 80.60% is the no-regression guard)"
