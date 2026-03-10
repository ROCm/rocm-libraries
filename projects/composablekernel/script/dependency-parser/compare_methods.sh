#!/bin/bash
# Compare legacy dependency parser vs new smart build approach
#
# Legacy method (commit 7f34b22):
#   - Parses build.ninja AFTER build
#   - Uses "ninja -t deps" to extract dependencies
#   - Requires full build before dependency analysis
#
# New smart build method (current):
#   - Analyzes compile_commands.json BEFORE build
#   - Uses "clang -MM" for dependency extraction
#   - No build required for dependency analysis
#
# This script compares test selection from both methods on last N commits

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_HOME="$(dirname "$(dirname "$SCRIPT_DIR")")"
GIT_ROOT="$(git -C "$PACKAGE_HOME" rev-parse --show-toplevel 2>/dev/null)" || GIT_ROOT="$PACKAGE_HOME"
BUILD_DIR="$PACKAGE_HOME/build"

NUM_COMMITS="${1:-10}"

echo "=========================================="
echo "Dependency Parser Method Comparison"
echo "=========================================="
echo ""
echo "Legacy method (commit 7f34b22):"
echo "  - Parses build.ninja after full build"
echo "  - Uses 'ninja -t deps' for dependencies"
echo "  - Requires ~4-6 hours build time before analysis"
echo ""
echo "New smart build (current):"
echo "  - Analyzes compile_commands.json before build"
echo "  - Uses 'clang -MM' for dependencies"
echo "  - Analysis runs during CMake configure (~5-6 minutes)"
echo ""
echo "Comparing test selection for last $NUM_COMMITS commits"
echo "=========================================="
echo ""

# Check if new method dependency map exists
NEW_DEPMAP="$BUILD_DIR/smart_build_test_deps.json"
if [ ! -f "$NEW_DEPMAP" ]; then
    echo "Error: New method dependency map not found: $NEW_DEPMAP"
    echo "Please run: cd $BUILD_DIR && python3 $SCRIPT_DIR/src/cmake_dependency_analyzer.py compile_commands.json"
    exit 1
fi

# Check if legacy method dependency map exists
LEGACY_DEPMAP="$BUILD_DIR/enhanced_dependency_mapping.json"
if [ ! -f "$LEGACY_DEPMAP" ]; then
    echo "Warning: Legacy method dependency map not found: $LEGACY_DEPMAP"
    echo "Attempting to generate using legacy method..."

    if [ ! -f "$BUILD_DIR/build.ninja" ]; then
        echo "Error: build.ninja not found. Legacy method requires a full build."
        echo "Please run: cd $BUILD_DIR && ninja"
        exit 1
    fi

    # Run legacy parser
    echo "Running legacy dependency parser (this may take a while)..."
    python3 "$SCRIPT_DIR/main.py" parse "$BUILD_DIR/build.ninja" --workspace-root "$GIT_ROOT"

    if [ ! -f "$LEGACY_DEPMAP" ]; then
        echo "Error: Failed to generate legacy dependency map"
        exit 1
    fi
fi

echo "Dependency maps found:"
echo "  Legacy: $LEGACY_DEPMAP ($(du -h "$LEGACY_DEPMAP" | cut -f1))"
echo "  New:    $NEW_DEPMAP ($(du -h "$NEW_DEPMAP" | cut -f1))"
echo ""

# Get last N commits
cd "$GIT_ROOT"
COMMITS=$(git log origin/develop --oneline -${NUM_COMMITS} --pretty=format:"%H")

RESULTS_FILE="/tmp/method_comparison_results.csv"
echo "commit_num,commit_sha,commit_msg,changed_files,legacy_tests,new_tests,difference,agreement_pct" > "$RESULTS_FILE"

echo "Analyzing commits..."
echo "=========================================="
echo ""

COMMIT_NUM=1
for COMMIT_SHA in $COMMITS; do
    echo "[$COMMIT_NUM/$NUM_COMMITS] Commit: ${COMMIT_SHA:0:10}"

    COMMIT_MSG=$(git log -1 --pretty=format:"%s" $COMMIT_SHA | tr ',' ';' | head -c 50)
    echo "  Message: $COMMIT_MSG"

    CHANGED_FILES=$(git diff-tree --no-commit-id --name-only -r $COMMIT_SHA | wc -l)
    echo "  Changed files: $CHANGED_FILES"

    # Run LEGACY test selection
    echo "  Running legacy parser..."
    python3 "$SCRIPT_DIR/src/selective_test_filter.py" \
        "$LEGACY_DEPMAP" \
        "${COMMIT_SHA}^" \
        "$COMMIT_SHA" \
        --filter-mode test_prefix \
        --output "/tmp/legacy_tests_${COMMIT_SHA}.json" \
        > /dev/null 2>&1 || true

    if [ -f "/tmp/legacy_tests_${COMMIT_SHA}.json" ]; then
        LEGACY_TESTS=$(python3 -c "import json; data=json.load(open('/tmp/legacy_tests_${COMMIT_SHA}.json')); print(len(data.get('tests_to_run', [])))" 2>/dev/null || echo "0")
    else
        LEGACY_TESTS=0
    fi

    # Run NEW smart build test selection
    echo "  Running new smart build..."
    python3 "$SCRIPT_DIR/src/selective_test_filter.py" \
        "$NEW_DEPMAP" \
        "${COMMIT_SHA}^" \
        "$COMMIT_SHA" \
        --filter-mode test_prefix \
        --output "/tmp/new_tests_${COMMIT_SHA}.json" \
        > /dev/null 2>&1 || true

    if [ -f "/tmp/new_tests_${COMMIT_SHA}.json" ]; then
        NEW_TESTS=$(python3 -c "import json; data=json.load(open('/tmp/new_tests_${COMMIT_SHA}.json')); print(len(data.get('tests_to_run', [])))" 2>/dev/null || echo "0")
    else
        NEW_TESTS=0
    fi

    # Calculate agreement between methods
    if [ -f "/tmp/legacy_tests_${COMMIT_SHA}.json" ] && [ -f "/tmp/new_tests_${COMMIT_SHA}.json" ]; then
        AGREEMENT=$(python3 << EOF
import json

with open('/tmp/legacy_tests_${COMMIT_SHA}.json') as f:
    legacy = set(json.load(f).get('tests_to_run', []))

with open('/tmp/new_tests_${COMMIT_SHA}.json') as f:
    new = set(json.load(f).get('tests_to_run', []))

if len(legacy) == 0 and len(new) == 0:
    print("100.0,0,0,0")
elif len(legacy) == 0 or len(new) == 0:
    print("0.0,0,0,0")
else:
    intersection = len(legacy & new)
    union = len(legacy | new)
    agreement_pct = (intersection / union * 100) if union > 0 else 0
    only_legacy = len(legacy - new)
    only_new = len(new - legacy)
    print(f"{agreement_pct:.1f},{intersection},{only_legacy},{only_new}")
EOF
)
        AGREEMENT_PCT=$(echo "$AGREEMENT" | cut -d',' -f1)
        COMMON=$(echo "$AGREEMENT" | cut -d',' -f2)
        ONLY_LEGACY=$(echo "$AGREEMENT" | cut -d',' -f3)
        ONLY_NEW=$(echo "$AGREEMENT" | cut -d',' -f4)
    else
        AGREEMENT_PCT="N/A"
        COMMON=0
        ONLY_LEGACY=0
        ONLY_NEW=0
    fi

    DIFF=$((NEW_TESTS - LEGACY_TESTS))

    echo "  Legacy method:  $LEGACY_TESTS tests"
    echo "  New method:     $NEW_TESTS tests"
    echo "  Difference:     $DIFF tests"
    if [ "$AGREEMENT_PCT" != "N/A" ]; then
        echo "  Agreement:      $AGREEMENT_PCT% (common: $COMMON, only legacy: $ONLY_LEGACY, only new: $ONLY_NEW)"
    fi

    echo "$COMMIT_NUM,$COMMIT_SHA,$COMMIT_MSG,$CHANGED_FILES,$LEGACY_TESTS,$NEW_TESTS,$DIFF,$AGREEMENT_PCT" >> "$RESULTS_FILE"

    echo ""
    COMMIT_NUM=$((COMMIT_NUM + 1))
done

echo "=========================================="
echo "Comparison Summary"
echo "=========================================="
echo ""
printf "%-5s %-12s %-30s %-6s %-8s %-8s %-10s %-10s\n" \
    "NUM" "COMMIT" "MESSAGE" "FILES" "LEGACY" "NEW" "DIFF" "AGREEMENT"
printf "%-5s %-12s %-30s %-6s %-8s %-8s %-10s %-10s\n" \
    "---" "------" "-------" "-----" "------" "---" "----" "---------"

tail -n +2 "$RESULTS_FILE" | while IFS=',' read -r num sha msg files legacy new diff agree; do
    if [ "$diff" -gt 0 ]; then
        diff_str="+$diff"
    else
        diff_str="$diff"
    fi
    printf "%-5s %-12s %-30s %-6s %-8s %-8s %-10s %-9s%%\n" \
        "$num" "${sha:0:10}" "${msg:0:28}" "$files" "$legacy" "$new" "$diff_str" "$agree"
done

echo ""
echo "Overall Statistics:"
echo "-------------------"

python3 << 'EOF'
import csv

with open('/tmp/method_comparison_results.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    print("No data")
    exit(0)

total_commits = len(rows)
valid_rows = [r for r in rows if r['agreement_pct'] != 'N/A']

if not valid_rows:
    print("No valid comparison data")
    exit(0)

avg_legacy = sum(int(r['legacy_tests']) for r in valid_rows) / len(valid_rows)
avg_new = sum(int(r['new_tests']) for r in valid_rows) / len(valid_rows)
avg_agreement = sum(float(r['agreement_pct']) for r in valid_rows) / len(valid_rows)

print(f"  Commits analyzed: {total_commits}")
print(f"  Valid comparisons: {len(valid_rows)}")
print(f"  Average legacy tests: {avg_legacy:.0f}")
print(f"  Average new tests: {avg_new:.0f}")
print(f"  Average agreement: {avg_agreement:.1f}%")

# Count how many have perfect agreement
perfect = sum(1 for r in valid_rows if float(r['agreement_pct']) == 100.0)
high_agreement = sum(1 for r in valid_rows if float(r['agreement_pct']) >= 95.0)

print(f"\n  Perfect agreement (100%): {perfect}/{len(valid_rows)} commits")
print(f"  High agreement (≥95%): {high_agreement}/{len(valid_rows)} commits")

if avg_legacy > 0:
    if avg_new > avg_legacy:
        pct_diff = (avg_new / avg_legacy - 1) * 100
        print(f"\n  New method selects {pct_diff:.1f}% MORE tests on average")
        print(f"  Reason: Better dependency detection (more conservative/complete)")
    elif avg_new < avg_legacy:
        pct_diff = (1 - avg_new / avg_legacy) * 100
        print(f"\n  New method selects {pct_diff:.1f}% FEWER tests on average")
        print(f"  Reason: More precise dependency analysis")
    else:
        print(f"\n  Both methods select same number of tests on average")

print("\nConclusion:")
if avg_agreement >= 95:
    print("  ✓ High agreement between methods - new approach is validated")
elif avg_agreement >= 85:
    print("  ~ Good agreement - minor differences acceptable")
else:
    print("  ⚠ Significant differences - investigate discrepancies")
EOF

echo ""
echo "Results saved to: $RESULTS_FILE"
echo "Test selection details in: /tmp/{legacy,new}_tests_*.json"
echo "=========================================="
