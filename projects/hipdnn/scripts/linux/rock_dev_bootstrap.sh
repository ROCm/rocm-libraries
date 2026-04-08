#!/bin/bash
# TheRock developer bootstrap and build tool.
#
# Downloads prebuilt CI artifacts and lets you selectively build
# ml-libs components (hipdnn, hipkernelprovider, etc.) from source
# while everything else uses prebuilt binaries.
#
# Usage:
#   ./dev_bootstrap.sh <command> [options] [components...]
#
# Commands:
#   bootstrap [run-id]                  Download CI artifacts (auto-detects latest if omitted)
#   configure [components...]           Remove .prebuilt markers + run cmake
#   build [components...]               Build components with ninja
#   rebuild [components...]             Expunge + rebuild components
#
# Examples:
#   # One-time setup (auto-detect latest nightly):
#   ./dev_bootstrap.sh bootstrap
#
#   # One-time setup with specific run:
#   ./dev_bootstrap.sh bootstrap 22884930750
#
#   # Configure to build hipdnn + miopenprovider from source:
#   ./dev_bootstrap.sh configure hipdnn miopenprovider
#
#   # Configure all default components from source:
#   ./dev_bootstrap.sh configure
#
#   # Build everything that was configured:
#   ./dev_bootstrap.sh build
#
#   # Build just one component:
#   ./dev_bootstrap.sh build miopenprovider
#
#   # Clean rebuild of a component:
#   ./dev_bootstrap.sh rebuild hipkernelprovider
#
# Global options (work with any command):
#   --gpu <family>      GPU family (default: gfx94X-dcgpu)
#   --build-dir <dir>   Build directory (default: ~/therock-build-<gpu>)
#   --workflow <file>    Workflow for auto-detect (default: ci_nightly.yml)
#   -h, --help          Show this help
#
# Available components:
#   hipdnn, hipkernelprovider, miopenprovider, hipblasltprovider
#
# Per-component ninja targets (for manual use):
#   component+configure  - Run cmake configure
#   component+build      - Incremental build
#   component+stage      - Install to stage dir
#   component+expunge    - Full clean (build/, stage/, stamp/, dist/)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# All known components that can be built from source
ALL_COMPONENTS=(hipdnn hipkernelprovider miopenprovider hipblasltprovider)

# Ninja target names (must match cmake subproject names exactly)
declare -A NINJA_TARGETS=(
    [hipdnn]="hipDNN"
    [hipkernelprovider]="hipkernelprovider"
    [miopenprovider]="miopenprovider"
    [hipblasltprovider]="hipblasltprovider"
)

# ---- Defaults ----
GPU_FAMILY="gfx94X-dcgpu"
BUILD_DIR=""
WORKFLOW="ci_nightly.yml"
COMMAND=""
RUN_ID=""
COMPONENTS=()

# ---- Helpers ----

# Convert component name to cmake enable flag name (e.g., hipdnn -> HIPDNN)
component_to_flag() {
    echo "$1" | tr '[:lower:]-' '[:upper:]_'
}

# Convert component name to ninja target name (e.g., hipdnn -> hipDNN)
component_to_ninja_target() {
    echo "${NINJA_TARGETS[$1]:-$1}"
}

# Validate that a component name is known
validate_component() {
    local comp="$1"
    for known in "${ALL_COMPONENTS[@]}"; do
        if [[ "$comp" == "$known" ]]; then
            return 0
        fi
    done
    echo "ERROR: Unknown component '$comp'"
    echo "Available components: ${ALL_COMPONENTS[*]}"
    exit 1
}

# Config file in build dir to remember settings between commands
config_file() {
    echo "$BUILD_DIR/.dev_bootstrap_config"
}

save_config() {
    local cfg
    cfg=$(config_file)
    mkdir -p "$BUILD_DIR"
    cat > "$cfg" <<EOF
GPU_FAMILY=$GPU_FAMILY
COMPONENTS=(${COMPONENTS[*]})
EOF
}

load_config() {
    local cfg
    cfg=$(config_file)
    if [ -f "$cfg" ]; then
        source "$cfg"
    fi
}

# ---- Usage ----
usage() {
    echo "Usage: $0 <command> [options] [components...]"
    echo ""
    echo "Commands:"
    echo "  bootstrap [run-id]       Download CI artifacts (auto-detects latest if omitted)"
    echo "  configure [components]   Remove .prebuilt markers + run cmake"
    echo "  build [components]       Build with ninja"
    echo "  rebuild [components]     Expunge + rebuild"
    echo ""
    echo "Options:"
    echo "  --gpu <family>           GPU family (default: gfx94X-dcgpu)"
    echo "  --build-dir <dir>        Build directory (default: ~/therock-build-<gpu>)"
    echo "  --workflow <file>        Workflow for auto-detect (default: ci_nightly.yml)"
    echo "  -h, --help               Show this help"
    echo ""
    echo "Components (default: all):"
    for comp in "${ALL_COMPONENTS[@]}"; do
        echo "  - $comp"
    done
    echo ""
    echo "Examples:"
    echo "  $0 bootstrap                          # download latest nightly artifacts"
    echo "  $0 configure hipdnn miopenprovider     # configure those two for source build"
    echo "  $0 build                               # build all configured components"
    echo "  $0 rebuild miopenprovider               # expunge + rebuild one component"
    exit "${1:-0}"
}

# ---- Parse arguments ----

# Need at least one arg (the command)
if [ $# -lt 1 ]; then
    usage 0
fi

# Parse global options and command
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu)
            GPU_FAMILY="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --workflow)
            WORKFLOW="$2"
            shift 2
            ;;
        -h|--help)
            usage 0
            ;;
        bootstrap|configure|build|rebuild)
            COMMAND="$1"
            shift
            break  # remaining args are command-specific
            ;;
        -*)
            echo "ERROR: Unknown option: $1"
            usage 1
            ;;
        *)
            echo "ERROR: Unknown command: $1"
            usage 1
            ;;
    esac
done

if [ -z "$COMMAND" ]; then
    echo "ERROR: No command specified"
    usage 1
fi

# Parse command-specific args (remaining after break)
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu|--build-dir|--workflow)
            # Allow global options after command too
            if [ "$1" = "--gpu" ]; then GPU_FAMILY="$2"; fi
            if [ "$1" = "--build-dir" ]; then BUILD_DIR="$2"; fi
            if [ "$1" = "--workflow" ]; then WORKFLOW="$2"; fi
            shift 2
            ;;
        -h|--help)
            usage 0
            ;;
        -*)
            echo "ERROR: Unknown option: $1"
            usage 1
            ;;
        *)
            # For bootstrap: first positional is run ID (numeric)
            if [[ "$COMMAND" == "bootstrap" ]] && [[ "$1" =~ ^[0-9]+$ ]] && [ -z "$RUN_ID" ]; then
                RUN_ID="$1"
            else
                # Component name
                validate_component "$1"
                COMPONENTS+=("$1")
            fi
            shift
            ;;
    esac
done

# Derive build dir from GPU family if not set
GPU_SHORT="${GPU_FAMILY%%-*}"
if [ -z "$BUILD_DIR" ]; then
    BUILD_DIR="$HOME/therock-build-${GPU_SHORT}"
fi

# Validate script location
if [ ! -f "$SCRIPT_DIR/build_tools/artifact_manager.py" ]; then
    echo "ERROR: This script must be run from the TheRock repo root"
    exit 1
fi

# ---- Setup Python environment ----
setup_python() {
    cd "$SCRIPT_DIR"
    if [ ! -d ".venv" ]; then
        echo "Creating Python virtual environment..."
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    export PYTHONWARNINGS="ignore:Unverified HTTPS request"
}

# ---- Remove .prebuilt markers for given components ----
remove_prebuilt_markers() {
    local comps=("$@")
    echo ""
    echo "Removing .prebuilt markers for source-build components..."
    echo ""

    for comp in "${comps[@]}"; do
        PREBUILT_FILES=$(find "$BUILD_DIR" -ipath "*/${comp}/stage.prebuilt" -o -ipath "*/${comp}/*/stage.prebuilt" 2>/dev/null || true)
        if [ -n "$PREBUILT_FILES" ]; then
            echo "  $comp: removing .prebuilt marker"
            for f in $PREBUILT_FILES; do
                rm -f "$f"
                rm -rf "${f%.prebuilt}"
            done
        else
            echo "  $comp: no .prebuilt marker (will build from source)"
        fi
    done
    echo ""
}

# ========================================
# COMMAND: bootstrap
# ========================================
do_bootstrap() {
    setup_python

    # Fetch sources if needed
    if [ ! -f "short-hip-runtime/hip/VERSION" ] && [ ! -f "rocm-systems/projects/hip/VERSION" ]; then
        echo "Fetching submodule sources..."
        python3 build_tools/fetch_sources.py
        echo ""
    fi

    # Find latest run ID if not specified
    if [ -z "$RUN_ID" ]; then
        echo "=========================================="
        echo "Finding latest CI run with artifacts"
        echo "=========================================="
        echo ""
        echo "Searching $WORKFLOW for $GPU_FAMILY artifacts..."
        echo ""

        FIND_OUTPUT=$(python build_tools/find_latest_artifacts.py \
            --artifact-group "$GPU_FAMILY" \
            --workflow "$WORKFLOW" \
            --verbose 2>&1)

        if [ $? -ne 0 ]; then
            echo "$FIND_OUTPUT"
            echo ""
            echo "ERROR: Could not find a recent CI run with artifacts for $GPU_FAMILY"
            echo ""
            echo "Try specifying a run ID manually:"
            echo "  $0 bootstrap <run-id>"
            echo ""
            echo "Find run IDs at:"
            echo "  https://github.com/ROCm/TheRock/actions/workflows/$WORKFLOW"
            exit 1
        fi

        RUN_ID=$(echo "$FIND_OUTPUT" | grep "Workflow run ID:" | head -1 | awk '{print $NF}')

        if [ -z "$RUN_ID" ]; then
            echo "$FIND_OUTPUT"
            echo "ERROR: Could not parse run ID from output"
            exit 1
        fi

        echo "$FIND_OUTPUT"
        echo ""
        echo "Using run ID: $RUN_ID"
    fi

    echo ""
    echo "=========================================="
    echo "Fetching all artifacts from run $RUN_ID"
    echo "=========================================="
    echo ""
    echo "GPU Family: $GPU_FAMILY"
    echo "Build Dir:  $BUILD_DIR"
    echo ""

    python build_tools/artifact_manager.py fetch \
        --stage all \
        --run-id "$RUN_ID" \
        --amdgpu-families "$GPU_FAMILY" \
        --output-dir "$BUILD_DIR" \
        --bootstrap

    echo ""
    echo "=========================================="
    echo "Bootstrap complete!"
    echo "=========================================="
    PREBUILT_COUNT=$(find "$BUILD_DIR" -name "*.prebuilt" 2>/dev/null | wc -l)
    echo ""
    echo "  $PREBUILT_COUNT components marked as prebuilt"
    echo ""
    echo "Next: $0 configure [components...]"
    echo ""
}

# ========================================
# COMMAND: configure
# ========================================
do_configure() {
    setup_python

    # Default to all components if none specified
    if [ ${#COMPONENTS[@]} -eq 0 ]; then
        COMPONENTS=("${ALL_COMPONENTS[@]}")
    fi

    echo "=========================================="
    echo "Configuring TheRock build"
    echo "=========================================="
    echo ""
    echo "GPU Family: $GPU_FAMILY"
    echo "Build Dir:  $BUILD_DIR"
    echo ""
    echo "Building from source:"
    for comp in "${COMPONENTS[@]}"; do
        echo "  - $comp"
    done

    # Remove .prebuilt markers
    remove_prebuilt_markers "${COMPONENTS[@]}"

    # Save config for build/rebuild commands
    save_config

    # Build cmake args
    CMAKE_ARGS=(
        -B "$BUILD_DIR"
        -GNinja
        -DTHEROCK_ENABLE_ALL=OFF
        -DTHEROCK_AMDGPU_FAMILIES="$GPU_FAMILY"
        -DCMAKE_BUILD_TYPE=RelWithDebInfo
    )

    for comp in "${COMPONENTS[@]}"; do
        flag=$(component_to_flag "$comp")
        CMAKE_ARGS+=("-DTHEROCK_ENABLE_${flag}=ON")
    done

    echo "Running cmake..."
    echo "  cmake ${CMAKE_ARGS[*]} ."
    echo ""

    cd "$SCRIPT_DIR"
    cmake "${CMAKE_ARGS[@]}" .

    echo ""
    echo "=========================================="
    echo "Configure complete!"
    echo "=========================================="
    echo ""
    echo "Next: $0 build [components...]"
    echo ""
}

# ========================================
# COMMAND: build
# ========================================
do_build() {
    # Load saved config if no components specified
    if [ ${#COMPONENTS[@]} -eq 0 ]; then
        load_config
    fi

    if [ ${#COMPONENTS[@]} -eq 0 ]; then
        echo "ERROR: No components specified and no saved config found."
        echo "Run '$0 configure [components...]' first."
        exit 1
    fi

    # Resolve to ninja target names
    local TARGETS=()
    for comp in "${COMPONENTS[@]}"; do
        TARGETS+=("$(component_to_ninja_target "$comp")")
    done

    echo "=========================================="
    echo "Building components"
    echo "=========================================="
    echo ""
    echo "Build Dir: $BUILD_DIR"
    echo "Targets:   ${TARGETS[*]}"
    echo ""

    ninja -C "$BUILD_DIR" "${TARGETS[@]}"

    echo ""
    echo "Build complete!"
    echo ""
}

# ========================================
# COMMAND: rebuild
# ========================================
do_rebuild() {
    # Load saved config if no components specified
    if [ ${#COMPONENTS[@]} -eq 0 ]; then
        load_config
    fi

    if [ ${#COMPONENTS[@]} -eq 0 ]; then
        echo "ERROR: No components specified and no saved config found."
        echo "Run '$0 configure [components...]' first."
        exit 1
    fi

    # Resolve to ninja target names
    local TARGETS=()
    for comp in "${COMPONENTS[@]}"; do
        TARGETS+=("$(component_to_ninja_target "$comp")")
    done

    echo "=========================================="
    echo "Rebuilding components (expunge + build)"
    echo "=========================================="
    echo ""
    echo "Build Dir: $BUILD_DIR"
    echo "Targets:   ${TARGETS[*]}"
    echo ""

    # Expunge each component
    EXPUNGE_TARGETS=()
    for t in "${TARGETS[@]}"; do
        EXPUNGE_TARGETS+=("${t}+expunge")
    done

    echo "Expunging: ${EXPUNGE_TARGETS[*]}"
    ninja -C "$BUILD_DIR" "${EXPUNGE_TARGETS[@]}"

    echo ""
    echo "Building: ${TARGETS[*]}"
    ninja -C "$BUILD_DIR" "${TARGETS[@]}"

    echo ""
    echo "Rebuild complete!"
    echo ""
}

# ---- Dispatch command ----
case "$COMMAND" in
    bootstrap)
        do_bootstrap
        ;;
    configure)
        do_configure
        ;;
    build)
        do_build
        ;;
    rebuild)
        do_rebuild
        ;;
esac
