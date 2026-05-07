#!/bin/sh
# Generate objcopy symbol renaming definitions for ALL OpenBLAS symbols
# Usage: prefix_openblas_symbols.sh <lib_dir> <prefix> <output_def>

LIB_DIR="$1"
PREFIX="$2"
OUTPUT="$3"

# Find libopenblas.a - it might be in lib/ or lib/DEBUG/ or lib/RELEASE/
LIBRARY=""
for dir in "$LIB_DIR" "$LIB_DIR/DEBUG" "$LIB_DIR/RELEASE" "$LIB_DIR/Debug" "$LIB_DIR/Release"; do
    if [ -f "$dir/libopenblas.a" ]; then
        LIBRARY="$dir/libopenblas.a"
        break
    fi
done

if [ -z "$LIBRARY" ] || [ ! -f "$LIBRARY" ]; then
    echo "Error: Library file libopenblas.a not found in $LIB_DIR or subdirectories"
    exit 1
fi

echo "Found library: $LIBRARY"

# Extract all global symbols (text T, data D, BSS B, and other global G) and create objcopy rename definitions
# We need to rename both functions AND global variables to avoid conflicts
nm "$LIBRARY" | grep ' [TDBG] ' | awk '{print $3}' | while read sym; do
    # Skip symbols that already have the prefix
    case "$sym" in
        ${PREFIX}*)
            ;;
        *)
            echo "$sym ${PREFIX}${sym}"
            ;;
    esac
done > "$OUTPUT"

echo "Generated $(wc -l < "$OUTPUT") symbol mappings (functions + global data) in $OUTPUT"

# Apply the renaming
if [ -f "$OUTPUT" ] && [ -s "$OUTPUT" ]; then
    objcopy --redefine-syms="$OUTPUT" "$LIBRARY"
    echo "Applied symbol renaming to $LIBRARY"
else
    echo "Error: No symbol mappings generated"
    exit 1
fi
