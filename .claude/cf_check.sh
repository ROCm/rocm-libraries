#!/usr/bin/env bash
CF=/usr/lib/llvm-18/bin/clang-format
while read -r f; do
  [ -z "$f" ] && continue
  if ! "$CF" -style=file --dry-run -Werror "$f" >/dev/null 2>&1; then
    echo "NEEDS-FORMAT: $f"
  fi
done < /tmp/cxx_files.txt
echo "done"
