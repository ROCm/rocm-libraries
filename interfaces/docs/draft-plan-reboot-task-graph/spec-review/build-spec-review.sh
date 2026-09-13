#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
docs_dir=$(cd "${script_dir}/../.." && pwd)
spec_dir="${docs_dir}/draft-plan-reboot-specs"
output_dir="${docs_dir}/draft-plan-reboot-assets/specifications"

mkdir -p "${output_dir}"
cp "${script_dir}/spec-review.css" "${output_dir}/spec-review.css"

render() {
  local input=$1
  local output=$2
  local title=$3
  pandoc --from=gfm --to=html5 --standalone --toc --toc-depth=3 \
    --css=spec-review.css \
    --include-before-body="${script_dir}/spec-navigation.html" \
    --lua-filter="${script_dir}/rewrite-review-links.lua" \
    --metadata "title=${title}" \
    --output="${output_dir}/${output}.html" \
    "${input}"
}

render "${script_dir}/index.md" index "ROCm interfaces specifications"
render "${spec_dir}/architecture-component-model.md" architecture-component-model "Architecture component model specification"
render "${spec_dir}/broker.md" broker "Broker specification"
render "${spec_dir}/cohort.md" cohort "Cohort specification"
render "${spec_dir}/facade.md" facade "Facade specification"
render "${spec_dir}/hipblaslt-horizontal.md" hipblaslt-horizontal "hipBLASLt horizontal C API and ABI specification"
render "${spec_dir}/hipblaslt-facade-path.md" hipblaslt-facade-path "hipBLASLt facade path specification"
render "${spec_dir}/manifest.md" manifest "Provider manifest specification"
render "${spec_dir}/provider-adapter.md" provider-adapter "Provider adapter specification"
render "${spec_dir}/provider-binding.md" provider-binding "Provider binding specification"
render "${spec_dir}/provider-module.md" provider-module "Provider module specification"
render "${spec_dir}/provider-protocol.md" provider-protocol "Provider protocol specification"
render "${spec_dir}/provider.md" provider "Provider specification"
