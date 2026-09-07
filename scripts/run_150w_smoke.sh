#!/usr/bin/env bash
# Smoke run against 20260323_150w_anonymized (初赛数据).
#
# Reads a subset of parquet row groups so it finishes in a few minutes on a
# single GPU. To run the full data, drop `train_row_groups` from the JSON
# config (or set to 0).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

RECIPE="${1:-hyformer}"     # hyformer | onetrans | both
DEVICE="${DEVICE:-auto}"

run_config() {
    local name="$1"
    local config="${REPO_ROOT}/configs/${name}_150w_smoke.json"
    if [[ ! -f "${config}" ]]; then
        echo "config missing: ${config}" >&2
        exit 1
    fi
    echo "=== running ${name} on 150w preliminary data ==="
    python -m algo26bench.bench.run --config "${config}"
}

case "${RECIPE}" in
    hyformer|onetrans)
        run_config "${RECIPE}"
        ;;
    both)
        run_config "hyformer"
        run_config "onetrans"
        ;;
    *)
        echo "usage: $0 [hyformer|onetrans|both]" >&2
        exit 2
        ;;
esac
