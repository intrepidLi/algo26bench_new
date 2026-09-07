#!/usr/bin/env bash
# Smoke run against 20260323_150w_anonymized (初赛数据).
#
# Usage:
#   bash scripts/run_150w_smoke.sh [hyformer|onetrans|both]
#
# Environment overrides:
#   NGPUS=4        # 1..8, default = auto-detect from CUDA_VISIBLE_DEVICES / nvidia-smi
#   MASTER_PORT    # override torchrun rendezvous port, default 29500
#
# NGPUS >= 2 launches with torchrun (DDP + NCCL). NGPUS == 1 launches plain python.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

RECIPE="${1:-hyformer}"

# ── GPU count detection ──
if [[ -z "${NGPUS:-}" ]]; then
    if [[ -v CUDA_VISIBLE_DEVICES ]]; then
        if [[ -z "${CUDA_VISIBLE_DEVICES// /}" ]]; then
            NGPUS=0
        else
            NGPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c .)
        fi
    elif command -v nvidia-smi &>/dev/null; then
        NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    else
        NGPUS=1
    fi
fi
export NGPUS
echo "Using ${NGPUS} GPU(s) (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<unset>}')"

MASTER_PORT="${MASTER_PORT:-29500}"

run_config() {
    local name="$1"
    local config="${REPO_ROOT}/configs/${name}_150w_smoke.json"
    if [[ ! -f "${config}" ]]; then
        echo "config missing: ${config}" >&2
        exit 1
    fi
    echo "=== running ${name} on 150w preliminary data (NGPUS=${NGPUS}) ==="
    if (( NGPUS >= 2 )); then
        torchrun \
            --standalone \
            --nproc_per_node="${NGPUS}" \
            --master_port="${MASTER_PORT}" \
            -m algo26bench.bench.run --config "${config}"
    else
        python -m algo26bench.bench.run --config "${config}"
    fi
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
