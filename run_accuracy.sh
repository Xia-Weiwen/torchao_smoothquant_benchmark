#!/usr/bin/env bash
# run_accuracy.sh — Accuracy benchmark (SQuAD / MultiNLI) for SmoothQuant encoder models
#
# Usage:
#   conda activate dev
#   bash run_accuracy.sh
#
# Environment overrides:
#   CORES="0-3"            core range for taskset
#   TASK=all               squad | mnli | all | report
#   MODE=all               fp32 | bf16 | smooth-static | smooth-static-autocast | smooth-dynamic | smooth-dynamic-autocast | all
#   MODEL_KEY=distilbert-base
#                          bert-large | xlm-roberta-base | distilbert-base
#   NUM_SAMPLES=500        validation samples ('all' = entire dataset)
#   NUM_CALIB=32           calibration samples for SmoothQuant
#   AOTI=1                 use AOT Inductor instead of torch.compile
#   REPORT_FORMAT=both     markdown | csv | both
#   REPORT_OUTPUT=path     write report to file (default: stdout)

set -euo pipefail

export LD_PRELOAD="${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export KMP_AFFINITY="${KMP_AFFINITY:-granularity=fine,compact,1,0}"
export ONEDNN_CACHE_CONTEXT_UNSAFE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="${SCRIPT_DIR}/results"
mkdir -p "${OUTDIR}"

CORES="${CORES:-0-3}"
TASK="${TASK:-all}"
MODE="${MODE:-all}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
NUM_CALIB="${NUM_CALIB:-32}"
AOTI="${AOTI:-0}"
REPORT_FORMAT="${REPORT_FORMAT:-both}"
REPORT_OUTPUT="${REPORT_OUTPUT:-}"

ALL_KEYS="bert-large xlm-roberta-base distilbert-base"
RUN_KEYS="${MODEL_KEY:-${ALL_KEYS}}"
for _k in ${RUN_KEYS}; do
    case "${_k}" in
        bert-large|xlm-roberta-base|distilbert-base) ;;
        *) echo "ERROR: Unknown MODEL_KEY '${_k}'. Valid: ${ALL_KEYS}"; exit 1 ;;
    esac
done

# Validate MODE
case "${MODE}" in
    fp32|bf16|smooth-static|smooth-static-autocast|smooth-dynamic|smooth-dynamic-autocast|all) ;;
    *) echo "ERROR: Unknown MODE '${MODE}'. Valid: fp32 bf16 smooth-static smooth-static-autocast smooth-dynamic smooth-dynamic-autocast all"; exit 1 ;;
esac

ALPHA_SWEEP="0.25 0.4 0.5 0.7 0.75 0.8 0.85 0.9"

COMPILE_FLAG=""
[[ "${AOTI}" == "1" ]] && COMPILE_FLAG="--aoti"

# ── SQuAD ─────────────────────────────────────────────────────────────────────
if [[ "${TASK}" == "squad" || "${TASK}" == "all" ]]; then
    echo ""
    echo "========================================"
    echo "  SQuAD 1.1 ACCURACY (F1 / EM)"
    echo "========================================"

    declare -A SQUAD_MODELS=(
        ["bert-large"]="google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
        ["xlm-roberta-base"]="deepset/xlm-roberta-base-squad2"
        ["distilbert-base"]="distilbert/distilbert-base-cased-distilled-squad"
    )

    for KEY in ${RUN_KEYS}; do
        MODEL="${SQUAD_MODELS[$KEY]}"
        echo ""
        echo ">> ${KEY}  (${MODEL})"
        taskset -c "${CORES}" python "${SCRIPT_DIR}/benchmark_accuracy.py" \
            --model       "${MODEL}" \
            --task        squad \
            --quant-mode  ${MODE} \
            --alpha       ${ALPHA_SWEEP} \
            --num-samples "${NUM_SAMPLES}" \
            --num-calib   "${NUM_CALIB}" \
            --output      "${OUTDIR}/squad_${KEY}.json" \
            ${COMPILE_FLAG} \
            || echo "WARNING: SQuAD benchmark failed for ${KEY}"
    done
fi

# ── MultiNLI ──────────────────────────────────────────────────────────────────
if [[ "${TASK}" == "mnli" || "${TASK}" == "all" ]]; then
    echo ""
    echo "========================================"
    echo "  MultiNLI ACCURACY"
    echo "========================================"

    declare -A MNLI_MODELS=(
        ["bert-large"]="yoshitomo-matsubara/bert-large-uncased-mnli"
        ["xlm-roberta-base"]="symanto/xlm-roberta-base-snli-mnli-anli-xnli"
        ["distilbert-base"]="typeform/distilbert-base-uncased-mnli"
    )

    for KEY in ${RUN_KEYS}; do
        MODEL="${MNLI_MODELS[$KEY]}"
        echo ""
        echo ">> ${KEY}  (${MODEL})"
        taskset -c "${CORES}" python "${SCRIPT_DIR}/benchmark_accuracy.py" \
            --model       "${MODEL}" \
            --task        mnli \
            --quant-mode  ${MODE} \
            --alpha       ${ALPHA_SWEEP} \
            --num-samples "${NUM_SAMPLES}" \
            --num-calib   "${NUM_CALIB}" \
            --output      "${OUTDIR}/mnli_${KEY}.json" \
            ${COMPILE_FLAG} \
            || echo "WARNING: MNLI benchmark failed for ${KEY}"
    done
fi

# ── Report ────────────────────────────────────────────────────────────────────
if [[ "${TASK}" == "report" || "${TASK}" == "all" ]]; then
    echo ""
    echo "========================================"
    echo "  RESULTS REPORT"
    echo "========================================"

    REPORT_FLAGS="--results-dir ${OUTDIR} --report-format ${REPORT_FORMAT}"
    [[ -n "${REPORT_OUTPUT}" ]] && REPORT_FLAGS+=" --report-output ${REPORT_OUTPUT}"

    python "${SCRIPT_DIR}/benchmark_accuracy.py" --task report ${REPORT_FLAGS}
fi

echo ""
echo "Done. Results in ${OUTDIR}/"
