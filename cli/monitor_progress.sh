#!/bin/bash
# Monitor workflow progress and print DSA viewer links for completed images.
# Usage: ./monitor_progress.sh [--once]
#   --once  Print status once and exit (no live tail)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/output"
LOG_DIR="${OUT_DIR}/logs"
PID_FILE="${LOG_DIR}/workflow.pid"
STEP1_JSON="${OUT_DIR}/step2_images_output.json"
DSA_BASE="http://bdsa.pathology.emory.edu:8080"

ONCE=false
[[ "$1" == "--once" ]] && ONCE=true

# ---- helper: parse JSON with python (always available in this repo) ----
VENV_PYTHON="${SCRIPT_DIR}/../.venv/bin/python"
[[ ! -x "${VENV_PYTHON}" ]] && VENV_PYTHON="python3"

print_status() {
    echo ""
    echo "=========================================="
    echo " Workflow Progress  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    # --- process status ---
    if [[ -f "${PID_FILE}" ]]; then
        PID=$(cat "${PID_FILE}")
        if kill -0 "${PID}" 2>/dev/null; then
            echo "Status : RUNNING  (PID ${PID})"
            CPU=$(ps -p "${PID}" -o pcpu= 2>/dev/null | tr -d ' ')
            MEM=$(ps -p "${PID}" -o pmem= 2>/dev/null | tr -d ' ')
            ELAPSED=$(ps -p "${PID}" -o etime= 2>/dev/null | tr -d ' ')
            echo "         CPU ${CPU}%  MEM ${MEM}%  elapsed ${ELAPSED}"
        else
            echo "Status : FINISHED (PID ${PID} no longer running)"
        fi
    else
        RUNNING_PID=$(pgrep -f "run_workflow.py" | head -1)
        if [[ -n "${RUNNING_PID}" ]]; then
            echo "Status : RUNNING  (PID ${RUNNING_PID})"
        else
            echo "Status : not running"
        fi
    fi

    # --- latest log file ---
    LATEST_LOG=$(ls -t "${LOG_DIR}"/workflow_*.log 2>/dev/null | head -1)
    if [[ -n "${LATEST_LOG}" ]]; then
        echo "Log    : ${LATEST_LOG}"
        echo ""
        echo "--- Last 15 log lines ---"
        tail -15 "${LATEST_LOG}"
        echo "-------------------------"
    fi

    # --- per-image completion table ---
    if [[ ! -f "${STEP1_JSON}" ]]; then
        echo ""
        echo "(Step 1 output not found yet: ${STEP1_JSON})"
        return
    fi

    echo ""
    echo "Image completion:"
    printf "  %-35s %-6s %-6s  %s\n" "Image" "Seg" "PPC" "DSA Link"
    printf "  %-35s %-6s %-6s  %s\n" "-----" "---" "---" "--------"

    ITEMS=$("${VENV_PYTHON}" -c "
import json, sys
with open('${STEP1_JSON}') as f:
    data = json.load(f)
for item in data:
    name = item.get('name', 'unknown')
    stem = name.rsplit('.', 1)[0] if '.' in name else name
    item_id = item.get('_id', '')
    print(f'{stem}\t{name}\t{item_id}')
")

    TOTAL=0
    SEG_DONE=0
    PPC_DONE=0

    while IFS=$'\t' read -r STEM NAME ITEM_ID; do
        [[ -z "${STEM}" ]] && continue
        TOTAL=$((TOTAL + 1))

        SEG_FILE="${OUT_DIR}/${STEM}.anot"
        PPC_TIFF="${OUT_DIR}/${STEM}.tiff"
        PPC_ANOT="${OUT_DIR}/${STEM}-ppc.anot"

        SEG_STATUS="[ ]"
        PPC_STATUS="[ ]"

        if [[ -f "${SEG_FILE}" ]]; then
            SEG_STATUS="[x]"
            SEG_DONE=$((SEG_DONE + 1))
        fi
        if [[ -f "${PPC_TIFF}" ]] || [[ -f "${PPC_ANOT}" ]]; then
            PPC_STATUS="[x]"
            PPC_DONE=$((PPC_DONE + 1))
        fi

        if [[ -n "${ITEM_ID}" ]]; then
            LINK="${DSA_BASE}/#item/${ITEM_ID}"
        else
            LINK="(no item id)"
        fi

        printf "  %-35s %-6s %-6s  %s\n" "${STEM:0:35}" "${SEG_STATUS}" "${PPC_STATUS}" "${LINK}"
    done <<< "${ITEMS}"

    echo ""
    echo "Summary: Segmentation ${SEG_DONE}/${TOTAL}  |  PPC ${PPC_DONE}/${TOTAL}"
    echo ""
    echo "Folder: ${DSA_BASE}/#collection/67c5f81365fd0aa5859665b6/folder/67ddb0782fc8ce397c5ef7fb"
    echo "=========================================="
}

if $ONCE; then
    print_status
    exit 0
fi

# Live mode: print status every 15 seconds
echo "Monitoring workflow (Ctrl+C to stop)..."
while true; do
    print_status
    sleep 15
done
