#!/bin/bash
# Run the workflow in the background with GPU support.
# Usage:
#   ./run_workflow_background.sh --root-folder-id <ID>                    # steps 1,2,3 on all images
#   ./run_workflow_background.sh --root-folder-id <ID> --steps 1,2,3,4   # full pipeline incl. upload
#   ./run_workflow_background.sh --root-folder-id <ID> --steps 2,3       # skip step 1
#   ./run_workflow_background.sh --steps 3                                # PPC only
#   ./run_workflow_background.sh --steps 4                                # upload PPC overlay (optional)
#   ./run_workflow_background.sh --root-folder-id <ID> --force            # force reprocess all

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/../.venv/bin/python"
LOG_DIR="${SCRIPT_DIR}/output/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/workflow_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/workflow.pid"

# Expose user site-packages (torch/transformers) to the venv python
USER_SITE="${HOME}/.local/lib/python3.12/site-packages"
export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"

# Default args
STEPS="1,2,3"
ROOT_FOLDER_ID=""
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)          STEPS="$2"; shift 2 ;;
        --root-folder-id) ROOT_FOLDER_ID="$2"; shift 2 ;;
        *)                EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "${ROOT_FOLDER_ID}" ]] && echo "${STEPS}" | grep -q "1"; then
    echo "ERROR: --root-folder-id is required when running Step 1."
    echo "Usage: ./run_workflow_background.sh --root-folder-id <DSA_FOLDER_ID> [--steps 1,2,3]"
    exit 1
fi

FOLDER_ARG=()
[[ -n "${ROOT_FOLDER_ID}" ]] && FOLDER_ARG=("--root-folder-id" "${ROOT_FOLDER_ID}")

echo "=========================================="
echo "Starting workflow in background"
echo "Steps:  ${STEPS}"
[[ -n "${ROOT_FOLDER_ID}" ]] && echo "Folder: ${ROOT_FOLDER_ID}"
echo "Log:    ${LOG_FILE}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'check nvidia-smi')"
echo "=========================================="

nohup "${VENV_PYTHON}" -u "${SCRIPT_DIR}/run_workflow.py" \
    --config "${SCRIPT_DIR}/dsa_workflow_config.json" \
    --steps "${STEPS}" \
    "${FOLDER_ARG[@]}" \
    "${EXTRA_ARGS[@]}" \
    > "${LOG_FILE}" 2>&1 &

JOB_PID=$!
echo "${JOB_PID}" > "${PID_FILE}"

echo "Started with PID: ${JOB_PID}"
echo "PID saved to:     ${PID_FILE}"
echo ""
echo "Monitor with:"
echo "  ./monitor_progress.sh"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "Stop with:"
echo "  ./stop.sh"

