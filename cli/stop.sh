#!/bin/bash
# Stop the background workflow process.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/output/logs/workflow.pid"

if [[ -f "${PID_FILE}" ]]; then
    PID=$(cat "${PID_FILE}")
    if kill -0 "${PID}" 2>/dev/null; then
        echo "Stopping workflow (PID ${PID})..."
        kill "${PID}"
        sleep 2
        if kill -0 "${PID}" 2>/dev/null; then
            echo "Process still running, force killing..."
            kill -9 "${PID}"
        fi
        echo "Stopped."
    else
        echo "Process ${PID} is not running."
    fi
    rm -f "${PID_FILE}"
else
    # Fallback: search by process name
    PIDS=$(pgrep -f "run_workflow.py")
    if [[ -n "${PIDS}" ]]; then
        echo "Found workflow processes: ${PIDS}"
        kill ${PIDS}
        echo "Stopped."
    else
        echo "No workflow process found."
    fi
fi
