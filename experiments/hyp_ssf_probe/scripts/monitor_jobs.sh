#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 LOG_FILE JOB_ID [JOB_ID ...]" >&2
    exit 1
fi

LOG_FILE=$1
shift
JOB_IDS=("$@")

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
LOG_DIR="$REPO_ROOT/experiments/hyp_ssf_probe/logs"
mkdir -p "$(dirname "$LOG_FILE")" "$LOG_DIR"

declare -A PREV_STATE
TERMINAL_STATES="COMPLETED FAILED CANCELLED TIMEOUT OUT_OF_MEMORY NODE_FAIL PREEMPTED BOOT_FAIL DEADLINE"

is_terminal() {
    local state=$1
    for terminal in $TERMINAL_STATES; do
        if [ "$state" = "$terminal" ]; then
            return 0
        fi
    done
    return 1
}

get_squeue_field() {
    local job_id=$1
    local fmt=$2
    squeue -h -j "$job_id" -o "$fmt" 2>/dev/null | head -n 1 | xargs
}

get_sacct_field() {
    local job_id=$1
    local field=$2
    sacct -j "$job_id" --format="$field" -n -P 2>/dev/null | head -n 1 | cut -d'|' -f1 | xargs
}

get_state() {
    local job_id=$1
    local state
    state=$(get_squeue_field "$job_id" "%T")
    if [ -z "$state" ]; then
        state=$(get_sacct_field "$job_id" "State")
    fi
    if [ -z "$state" ]; then
        state="UNKNOWN"
    fi
    printf '%s\n' "$state"
}

get_reason() {
    local job_id=$1
    local reason
    reason=$(get_squeue_field "$job_id" "%R")
    if [ -z "$reason" ]; then
        reason=$(get_sacct_field "$job_id" "Reason")
    fi
    printf '%s\n' "$reason"
}

append_failure_tail() {
    local ts=$1
    local job_id=$2
    local match
    match=$(ls "$LOG_DIR"/*_"$job_id".out 2>/dev/null | head -n 1 || true)
    if [ -z "$match" ]; then
        return 0
    fi
    {
        echo "[$ts] tail of $match"
        tail -n 40 "$match" || true
    } >> "$LOG_FILE"
}

{
    echo "[$(date '+%F %T')] monitoring jobs: ${JOB_IDS[*]}"
    echo "[$(date '+%F %T')] writing status changes to $LOG_FILE"
} >> "$LOG_FILE"

while true; do
    ts=$(date '+%F %T')
    active=0

    for job_id in "${JOB_IDS[@]}"; do
        state=$(get_state "$job_id")
        reason=$(get_reason "$job_id")
        prev=${PREV_STATE[$job_id]-}

        if [ "$state" != "$prev" ]; then
            if [ -n "$reason" ] && [ "$reason" != "None" ] && [ "$reason" != "None assigned" ]; then
                line="[$ts] $job_id $state reason=$reason"
            else
                line="[$ts] $job_id $state"
            fi
            echo "$line" | tee -a "$LOG_FILE"
            PREV_STATE[$job_id]=$state

            case "$state" in
                FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)
                    append_failure_tail "$ts" "$job_id"
                    ;;
            esac
        fi

        if ! is_terminal "$state"; then
            active=1
        fi
    done

    if [ "$active" -eq 0 ]; then
        echo "[$ts] all monitored jobs reached terminal states" | tee -a "$LOG_FILE"
        break
    fi

    sleep 120
done
