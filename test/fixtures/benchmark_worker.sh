#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >"$SUPERVISOR_TEST_ARGS"
printf '%s\n' "$$" >"$SUPERVISOR_TEST_PID"
if [[ "$SUPERVISOR_TEST_MODE" == stubborn ]]; then
    trap '' TERM
    bash -c 'trap "" TERM; while true; do sleep 1; done' &
    wait
fi
