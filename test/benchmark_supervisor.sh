#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/joptuna-supervisor-test.XXXXXX")"
# A tiny fake Julia worker exercises cleanup without running a training workload.
cp "$ROOT/test/fixtures/benchmark_worker.sh" "$SCRATCH/worker"
chmod +x "$SCRATCH/worker"
export JOPTUNALEARNERS_JULIA_BIN="$SCRATCH/worker"
export JOPTUNALEARNERS_BENCHMARK_RESULTS="$SCRATCH/results"
export JOPTUNALEARNERS_BENCHMARK_TIMEOUT_SECONDS=2
export SUPERVISOR_TEST_PID="$SCRATCH/pid"
export SUPERVISOR_TEST_ARGS="$SCRATCH/args"

SUPERVISOR_TEST_MODE=success bash "$ROOT/benchmark/execution_backends/supervise.sh" eager_cpu
grep -q -- '--heap-size-hint=3072M' "$SCRATCH/args"

set +e
SUPERVISOR_TEST_MODE=stubborn bash "$ROOT/benchmark/execution_backends/supervise.sh" eager_cpu
status=$?
set -e
[[ "$status" == 124 ]]
worker="$(<"$SCRATCH/pid")"
if kill -0 -- "-$worker" 2>/dev/null; then
    echo "timed-out worker group survived cleanup" >&2
    exit 1
fi
set +e
SUPERVISOR_TEST_MODE=stubborn JOPTUNALEARNERS_BENCHMARK_RSS_LIMIT_MIB=2 \
    JOPTUNALEARNERS_BENCHMARK_TIMEOUT_SECONDS=10 \
    bash "$ROOT/benchmark/execution_backends/supervise.sh" eager_cpu
status=$?
set -e
[[ "$status" == 137 ]]
worker="$(<"$SCRATCH/pid")"
! kill -0 -- "-$worker" 2>/dev/null
echo "Benchmark supervisor: heap hint, RSS, timeout, and TERM-resistant group cleanup passed"
