#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND="${1:-eager_cpu}"
case "$BACKEND" in eager_cpu|reactant_cpu|metal_gpu) ;; *) exit 2 ;; esac
JULIA_BIN="${JOPTUNALEARNERS_JULIA_BIN:-julia}"
RSS_LIMIT_MIB="${JOPTUNALEARNERS_BENCHMARK_RSS_LIMIT_MIB:-6144}"
[[ "$RSS_LIMIT_MIB" =~ ^[0-9]+$ ]] && (( RSS_LIMIT_MIB >= 2 )) || {
  echo "RSS limit must be an integer of at least 2 MiB" >&2; exit 2;
}
# RSS includes host compiler/native allocations, but not all Metal driver memory.
# Give the managed heap a smaller collection target; report GPU allocation separately.
HEAP_SIZE_HINT="${JOPTUNALEARNERS_BENCHMARK_HEAP_SIZE_HINT:-$((RSS_LIMIT_MIB / 2))M}"
PROJECT="${JOPTUNALEARNERS_BENCHMARK_PROJECT:-test/accelerators}"
TIMEOUT_SECONDS="${JOPTUNALEARNERS_BENCHMARK_TIMEOUT_SECONDS:-3600}"
[[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] && (( TIMEOUT_SECONDS >= 1 )) || exit 2
RESULTS="${JOPTUNALEARNERS_BENCHMARK_RESULTS:-$ROOT/benchmark/execution_backends/results}"
SCRIPT="${JOPTUNALEARNERS_BENCHMARK_SCRIPT:-benchmark/execution_backends/run.jl}"
MODEL="${JOPTUNALEARNERS_BENCHMARK_MODEL:-window_dlinear}"
mkdir -p "$RESULTS"
LOG="$RESULTS/$BACKEND.log"

rss_group_kib() {
  ps -axo pgid=,rss= | awk -v group="$1" '$1 == group { total += $2 } END { print total+0 }'
}

cd "$ROOT"
started="$(date +%s)"; peak=0; status=0
# Bash job control gives the single background worker its own process group on
# macOS as well as Linux. Descendants remain covered even if their parent exits.
set -m
JOPTUNALEARNERS_BENCHMARK_BACKEND="$BACKEND" JOPTUNALEARNERS_BENCHMARK_MODEL="$MODEL" \
JOPTUNALEARNERS_BENCHMARK_OUTPUT="$RESULTS/$BACKEND.json" JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}" \
  "$JULIA_BIN" --project="$PROJECT" --heap-size-hint="$HEAP_SIZE_HINT" --startup-file=no \
  "$SCRIPT" >>"$LOG" 2>&1 &
worker=$!
cleanup() {
  if kill -0 -- "-$worker" 2>/dev/null; then
    kill -TERM -- "-$worker" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-$worker" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
while kill -0 "$worker" 2>/dev/null; do
  rss=$(( $(rss_group_kib "$worker") / 1024 )); (( rss > peak )) && peak="$rss"
  elapsed=$(( $(date +%s) - started ))
  if (( rss > RSS_LIMIT_MIB )); then status=137; break; fi
  if (( elapsed >= TIMEOUT_SECONDS )); then status=124; break; fi
  sleep 1
done
cleanup
set +e; wait "$worker"; worker_status=$?; set -e
(( status == 0 )) && status="$worker_status"
trap - INT TERM EXIT
elapsed=$(( $(date +%s) - started ))
printf 'backend\telapsed_seconds\tpeak_rss_mib\tstatus\n%s\t%s\t%s\t%s\n' \
  "$BACKEND" "$elapsed" "$peak" "$status" >"$RESULTS/$BACKEND-supervisor.tsv"
echo "backend=$BACKEND status=$status elapsed_seconds=$elapsed peak_rss_mib=$peak"
exit "$status"
