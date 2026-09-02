#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND="${1:-eager_cpu}"
case "$BACKEND" in eager_cpu|reactant_cpu|metal_gpu) ;; *) exit 2 ;; esac
JULIA_BIN="${JOPTUNALEARNERS_JULIA_BIN:-julia}"
RSS_LIMIT_MIB="${JOPTUNALEARNERS_BENCHMARK_RSS_LIMIT_MIB:-6144}"
TIMEOUT_SECONDS="${JOPTUNALEARNERS_BENCHMARK_TIMEOUT_SECONDS:-3600}"
RESULTS="${JOPTUNALEARNERS_BENCHMARK_RESULTS:-$ROOT/benchmark/execution_backends/results}"
SCRIPT="${JOPTUNALEARNERS_BENCHMARK_SCRIPT:-benchmark/execution_backends/run.jl}"
MODEL="${JOPTUNALEARNERS_BENCHMARK_MODEL:-window_dlinear}"
mkdir -p "$RESULTS"
LOG="$RESULTS/$BACKEND.log"

children_of() { pgrep -P "$1" 2>/dev/null || true; }
rss_tree_kib() {
  local pid="$1" total=0 own child
  own="$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  [[ -n "$own" ]] && total=$((total + own))
  for child in $(children_of "$pid"); do total=$((total + $(rss_tree_kib "$child"))); done
  echo "$total"
}
stop_tree() {
  local pid="$1" child
  for child in $(children_of "$pid"); do stop_tree "$child"; done
  kill -TERM "$pid" 2>/dev/null || true
}

cd "$ROOT"
started="$(date +%s)"; peak=0; status=0
JOPTUNALEARNERS_BENCHMARK_BACKEND="$BACKEND" JOPTUNALEARNERS_BENCHMARK_MODEL="$MODEL" \
JOPTUNALEARNERS_BENCHMARK_OUTPUT="$RESULTS/$BACKEND.json" JULIA_NUM_THREADS=1 \
  "$JULIA_BIN" --project=test/accelerators --startup-file=no \
  "$SCRIPT" >>"$LOG" 2>&1 &
worker=$!
cleanup() { kill -0 "$worker" 2>/dev/null && stop_tree "$worker" || true; }
trap cleanup INT TERM EXIT
while kill -0 "$worker" 2>/dev/null; do
  rss=$(( $(rss_tree_kib "$worker") / 1024 )); (( rss > peak )) && peak="$rss"
  elapsed=$(( $(date +%s) - started ))
  if (( rss > RSS_LIMIT_MIB )); then stop_tree "$worker"; status=137; break; fi
  if (( elapsed > TIMEOUT_SECONDS )); then stop_tree "$worker"; status=124; break; fi
  sleep 5
done
set +e; wait "$worker"; worker_status=$?; set -e
(( status == 0 )) && status="$worker_status"
trap - INT TERM EXIT
elapsed=$(( $(date +%s) - started ))
printf 'backend\telapsed_seconds\tpeak_rss_mib\tstatus\n%s\t%s\t%s\t%s\n' \
  "$BACKEND" "$elapsed" "$peak" "$status" >"$RESULTS/$BACKEND-supervisor.tsv"
echo "backend=$BACKEND status=$status elapsed_seconds=$elapsed peak_rss_mib=$peak"
exit "$status"
