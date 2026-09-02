#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND="${1:-}"
case "$BACKEND" in reactant_cpu|metal_gpu) ;; *) echo "Backend must be reactant_cpu or metal_gpu." >&2; exit 2 ;; esac
JULIA_BIN="${JOPTUNALEARNERS_JULIA_BIN:-julia}"
RSS_LIMIT_MIB="${JOPTUNALEARNERS_QUALIFICATION_RSS_LIMIT_MIB:-8192}"
TIMEOUT_SECONDS="${JOPTUNALEARNERS_QUALIFICATION_TIMEOUT_SECONDS:-7200}"
LOG="${JOPTUNALEARNERS_QUALIFICATION_LOG:-$ROOT/test/accelerators/$BACKEND.log}"

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
env_args=(JOPTUNALEARNERS_QUALIFY_MODEL_ZOO=true JOPTUNALEARNERS_QUALIFY_LIFECYCLE=true
          JOPTUNALEARNERS_QUALIFY_MEMORY=true JULIA_NUM_THREADS=1)
if [[ "$BACKEND" == "reactant_cpu" ]]; then
  env_args+=(JOPTUNALEARNERS_TEST_REACTANT=true)
else
  env_args+=(JOPTUNALEARNERS_TEST_METAL=true)
fi
env "${env_args[@]}" "$JULIA_BIN" --project=test/accelerators --startup-file=no \
  test/accelerators/runtests.jl >>"$LOG" 2>&1 &
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
  "$BACKEND" "$elapsed" "$peak" "$status" >"$ROOT/test/accelerators/$BACKEND-supervisor.tsv"
echo "backend=$BACKEND status=$status elapsed_seconds=$elapsed peak_rss_mib=$peak"
exit "$status"
