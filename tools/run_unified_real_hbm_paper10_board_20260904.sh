#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
xclbin="${XCLBIN:-$repo_root/u280/adaptive_hbm/adaptive_hbm_unified_real_hbm_burstflush_160mhz.xclbin}"
xo="${XO:-$repo_root/build/adaptive_hbm_unified_real_hbm_tapa_compile_v4_burstflush_160_accept/adaptive_hbm_unified_real_hbm_tapa_u280.xo}"
host="${HOST:-$repo_root/build/app_adaptive_hbm_unified_real.exe}"
link_root="${ADAPT_LINK_WORK_ROOT:-/tmp/xuxin_spgemm_hbm_link_burstflush_160}"
link_log="${LINK_LOG:-$repo_root/build/adaptive_hbm_unified_real_hbm_tapa_compile_v4_burstflush_160_accept/vpp_link_160.log}"
# Keep the current 160 MHz report as the default, while allowing a rebuilt
# frequency target to pass its own routed report to the same strict gate.
timing="${TIMING:-$link_root/_x_unified_real_hbm_wide64k_rtlportfix_160mhz/link/vivado/vpl/prj/prj.runs/impl_1/hw_bb_locked_timing_summary_postroute_physopted.rpt}"
result_dir="${RESULT_DIR:-$repo_root/u280/adaptive_hbm/unified_real_hbm_burstflush_160mhz_board_20260905}"
bdf="${U280_BDF:-0000:17:00.1}"
reps="${REPS:-10}"

paper_matrices=(
  poisson3Da raefsky1 crystk01 s3rmt3m3 t2dah_a
  nasa2910 bcsstk24 cavity26 ex9 af23560
)

[[ -x "$host" ]] || { echo "missing production Host: $host" >&2; exit 1; }
[[ -s "$xo" ]] || { echo "missing production XO: $xo" >&2; exit 1; }
[[ -s "$xclbin" ]] || { echo "missing production XCLBIN: $xclbin" >&2; exit 1; }
[[ -s "$link_log" ]] || { echo "missing v++ link log: $link_log" >&2; exit 1; }
[[ -s "$timing" ]] || { echo "missing routed timing report: $timing" >&2; exit 1; }
[[ "$reps" =~ ^[1-9][0-9]*$ ]] || { echo "REPS must be positive" >&2; exit 1; }
[[ ! -e "$result_dir" ]] || { echo "refusing to overwrite $result_dir" >&2; exit 1; }
for matrix in "${paper_matrices[@]}"; do
  [[ -s "$repo_root/matrices/$matrix.mtx" ]] || {
    echo "missing frozen paper matrix: $matrix" >&2; exit 1;
  }
done

xclbinutil --quiet --input "$xclbin" --info >/dev/null
rg -q 'Check VPL, containing [0-9]+ checks, has run: 0 errors' "$link_log"
rg -q 'Run vpl: Step impl: Completed' "$link_log"
python3 "$repo_root/tools/check_routed_timing.py" "$timing" \
  --clock clk_out1_ulp_clk_wiz_0 --min-wns 0 --min-whs 0
python3 "$repo_root/tools/check_routed_timing.py" "$timing" \
  --clock hbm_aclk --min-wns 0 --min-whs 0 \
  --allow-auto-scale-xclbin "$xclbin"
mkdir -p "$result_dir"
sha256sum "$xclbin" "$xo" "$host" "$timing" "$link_log" \
  "$repo_root/src/host_adaptive_hbm_unified_real.cpp" \
  "$repo_root/src/host_adaptive_hbm_spgemm_dim_commanded.cpp" \
  "$repo_root/src/adaptive_light_selector.h" \
  "$repo_root/src/rowpersistent_selector_policy_robust_v3_generated.h" \
  "$repo_root/src/adaptive_hbm_tapa/adaptive_hbm_tapa_scalable_windowed.cpp" \
  "$repo_root/src/adaptive_hbm_tapa/adaptive_hbm_tapa_scalable.h" \
  "$repo_root/tools/run_unified_real_hbm_paper10_board_20260904.sh" \
  > "$result_dir/ARTIFACT_SHA256.txt"
printf 'reps=%s\nbdf=%s\nwarmup_per_mode=1\n' "$reps" "$bdf" \
  >> "$result_dir/ARTIFACT_SHA256.txt"

source "$repo_root/setup_xrt_u280.sh"

# A U280 HOT reset can return before the PCIe user function is usable by XRT.
# Waiting a fixed two seconds made the first post-reset retry fail with
# "No such device or address" on this host.  Poll the exact BDF instead and
# give the driver a short settling interval after it becomes visible.
wait_for_u280_ready() {
  local attempts="${U280_READY_ATTEMPTS:-60}"
  local interval="${U280_READY_INTERVAL:-2}"
  local attempt
  for ((attempt = 1; attempt <= attempts; ++attempt)); do
    if xbutil examine -d "$bdf" >/dev/null 2>&1; then
      sleep 2
      echo "U280_READY bdf=$bdf attempt=$attempt"
      return 0
    fi
    sleep "$interval"
  done
  echo "U280 did not become ready after $attempts attempts: $bdf" >&2
  return 1
}

wait_for_u280_ready
xbutil examine | tee "$result_dir/xbutil_examine_before.log"

run_case() {
  local name="$1" matrix="$2" count="$3" timeout_value="$4" mode="${5:-all}"
  ADAPT_SCALABLE_MODE="$mode" timeout "$timeout_value" stdbuf -oL -eL \
    "$host" "$xclbin" auto "$matrix" "$result_dir/${name}_r${count}.csv" \
    "$count" 2>&1 | tee "$result_dir/${name}_r${count}.log"
  rg -q 'UNIFIED_REAL_HOST_BOARD: PASS' "$result_dir/${name}_r${count}.log"
}

# First prove the exact three-row DENSE slot-reuse case that deadlocked before
# the nonblocking stream-gate repair.  Try the card directly and reset only if
# this first programmed-image launch fails.
set +e
run_case gate_dense3 "$repo_root/build/unified_real_three_dense_smoke.mtx" 1 10m dense
smoke_status=$?
set -e
if [[ "$smoke_status" -ne 0 ]]; then
  mv "$result_dir/gate_dense3_r1.log" "$result_dir/gate_dense3_r1_before_reset.log" 2>/dev/null || true
  mv "$result_dir/gate_dense3_r1.csv" "$result_dir/gate_dense3_r1_before_reset.csv" 2>/dev/null || true
  xbutil --force --batch reset -d "$bdf" --type user </dev/null
  wait_for_u280_ready
  run_case gate_dense3 "$repo_root/build/unified_real_three_dense_smoke.mtx" 1 10m dense
fi

# Four rows force both physical DENSE slots to be reused more than once.
run_case gate_dense4 "$repo_root/build/unified_real_minimal_smoke.mtx" 1 10m dense

# Exercise both Host-visible paths and their shared completion/output network.
run_case gate_ex9 "$repo_root/matrices/ex9.mtx" 1 30m

# Proves that a column above 65535 is handled by the in-kernel fragment list.
run_case gate_70k "$repo_root/build/scalable_window_70k_smoke.mtx" 1 30m

for matrix in "${paper_matrices[@]}"; do
  echo "[$(date -u +'%F %T UTC')] unified-real paper10 matrix=$matrix reps=$reps"
  run_case "$matrix" "$repo_root/matrices/$matrix.mtx" "$reps" 4h
done

python3 "$repo_root/tools/summarize_unified_real_paper10.py" \
  "$result_dir" --reps "$reps" | tee "$result_dir/summary.log"
echo "UNIFIED_REAL_HBM_PAPER10_BOARD: PASS result_dir=$result_dir"
