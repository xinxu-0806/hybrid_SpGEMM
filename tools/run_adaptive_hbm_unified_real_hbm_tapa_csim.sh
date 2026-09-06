#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tapa_root="$repo_root/tools/tapa_local/usr"
build_dir="$repo_root/build/adaptive_hbm_unified_real_hbm_tapa_csim"

export XILINX_VITIS=/data/home/xuxin/Xilinx/Vitis/2022.2
export XILINX_VIVADO=/data/home/xuxin/Xilinx/Vivado/2022.2
export XILINX_HLS=/data/home/xuxin/Xilinx/Vitis_HLS/2022.2
export PATH="$tapa_root/bin:$XILINX_HLS/bin:$XILINX_VITIS/bin:$XILINX_VIVADO/bin:$PATH"
export LD_LIBRARY_PATH="$tapa_root/lib:${LD_LIBRARY_PATH:-}"
export TAPA_CONCURRENCY="${TAPA_CONCURRENCY:-4}"

mkdir -p "$build_dir"
tapa g++ -- \
  "$repo_root/src/adaptive_hbm_tapa/experimental/unified_real_hbm/adaptive_hbm_unified_real_hbm_tapa.cpp" \
  "$repo_root/src/adaptive_hbm_tapa/experimental/unified_real_hbm/adaptive_hbm_unified_real_hbm_tapa_tb.cpp" \
  -O2 -fsplit-stack -o "$build_dir/test"
"$build_dir/test" 2>&1 | tee "$build_dir/run.log"
