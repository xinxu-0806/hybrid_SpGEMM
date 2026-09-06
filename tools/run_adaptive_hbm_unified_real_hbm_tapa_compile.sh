#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tapa_root="$repo_root/tools/tapa_local/usr"
platform="$repo_root/.local/u280-platform-2022.1/opt/xilinx/platforms/xilinx_u280_gen3x16_xdma_1_202211_1"
build_dir="${TAPA_BUILD_DIR:-$repo_root/build/adaptive_hbm_unified_real_hbm_tapa_compile}"
jobs="${TAPA_JOBS:-4}"
clock_period="${TAPA_CLOCK_PERIOD:-6.25}"

export XILINX_VITIS=/data/home/xuxin/Xilinx/Vitis/2022.2
export XILINX_VIVADO=/data/home/xuxin/Xilinx/Vivado/2022.2
export XILINX_HLS=/data/home/xuxin/Xilinx/Vitis_HLS/2022.2
export PATH="$tapa_root/bin:$XILINX_HLS/bin:$XILINX_VITIS/bin:$XILINX_VIVADO/bin:$PATH"
export LD_LIBRARY_PATH="$tapa_root/lib:${LD_LIBRARY_PATH:-}"
export XILINXD_LICENSE_FILE=/data/home/xuxin/Xilinx/license/2022.2/License.lic:/data/home/xuxin/Xilinx/license/2022.2/vivado_lic2037.lic

mkdir -p "$build_dir"
cd "$build_dir"
tapa compile \
  --top adaptive_hbm_unified_real_hbm_tapa \
  --platform "$platform" \
  --clock-period "$clock_period" \
  --jobs "$jobs" \
  --keep-hls-work-dir \
  -f "$repo_root/src/adaptive_hbm_tapa/experimental/unified_real_hbm/adaptive_hbm_unified_real_hbm_tapa.cpp" \
  -o adaptive_hbm_unified_real_hbm_tapa_u280.xo \
  2>&1 | tee tapa_compile.log
