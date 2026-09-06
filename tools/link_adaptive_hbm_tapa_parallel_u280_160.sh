#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_root/u280/adaptive_hbm"
compile_dir="${TAPA_LINK_COMPILE_DIR:-$repo_root/build/adaptive_hbm_tapa_parallel_compile}"
platform_default="$repo_root/.local/u280-platform-2022.1/opt/xilinx/platforms/xilinx_u280_gen3x16_xdma_1_202211_1/xilinx_u280_gen3x16_xdma_1_202211_1.xpfm"
platform="${PLATFORM:-$platform_default}"
xo="${TAPA_LINK_XO:-$compile_dir/adaptive_hbm_spgemm_tapa_parallel_u280.xo}"
xclbin="${TAPA_LINK_XCLBIN:-$build_dir/adaptive_hbm_spgemm_u280_tapa_parallel_bankdense_160mhz.xclbin}"
run_tag="${TAPA_LINK_RUN_TAG:-tapa_parallel_bankdense_160mhz}"
temp_dir="${TAPA_LINK_TEMP_DIR:-$build_dir/_x_${run_tag}}"
from_step="${1:-}"
post_route_phys_opt="${POST_ROUTE_PHYS_OPT:-0}"
impl_opt_pre_tcl="${TAPA_IMPL_OPT_PRE_TCL:-}"
impl_physopt_pre_tcl="${TAPA_IMPL_PHYSOPT_PRE_TCL:-$repo_root/tools/capture_adaptive_hbm_postplace_diagnostics_20260828.tcl}"
place_directive="${TAPA_PLACE_DIRECTIVE:-SSI_SpreadLogic_high}"
physopt_directive="${TAPA_PHYSOPT_DIRECTIVE:-AggressiveExplore}"
route_directive="${TAPA_ROUTE_DIRECTIVE:-Explore}"
link_config="${TAPA_LINK_CONFIG:-$repo_root/src/u280_adaptive_hbm_tapa_160.cfg}"

xilinx_root="${XILINX_ROOT:-/data/home/xuxin/Xilinx}"
export XILINX_VITIS="$xilinx_root/Vitis/2022.2"
export XILINX_VIVADO="$xilinx_root/Vivado/2022.2"
export XILINX_HLS="$xilinx_root/Vitis_HLS/2022.2"
export PATH="$XILINX_VITIS/bin:$XILINX_VIVADO/bin:$XILINX_HLS/bin:$PATH"
export XILINXD_LICENSE_FILE=/data/home/xuxin/Xilinx/license/2022.2/License.lic:/data/home/xuxin/Xilinx/license/2022.2/vivado_lic2037.lic

[[ -s "$xo" ]] || { echo "Missing TAPA XO: $xo" >&2; exit 1; }
[[ -f "$platform" ]] || { echo "Missing U280 platform: $platform" >&2; exit 1; }
[[ -s "$link_config" ]] || { echo "Missing Vitis link config: $link_config" >&2; exit 1; }
[[ ! -e "$xclbin" ]] || {
  echo "Refusing to overwrite existing TAPA XCLBIN: $xclbin" >&2
  exit 1
}

from_step_args=()
if [[ -n "$from_step" ]]; then
  from_step_args=(--from_step "$from_step")
  echo "Resuming Vitis link from step: $from_step"
fi

# Vivado 2022.2 can segfault while producing the timing report after the
# optional post-route phys_opt step, even when route_design has already met
# timing.  Keep that optional step disabled by default.  It can still be
# enabled explicitly for experiments with POST_ROUTE_PHYS_OPT=1.
post_route_args=(
  --vivado.prop run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED=false
)
if [[ "$post_route_phys_opt" == 1 ]]; then
  post_route_args=(
    --vivado.prop run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED=true
    --vivado.prop run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE=AggressiveExplore
  )
fi

impl_hook_args=()
if [[ -n "$impl_opt_pre_tcl" ]]; then
  [[ -s "$impl_opt_pre_tcl" ]] || {
    echo "Missing implementation opt-design pre-hook: $impl_opt_pre_tcl" >&2
    exit 1
  }
  impl_opt_pre_tcl="$(readlink -f "$impl_opt_pre_tcl")"
  impl_hook_args=(
    --vivado.prop "run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=$impl_opt_pre_tcl"
  )
  echo "Using implementation opt-design pre-hook: $impl_opt_pre_tcl"
fi

impl_physopt_hook_args=()
if [[ -n "$impl_physopt_pre_tcl" ]]; then
  [[ -s "$impl_physopt_pre_tcl" ]] || {
    echo "Missing implementation phys-opt pre-hook: $impl_physopt_pre_tcl" >&2
    exit 1
  }
  impl_physopt_pre_tcl="$(readlink -f "$impl_physopt_pre_tcl")"
  impl_physopt_hook_args=(
    --vivado.prop "run.impl_1.STEPS.PHYS_OPT_DESIGN.TCL.PRE=$impl_physopt_pre_tcl"
  )
  echo "Using implementation phys-opt pre-hook: $impl_physopt_pre_tcl"
fi

v++ -l -t hw \
  --platform "$platform" \
  --config "$link_config" \
  --vivado.synth.jobs 32 \
  --vivado.impl.jobs 32 \
  --vivado.prop "run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=$place_directive" \
  --vivado.prop run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=true \
  --vivado.prop "run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=$physopt_directive" \
  --vivado.prop "run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=$route_directive" \
  "${impl_hook_args[@]}" \
  "${impl_physopt_hook_args[@]}" \
  "${post_route_args[@]}" \
  "${from_step_args[@]}" \
  "$xo" \
  --temp_dir "$temp_dir" \
  --log_dir "$build_dir/logs_${run_tag}" \
  --report_dir "$build_dir/reports_${run_tag}" \
  -o "$xclbin"
