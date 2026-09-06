#!/usr/bin/env bash
set -euo pipefail

# Second floorplan candidate after the 3/2/3 placement showed SLR1 at 94.32%
# CLB and a 63.97% SLR0<->SLR1 SLL bridge.  Keep all eight HBM shards and the
# full heavy-row merge capacity, but move shard 5 from the central SLR1 to the
# underutilised SLR2.  The floorplan Tcl recognizes TAPA_SHARD_LAYOUT=3_1_4.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
xo="${XO:-$repo_root/build/adaptive_hbm_unified_real_hbm_tapa_compile_v11_buffer0_uram_140/adaptive_hbm_unified_real_hbm_tapa_u280.xo}"
config="${CONFIG:-$repo_root/src/u280_unified_real_hbm_140_plramstats.cfg}"
run_tag="${RUN_TAG:-unified_real_hbm_buffer0uram_140mhz_slr314}"
xclbin="${XCLBIN:-$repo_root/u280/adaptive_hbm/adaptive_hbm_unified_real_hbm_buffer0uram_140mhz_slr314.xclbin}"
temp_dir="${TAPA_LINK_TEMP_DIR:-/tmp/xuxin_spgemm_hbm_link_buffer0uram_140_slr314/_x_${run_tag}}"
floorplan="${FLOORPLAN:-$repo_root/tools/place_tapa_local_merge_balanced_slr_20260828.tcl}"

[[ -s "$xo" ]] || { echo "missing XO: $xo" >&2; exit 1; }
[[ -s "$config" ]] || { echo "missing config: $config" >&2; exit 1; }
[[ -s "$floorplan" ]] || { echo "missing floorplan Tcl: $floorplan" >&2; exit 1; }

TAPA_SHARD_LAYOUT=3_1_4 \
TAPA_LINK_XO="$xo" \
TAPA_LINK_XCLBIN="$xclbin" \
TAPA_LINK_RUN_TAG="$run_tag" \
TAPA_LINK_TEMP_DIR="$temp_dir" \
TAPA_LINK_CONFIG="$config" \
TAPA_IMPL_OPT_PRE_TCL="$floorplan" \
POST_ROUTE_PHYS_OPT=1 \
  bash "$repo_root/tools/link_adaptive_hbm_tapa_parallel_u280_160.sh"
