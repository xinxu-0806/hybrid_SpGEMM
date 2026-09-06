#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
xrt_root="${XILINX_XRT:-/opt/xilinx/xrt}"
output="${ADAPT_HOST_OUTPUT:-$repo_root/build/app_adaptive_hbm_unified_real.exe}"

mkdir -p "$(dirname "$output")"
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$repo_root/src" -I"$xrt_root/include" \
  "$repo_root/src/host_adaptive_hbm_unified_real.cpp" \
  -L"$xrt_root/lib" -Wl,-rpath,"$xrt_root/lib" \
  -lxrt_coreutil -pthread -o "$output"

sha256sum "$output"
