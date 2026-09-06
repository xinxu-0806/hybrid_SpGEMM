#!/usr/bin/env bash
# U280 运行时环境（XRT 2022.2 / 2.14.354，与 2026-07-20 板测记录同版本）
#
# 多数情况下不需要手动 source：
#   - XILINX_XRT 与 PATH 已由 /etc/profile.d/xrt-u280.sh 全局设置
#   - OpenCL 程序（TCAD_SCALER 的 lu_solver_host_*）靠 /etc/OpenCL/vendors/xilinx.icd
#     自动发现，libxilinxopencl 自带 RUNPATH，无需 LD_LIBRARY_PATH
#
# 本脚本仅为直接链接 libxrt_coreutil 的程序补 LD_LIBRARY_PATH
# （如 SpGEMM_hbm/u280/adaptive_hbm/app_adaptive_hbm_spgemm.exe）
export XILINX_XRT=${XILINX_XRT:-/data/home/xuxin/xrt_pkg/t22/opt/xilinx/xrt}
export LD_LIBRARY_PATH="$XILINX_XRT/lib:${LD_LIBRARY_PATH:-}"
export PATH="$XILINX_XRT/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}"
echo "XILINX_XRT = $XILINX_XRT"
