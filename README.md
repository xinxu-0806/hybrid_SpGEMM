# Hybrid SpGEMM: current unified-real-HBM checkpoint

This repository intentionally contains only the active U280/TAPA design, not
historical implementations, generated HLS/Vivado projects, matrices, or FPGA
bitstreams.

## Design boundary

The Host exposes three modes: `merge-only`, `dense-only`, and `adaptive`.
In adaptive mode it chooses MERGE or DENSE for each output row.  The FPGA
remains a single unified kernel: eight HBM B-shards feed the shared allocator,
the MERGE forward tree or DENSE dual-context accumulator, then the common
four-port CSR writer.  No external DMSA kernel is a selectable path.

The current v12 candidate retains all eight HBM shards and the full MERGE and
DENSE paths.  Its only pending physical experiment is shallower elastic FIFO
depths, intended to reduce CLB/SLL placement pressure without changing the
algorithm or Host-visible behavior.

## Layout

- `src/adaptive_hbm_tapa/experimental/unified_real_hbm/`: current top kernel
  and its two CSim tests.
- `src/adaptive_hbm_tapa/experimental/{merge_allocator8,merge_forward_tree8,
  merge15_packetmeta4,segmented8_shared,merge_keyed_carry8,
  unified_dense_dual_context,...}`: direct implementation dependencies of the
  unified top.
- `src/host_adaptive_hbm_unified_real.cpp` and included Host sources:
  selector, packing, XRT launch, validation, and kernel-time measurement.
- `src/u280_unified_real_hbm_140_plramstats.cfg`: U280 HBM port mapping and
  140 MHz link configuration.
- `tools/`: exactly the compile, CSim, Host-build, link, floorplan,
  post-place-diagnostics, and board-measurement scripts for this design.

## Reproduction order

1. Install TAPA, Vitis/Vivado 2022.2, XRT, and the U280 2022.1 platform.
   Point `tools/tapa_local` in the scripts to a local TAPA installation; it is
   deliberately not versioned.
2. Run `tools/run_adaptive_hbm_unified_real_hbm_tapa_csim.sh`.
3. Build the XO with `tools/run_adaptive_hbm_unified_real_hbm_tapa_compile.sh`.
4. Link with `tools/link_adaptive_hbm_unified_real_hbm_140_slr314.sh` and use
   its post-place reports as a congestion gate before routing.
5. Build the Host using `tools/build_adaptive_hbm_unified_real_host.sh`, then
   run the frozen paper-10 board script after a timing-clean xclbin exists.

Generated outputs are excluded by `.gitignore`.
