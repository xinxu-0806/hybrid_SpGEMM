#ifndef ADAPTIVE_HBM_UNIFIED_ADAPTIVE_FULL_OUTPUT_TAPA_H
#define ADAPTIVE_HBM_UNIFIED_ADAPTIVE_FULL_OUTPUT_TAPA_H

#include <ap_int.h>
#include <tapa.h>

#include <cstdint>

using id_t = std::uint32_t;

#define UNIFIED_FULL_ROUTE_MERGE 0
#define UNIFIED_FULL_ROUTE_DENSE 1
#define UNIFIED_FULL_OUTPUT_MERGE_ROW 1
#define UNIFIED_FULL_OUTPUT_DENSE 3

// Command layout:
//   row[31:0], descriptor/global-base index[63:32], source mask[71:64],
//   Host-selected route[73:72], row partial-product count[105:74].
//
// MERGE and DENSE are the only algorithm routes.  Tree level/exit and the two
// physical DENSE slots are private implementation resources allocated inside
// this one kernel invocation.  For a MERGE command, hardware compares the
// supplied row count with heavy_merge_product_threshold and internally selects
// either the flexible multi-row tree or the 8-wide heavy-row microengine; this
// does not create a third Host-visible algorithm route.
void adaptive_hbm_unified_adaptive_full_output_tapa(
		tapa::mmap<const ap_uint<128> > commands, id_t command_count,
		tapa::mmap<const id_t> global_bases,
		tapa::mmap<const ap_uint<64> > source0,
		tapa::mmap<const ap_uint<64> > source1,
		tapa::mmap<const ap_uint<64> > source2,
		tapa::mmap<const ap_uint<64> > source3,
		tapa::mmap<const ap_uint<64> > source4,
		tapa::mmap<const ap_uint<64> > source5,
		tapa::mmap<const ap_uint<64> > source6,
		tapa::mmap<const ap_uint<64> > source7,
		id_t dense_local_column_count, id_t heavy_merge_product_threshold,
		id_t output_data_word_capacity_per_port,
		id_t output_meta_word_capacity,
		id_t output_completion_capacity,
		tapa::mmap<ap_uint<512> > output_item0,
		tapa::mmap<ap_uint<512> > output_item1,
		tapa::mmap<ap_uint<512> > output_item2,
		tapa::mmap<ap_uint<512> > output_item3,
		tapa::mmap<ap_uint<512> > output_meta,
		tapa::mmap<ap_uint<64> > output_completion,
		tapa::mmap<id_t> output_port_stats0,
		tapa::mmap<id_t> output_port_stats1,
		tapa::mmap<id_t> output_port_stats2,
		tapa::mmap<id_t> output_port_stats3,
		tapa::mmap<id_t> output_meta_stats,
		tapa::mmap<id_t> output_completion_stats,
		tapa::mmap<id_t> scheduler_stats,
		tapa::mmap<id_t> dense_stats,
		tapa::mmap<id_t> heavy_merge_stats);

#endif
