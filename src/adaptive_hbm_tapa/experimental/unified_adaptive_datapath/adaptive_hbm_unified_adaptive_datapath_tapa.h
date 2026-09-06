#ifndef ADAPTIVE_HBM_UNIFIED_ADAPTIVE_DATAPATH_TAPA_H
#define ADAPTIVE_HBM_UNIFIED_ADAPTIVE_DATAPATH_TAPA_H

#include <ap_int.h>
#include <tapa.h>

#include <cstdint>

using id_t = std::uint32_t;

#define UNIFIED_ADAPTIVE_ROUTE_MERGE 0
#define UNIFIED_ADAPTIVE_ROUTE_DENSE 1

// Input metadata:
// row[31:0], mask[39:32], route[41:40], EOR[42], allocator context[45:43],
// generation[53:46], physical DENSE slot[54].
void adaptive_hbm_unified_adaptive_datapath_tapa(
		tapa::mmap<const ap_uint<512> > data_in,
		tapa::mmap<const ap_uint<64> > meta_in,
		id_t packet_count,
		id_t column_count,
		tapa::mmap<ap_uint<512> > merge_data_out,
		tapa::mmap<ap_uint<64> > merge_meta_out,
		tapa::mmap<ap_uint<64> > merge_completion_out,
		tapa::mmap<id_t> merge_stats_out,
		tapa::mmap<ap_uint<512> > dense_data_out0,
		tapa::mmap<ap_uint<64> > dense_meta_out0,
		tapa::mmap<ap_uint<64> > dense_completion_out0,
		tapa::mmap<id_t> dense_stats_out0,
		tapa::mmap<ap_uint<512> > dense_data_out1,
		tapa::mmap<ap_uint<64> > dense_meta_out1,
		tapa::mmap<ap_uint<64> > dense_completion_out1,
		tapa::mmap<id_t> dense_stats_out1);

#endif
