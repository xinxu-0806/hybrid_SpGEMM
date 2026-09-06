#ifndef ADAPTIVE_HBM_UNIFIED_ALLOCATOR_MERGE_OUTPUT_TAPA_H
#define ADAPTIVE_HBM_UNIFIED_ALLOCATOR_MERGE_OUTPUT_TAPA_H

#include <ap_int.h>
#include <tapa.h>

#include <cstdint>

using id_t = std::uint32_t;

// Command layout:
//   row[31:0], command descriptor index[63:32], source mask[71:64].
// Each source mmap is a sequence of active-row runs.  A run starts with a
// 64-bit count word and is followed by that many {FP32[63:32], local col[15:0]}
// items.  global_bases[descriptor index] restores the full 32-bit column.
void adaptive_hbm_unified_allocator_merge_output_tapa(
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
		tapa::mmap<id_t> allocator_stats);

#endif
