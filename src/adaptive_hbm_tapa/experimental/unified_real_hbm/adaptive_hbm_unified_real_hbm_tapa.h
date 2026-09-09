#ifndef ADAPTIVE_HBM_UNIFIED_REAL_HBM_TAPA_H
#define ADAPTIVE_HBM_UNIFIED_REAL_HBM_TAPA_H

#include <ap_int.h>
#include <tapa.h>

#include <cstdint>

using id_t = std::uint32_t;

// Production integration checkpoint.  Unlike the accepted synthetic-source
// gate, this top consumes eight packed B shards whose tails hold independent
// shard-local task lists.
// Each command is one <=64K column fragment.  row_logical[fragment] preserves
// the complete matrix row identity while row_source_mask/row_geometry and
// route/profile arrays are indexed by the physical fragment.  A Host selector
// runs once on the complete row and copies that route to every fragment.  The FPGA
// consumes all fragments in one invocation and the tagged four-port output
// lets the Host reunite disjoint fragments without numerical re-reduction.
void adaptive_hbm_unified_real_hbm_tapa(
		tapa::mmap<const ap_uint<32> > route,
		tapa::mmap<const ap_uint<32> > row_source_mask,
		tapa::mmap<const ap_uint<64> > row_geometry,
		tapa::mmap<const id_t> row_logical,
		tapa::mmap<const ap_uint<512> > B0,
		tapa::mmap<const ap_uint<512> > B1,
		tapa::mmap<const ap_uint<512> > B2,
		tapa::mmap<const ap_uint<512> > B3,
		tapa::mmap<const ap_uint<512> > B4,
		tapa::mmap<const ap_uint<512> > B5,
		tapa::mmap<const ap_uint<512> > B6,
		tapa::mmap<const ap_uint<512> > B7,
		id_t B0_beats, id_t B1_beats, id_t B2_beats, id_t B3_beats,
		id_t B4_beats, id_t B5_beats, id_t B6_beats, id_t B7_beats,
		id_t fragment_count, id_t heavy_merge_product_threshold,
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
