#ifndef ADAPTIVE_HBM_SEGMENTED8_SHARED_TAPA_H
#define ADAPTIVE_HBM_SEGMENTED8_SHARED_TAPA_H

#include <ap_int.h>
#include <tapa.h>

#include <cstdint>

using id_t = std::uint32_t;

#define SEGMENTED8_SHARED_MERGE 0
#define SEGMENTED8_SHARED_DENSE 1

// Packet token: items[511:0], row[543:512], mask[551:544], route[553:552],
// EOR[554], boundary duplicate[555], terminal[556].
using Segmented8SharedToken = ap_uint<557>;

void adaptive_hbm_segmented8_shared_tapa(
		tapa::mmap<const ap_uint<512> > data_in,
		tapa::mmap<const ap_uint<64> > meta_in,
		id_t packet_count,
		tapa::mmap<ap_uint<512> > merge_data_out,
		tapa::mmap<ap_uint<64> > merge_meta_out,
		tapa::mmap<id_t> merge_stats_out,
		tapa::mmap<ap_uint<512> > dense_data_out,
		tapa::mmap<ap_uint<64> > dense_meta_out,
		tapa::mmap<id_t> dense_stats_out);

#endif
