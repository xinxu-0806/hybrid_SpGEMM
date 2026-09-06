#ifndef ADAPTIVE_HBM_MERGE_KEYED_CARRY8_TAPA_H
#define ADAPTIVE_HBM_MERGE_KEYED_CARRY8_TAPA_H

#include <ap_int.h>
#include <tapa.h>

#include <cstdint>

using id_t = std::uint32_t;

// Input/output metadata:
//   row[31:0], item mask[39:32], context[42:40], generation[50:43], EOR[51].
// Each numerical lane is {FP32 value[63:32], global column[31:0]}.
// Input packets must already be sorted and packet-locally segmented.
using MergeKeyedCarry8InputToken = ap_uint<577>;

void adaptive_hbm_merge_keyed_carry8_tapa(
		tapa::mmap<const ap_uint<512> > data_in,
		tapa::mmap<const ap_uint<64> > meta_in,
		id_t packet_count,
		tapa::mmap<ap_uint<512> > data_out,
		tapa::mmap<ap_uint<64> > meta_out,
		tapa::mmap<id_t> statistics_out);

#endif
