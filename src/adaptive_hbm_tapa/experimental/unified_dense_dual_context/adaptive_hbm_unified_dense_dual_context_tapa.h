#ifndef ADAPTIVE_HBM_UNIFIED_DENSE_DUAL_CONTEXT_TAPA_H
#define ADAPTIVE_HBM_UNIFIED_DENSE_DUAL_CONTEXT_TAPA_H

#include <ap_int.h>
#include <tapa.h>

#include <cstdint>

using id_t = std::uint32_t;

// data[511:0], row[543:512], mask[551:544], physical dense slot[552],
// EOR[553], allocator context[556:554], generation[564:557],
// protocol error[565], terminal[566], logical 64K row uses both slots[567].
using DenseDualContextToken = ap_uint<568>;

// Input/output metadata matches unified_segmented_carry's DENSE ingress:
// row[31:0], mask[39:32], allocator context[42:40], generation[50:43],
// EOR[51], protocol error[52], physical dense slot[53], wide row[54].
void adaptive_hbm_unified_dense_dual_context_tapa(
		tapa::mmap<const ap_uint<512> > data_in,
		tapa::mmap<const ap_uint<64> > meta_in,
		id_t packet_count,
		id_t column_count,
		tapa::mmap<ap_uint<512> > output0,
		tapa::mmap<ap_uint<64> > output_meta0,
		tapa::mmap<ap_uint<64> > output_completion0,
		tapa::mmap<id_t> statistics0,
		tapa::mmap<ap_uint<512> > output1,
		tapa::mmap<ap_uint<64> > output_meta1,
		tapa::mmap<ap_uint<64> > output_completion1,
		tapa::mmap<id_t> statistics1);

#endif
