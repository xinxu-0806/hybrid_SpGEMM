#ifndef ADAPTIVE_HBM_MERGE_FORWARD_TREE8_TAPA_H
#define ADAPTIVE_HBM_MERGE_FORWARD_TREE8_TAPA_H

#include <ap_int.h>
#include <tapa.h>

#include <cstdint>

using id_t = std::uint32_t;

// Numerical token (128 bits): local column[15:0], FP32 value[47:16],
// row[79:48], context[82:80], generation[90:83], exit level[92:91],
// EOR[93], allocation sequence[109:94].
using MergeForwardTree8Token = ap_uint<129>;

void adaptive_hbm_merge_forward_tree8_tapa(
		tapa::mmap<const ap_uint<128> > leaf0, id_t count0,
		tapa::mmap<const ap_uint<128> > leaf1, id_t count1,
		tapa::mmap<const ap_uint<128> > leaf2, id_t count2,
		tapa::mmap<const ap_uint<128> > leaf3, id_t count3,
		tapa::mmap<const ap_uint<128> > leaf4, id_t count4,
		tapa::mmap<const ap_uint<128> > leaf5, id_t count5,
		tapa::mmap<const ap_uint<128> > leaf6, id_t count6,
		tapa::mmap<const ap_uint<128> > leaf7, id_t count7,
		tapa::mmap<ap_uint<512> > output0,
		tapa::mmap<ap_uint<512> > output1,
		tapa::mmap<ap_uint<512> > output2,
		tapa::mmap<ap_uint<512> > output3,
		tapa::mmap<ap_uint<8> > masks0,
		tapa::mmap<ap_uint<8> > masks1,
		tapa::mmap<ap_uint<8> > masks2,
		tapa::mmap<ap_uint<8> > masks3,
		tapa::mmap<id_t> statistics0,
		tapa::mmap<id_t> statistics1,
		tapa::mmap<id_t> statistics2,
		tapa::mmap<id_t> statistics3);

#endif
