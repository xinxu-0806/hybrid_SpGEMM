#ifndef ADAPTIVE_HBM_MERGE15_PACKETMETA4_TAPA_H
#define ADAPTIVE_HBM_MERGE15_PACKETMETA4_TAPA_H

#include <ap_int.h>
#include <tapa.h>

#include <cstdint>

using id_t = std::uint32_t;

// Gate input record.  Production integration replaces these four memory
// feeders with the 15 physical exit streams of merge_forward_tree8.
//
//   [127:0]   MergeForwardTree8 numerical token
//   [159:128] global column base for this row
//   [161:160] slot within one four-exit physical-port group
using Merge15PacketSourceRecord = ap_uint<192>;

void adaptive_hbm_merge15_packetmeta4_tapa(
		tapa::mmap<const Merge15PacketSourceRecord> source0, id_t source_count0,
		tapa::mmap<const Merge15PacketSourceRecord> source1, id_t source_count1,
		tapa::mmap<const Merge15PacketSourceRecord> source2, id_t source_count2,
		tapa::mmap<const Merge15PacketSourceRecord> source3, id_t source_count3,
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
		tapa::mmap<id_t> output_completion_stats);

#endif
