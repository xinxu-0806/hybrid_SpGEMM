#ifndef ADAPTIVE_HBM_MERGE_ALLOCATOR8_TAPA_H
#define ADAPTIVE_HBM_MERGE_ALLOCATOR8_TAPA_H

#include <ap_int.h>
#include <tapa.h>

#include <cstdint>

using id_t = std::uint32_t;

// Row command (128 bits):
//   row[31:0], global column base[63:32], nonempty shard mask[71:64].
//
// Dispatch record (128 bits):
//   row[31:0], base[63:32], source mask[71:64], allocated leaf mask[79:72],
//   context[82:80], generation[90:83], exit level[92:91],
//   exit index[95:93], intermediate/direct exit[96],
//   source-to-leaf map[120:97] (three bits per source; source mask qualifies it).
//
// Exit level 0 is a leaf forward, 1/2 are intermediate merge/forward nodes,
// and 3 is the root.  Therefore a short row does not traverse unused upper
// levels merely to serialize at the root.
using MergeAllocator8CommandToken = ap_uint<129>;
using MergeAllocator8EorToken = ap_uint<65>;
using MergeAllocator8DispatchToken = ap_uint<129>;
using MergeAllocator8AckToken = ap_uint<65>;

void adaptive_hbm_merge_allocator8_tapa(
		tapa::mmap<const ap_uint<128> > commands,
		id_t command_count,
		tapa::mmap<const ap_uint<64> > eor_events,
		id_t eor_count,
		tapa::mmap<ap_uint<128> > dispatches,
		tapa::mmap<ap_uint<64> > acknowledgements,
		tapa::mmap<id_t> statistics);

#endif
