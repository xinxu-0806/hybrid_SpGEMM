#ifndef AP_INT_MAX_W
#define AP_INT_MAX_W 4096
#endif
#include "adaptive_hbm_tapa_scalable.h"

#include <cstdint>
#include <hls_stream.h>

constexpr int kTapaNumShards = 8;
constexpr int kTapaDenseCapacity = ADAPT_FP32_DENSE_CAPACITY;
constexpr int kTapaPacketsPerShard = 1024;
// AiSpGEMM's c-52 and hangGlider_3 expose 1,714- and 9,119-run output
// rows.  B-row work sharding keeps those rows below roughly 1,200 runs per
// local merger, but the former 256-entry descriptor table rejected them before
// the existing base-8 hierarchy could do useful work.  Keep packet storage
// unchanged and enlarge only the small run metadata RAMs.  The Host verifies
// the exact post-sharding run/product bounds before launching the kernel.
constexpr int kTapaRunsPerShard = 2048;
constexpr int kTapaItemsPerShard = kTapaPacketsPerShard * 8;
constexpr int kTapaLocalMergeWays = 8;
constexpr int kTapaMergePacketsPerBank = 256;
// Stripe a very long sorted B row across local merge banks.  Otherwise a
// single 9k-entry run (hangGlider_3's hub row) fills one 256-packet BRAM bank
// while the other seven banks remain idle.
constexpr int kTapaMergeRunPackets = kTapaMergePacketsPerBank;
constexpr int kTapaLocalItemsPerShard
	= kTapaLocalMergeWays * kTapaMergePacketsPerBank * 8;
// Local MERGE descriptor bounds are much narrower than the external 32-bit
// ABI: a packet-bank address is at most 255 and a shard holds at most 16,384
// items.  Keeping the 2,048-entry long-row tables at id_t needlessly spends
// BRAM capacity and was a material placement/congestion cost on U280.
using TapaMergePacketIndex = ap_uint<9>;
using TapaMergeItemCount = ap_uint<15>;
constexpr int kTapaDenseBanks = 8;
constexpr int kTapaDensePhases = 11;
constexpr int kTapaDenseRouteSlots = 8;
constexpr int kTapaDenseRefillSlot = 5;
constexpr int kTapaDenseDecodeSlot = 4;
// Decoded bank heads are consumed at slot zero and regenerated whenever the
// packet reaches the decode slot.  They therefore need only the one-way
// decode-to-active window; recycling them through the remaining packet slots
// creates a redundant 8x8x8x64-bit feedback network.
constexpr int kTapaDenseHeadSlots = kTapaDenseDecodeSlot + 1;
constexpr int kTapaDenseIssueDelay = 3;
// Once every source has delivered its row boundary, no new packet can enter
// the router.  Terminate only after a complete slot rotation plus the delayed
// retire/decode pipeline has produced no issue.  This is both cheaper than a
// wide all-slot empty reduction and exact for an arbitrarily bank-conflicted
// tail (the former fixed 64-cycle drain could leave up to 512 lanes pending).
constexpr int kTapaDenseDrainQuietCycles
	= kTapaDenseRouteSlots + kTapaDenseIssueDelay + 2;
// A DENSE row is selected precisely when many A-row tasks revisit a small
// output footprint.  Those tasks also tend to revisit the same short B rows
// (for example, every row of a clique reads the same clique working set).
// Keep a tiny recent-B-row cache beside each of the eight HBM readers.  Sixteen
// rows by eight 512-bit beats are only 8 KiB per reader and cover the per-shard
// working set of a 64-column clique.  Tags are updated once per completed row,
// not once per beat, so the II=1 fetch loop has no tag write/read recurrence.
// MERGE/DIRECT retain the original uncached burst path.
constexpr int kTapaDenseBReaderCacheBeats = 128;
constexpr int kTapaDenseBReaderCacheRows = 16;
constexpr int kTapaDenseBReaderCacheRowBeats = 8;
constexpr int kTapaDenseBankCapacity
	= kTapaDenseCapacity / kTapaDenseBanks;
constexpr int kTapaDenseLeafWords = kTapaDenseBankCapacity / 64;
constexpr int kTapaDenseEpochBits = 16;
// Internal MERGE micro-route only.  Host-visible routing remains
// MERGE-only/DENSE-only/ADAPTIVE; a sufficiently heavy MERGE row uses all
// eight global roots while an ordinary row owns one of eight row contexts.
// The threshold is deliberately a compile-time architecture constant for the
// first integrated checkpoint and will be calibrated only after board data.
constexpr id_t kTapaMergeHeavyProductThreshold = 4096;

// RowHeader: physical fragment[31:0], task_count[63:32], mode[65:64],
// done[66], expected_products[98:67], command_last_row[99],
// segment_columns[131:100], dense_base[163:132], logical_row_last[164].
// The appended fields are zero in the legacy single-tile top.  A commanded
// invocation supplies them on every row so the shared reducer can change the
// N tile width without terminating and relaunching the TAPA task graph.
using TapaRowHeader = ap_uint<165>;
// ReaderCommand: descriptor[127:0], row[159:128], mode[161:160],
// row_end[162], done[163], dense_base[195:164], segment_columns[227:196],
// internal_merge_heavy[228].
using TapaReaderCommand = ap_uint<229>;
// ProductPacket: eight scaled [fp32|column] items[511:0], valid[515:512],
// row[547:516], mode[549:548], run_end[550], row_end[551], done[552],
// local_error[553], internal_merge_heavy[554].  Row-end packets carry the
// pre-merge product count in data[31:0] so numerical preprocessing does not
// change Host-visible stats.
using TapaProductPacket = ap_uint<555>;
// Internal B-reader token: raw B beat[511:0], valid[515:512], row[547:516],
// mode[549:548], run_end[550], row_end[551], done[552], scale[584:553],
// dense_base[616:585], segment_columns[648:617],
// internal_merge_heavy[649].
// Separating raw fetch/cache service from the eight FP32 multiplies prevents
// HLS from resource-sharing an AXI/cache state machine with the scale stage.
using TapaBReaderRawPacket = ap_uint<650>;
// A complete logical row may span several physical column fragments, but each
// fragment is at most 64K columns and is reduced independently.  Store a
// fragment-local 16-bit key plus FP32 value in the BRAM-heavy local MERGE
// hierarchy, then restore the command's 32-bit base at the global reducer.
// This preserves complete-row routing/global CSR columns without paying for a
// 64-bit item in both ping-pong buffers (96 extra BRAM tiles on U280).
using TapaLocalMergeItem = ap_uint<48>;
using TapaMergePacket = ap_uint<8 * 48>;
// RowLength: fragment nnz[31:0], done[32], logical_row_last[33].
using TapaRowLength = ap_uint<34>;
// OutputPacket: eight compacted [fp32|column] items[511:0], valid[515:512].
using TapaOutputPacket = ap_uint<516>;
using TapaDenseCell = ap_uint<32 + kTapaDenseEpochBits>;
// Bank-wide DENSE extraction keeps the eight logical low-column positions of
// one accumulator address together.  CandidateGroup: address[12:0],
// active-low mask[20:13], done[21].
using TapaDenseCandidateGroup = ap_uint<22>;
// ReducedGroup: eight [fp32|column] items in physical-bank order[511:0],
// candidate-bank mask[519:512], nonzero-bank mask[527:520],
// low-to-bank XOR permutation[530:528], done[531].
using TapaDenseReducedGroup = ap_uint<532>;
// Global-MERGE selection token: column[31:0], eight FP32 contributions
// [287:32], done[288].  Selection/stream-head advancement is deliberately
// separated from the floating-point tree so the 512-bit head recurrence does
// not inherit the FP adder latency.
using TapaMergeContributions = ap_uint<289>;
// Streaming MERGE token: packed [fp32|column] item[63:0], done[64].
// Keeping the floating-point reduction in the producer and the indexed
// row-output write in a separate consumer removes a false loop-carried
// dependence that otherwise forces the global merge loop to II=10.
using TapaMergeItem = ap_uint<65>;
// Persistent banked-MERGE output: eight compacted [fp32|local-column] items
// [511:0], valid[515:512], physical row[547:516], EOR[548], global done[549],
// protocol status[551:550], heavy-row micro-route[552], and the original
// (pre-local-merge) product count[584:553].
using TapaMergeResultPacket = ap_uint<585>;

// DENSE is deliberately split into three independently scheduled pipelines.
// The selector advances the eight packet heads and emits, for every bank, a
// reference column plus up to eight raw FP32 contributions.  A second stage
// owns the eight balanced FP trees.  The last stage alone touches the
// phase-partitioned accumulator/bitmap memories.  Keeping these recurrences in
// different processes avoids one enormous HLS scheduling graph and prevents
// FP latency from feeding back into packet-head arbitration.
constexpr int kTapaDenseContributionBankBits = 298;
constexpr int kTapaDenseContributionsBits
	= kTapaDenseBanks * kTapaDenseContributionBankBits + 1;
using TapaDenseContributions = ap_uint<kTapaDenseContributionsBits>;
constexpr int kTapaDenseUpdateBankBits = 70;
constexpr int kTapaDenseUpdatesBits
	= kTapaDenseBanks * kTapaDenseUpdateBankBits + 1;
using TapaDenseUpdates = ap_uint<kTapaDenseUpdatesBits>;
// Stateless packet-refill result: packet[554:0], bank metadata[578:555],
// valid mask[586:579], row context[587], data[588], boundary[589],
// reported products[621:590], protocol errors[623:622], end-both[624].
// Returning this value instead of mutating the route-slot arrays through a
// helper port lets HLS honor the outer distance-eight dependence declaration.
using TapaDenseRefill = ap_uint<625>;

// Append-only diagnostic payload returned through the TAPA statistics ABI.
// These are work/cycle counters, not a software timing estimate: counters for
// II=1 loops are the exact scheduled loop iterations, while the hierarchy
// counters expose the fixed per-group overhead reported by HLS.
struct TapaDenseDiagnostics {
	id_t stream_cycles;
	id_t route_cycles;
	id_t replay_cycles;
	id_t raw_dependencies;
	id_t epoch_clear_cycles;
	id_t bank_products[kTapaDenseBanks];
	id_t bank_unique_addresses[kTapaDenseBanks];
	id_t touched_addresses;
	id_t bitmap_clear_cycles;
	id_t bitmap_build_cycles;
	id_t bitmap_word_scans;
	id_t nonempty_words;
	id_t active_address_groups;
	id_t candidate_columns;
	id_t extract_iterations;
	id_t output_packets;
	id_t coalesced_products;
};

static float tapa_bits_to_float(ap_uint<32> bits) {
#pragma HLS INLINE
	uint32_t raw = bits;
	return reinterpret_cast<const float&>(raw);
}

static ap_uint<32> tapa_float_to_bits(float value) {
#pragma HLS INLINE
	uint32_t raw = reinterpret_cast<const uint32_t&>(value);
	return raw;
}

static ap_uint<64> tapa_pack_item(ap_uint<32> col, float value) {
#pragma HLS INLINE
	ap_uint<64> item = 0;
	item.range(31, 0) = col;
	item.range(63, 32) = tapa_float_to_bits(value);
	return item;
}

static TapaLocalMergeItem tapa_pack_merge_item(ap_uint<16> col, float value) {
#pragma HLS INLINE
	TapaLocalMergeItem item = 0;
	item.range(15, 0) = col;
	item.range(47, 16) = tapa_float_to_bits(value);
	return item;
}

static TapaMergePacket tapa_compress_merge_packet(ap_uint<512> packet) {
#pragma HLS INLINE
	TapaMergePacket compact = 0;
	for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
		compact.range(lane * 48 + 15, lane * 48)
			= packet.range(lane * 64 + 15, lane * 64);
		compact.range(lane * 48 + 47, lane * 48 + 16)
			= packet.range(lane * 64 + 63, lane * 64 + 32);
	}
	return compact;
}

static ap_uint<512> tapa_expand_merge_packet(TapaMergePacket compact) {
#pragma HLS INLINE
	ap_uint<512> packet = 0;
	for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
		packet.range(lane * 64 + 15, lane * 64)
			= compact.range(lane * 48 + 15, lane * 48);
		packet.range(lane * 64 + 63, lane * 64 + 32)
			= compact.range(lane * 48 + 47, lane * 48 + 16);
	}
	return packet;
}

static bool tapa_nonzero(float value) {
#pragma HLS INLINE
	return (tapa_float_to_bits(value) & 0x7fffffffU) != 0;
}

static ap_uint<3> tapa_dense_bank(id_t col) {
#pragma HLS INLINE
	const ap_uint<32> value = col;
	// XOR folding preserves a one-to-one mapping for the eight low column
	// bits at each local address while breaking common stride-8 bank hot spots.
	return (value ^ (value >> 3) ^ (value >> 6)).range(2, 0);
}

static ap_uint<6> tapa_first_set64(ap_uint<64> bits) {
#pragma HLS INLINE
	ap_uint<3> byte = 0;
	if (bits.range(7, 0) != 0) byte = 0;
	else if (bits.range(15, 8) != 0) byte = 1;
	else if (bits.range(23, 16) != 0) byte = 2;
	else if (bits.range(31, 24) != 0) byte = 3;
	else if (bits.range(39, 32) != 0) byte = 4;
	else if (bits.range(47, 40) != 0) byte = 5;
	else if (bits.range(55, 48) != 0) byte = 6;
	else byte = 7;
	const ap_uint<8> octet = bits >> ((ap_uint<6>)byte << 3);
	ap_uint<3> bit = 0;
	if (octet[0]) bit = 0;
	else if (octet[1]) bit = 1;
	else if (octet[2]) bit = 2;
	else if (octet[3]) bit = 3;
	else if (octet[4]) bit = 4;
	else if (octet[5]) bit = 5;
	else if (octet[6]) bit = 6;
	else bit = 7;
	return ((ap_uint<6>)byte << 3) | bit;
}

static ap_uint<3> tapa_first_set8(ap_uint<8> bits) {
#pragma HLS INLINE
	ap_uint<3> bit = 0;
	if (bits[0]) bit = 0;
	else if (bits[1]) bit = 1;
	else if (bits[2]) bit = 2;
	else if (bits[3]) bit = 3;
	else if (bits[4]) bit = 4;
	else if (bits[5]) bit = 5;
	else if (bits[6]) bit = 6;
	else bit = 7;
	return bit;
}

static ap_uint<4> tapa_popcount8(ap_uint<8> bits) {
#pragma HLS INLINE
	const ap_uint<2> count01 = bits[0] + bits[1];
	const ap_uint<2> count23 = bits[2] + bits[3];
	const ap_uint<2> count45 = bits[4] + bits[5];
	const ap_uint<2> count67 = bits[6] + bits[7];
	const ap_uint<3> count03 = count01 + count23;
	const ap_uint<3> count47 = count45 + count67;
	return count03 + count47;
}

static float tapa_sum8_fp32(float value0, float value1, float value2,
		float value3, float value4, float value5, float value6, float value7) {
#pragma HLS INLINE
	const float sum01 = value0 + value1;
	const float sum23 = value2 + value3;
	const float sum45 = value4 + value5;
	const float sum67 = value6 + value7;
	const float sum03 = sum01 + sum23;
	const float sum47 = sum45 + sum67;
	const float sum = sum03 + sum47;
#pragma HLS BIND_OP variable=sum01 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum23 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum45 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum67 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum03 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum47 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum op=fadd impl=fulldsp latency=5
	return sum;
}

static ap_uint<7> tapa_first_set128(ap_uint<128> bits) {
#pragma HLS INLINE
	const ap_uint<64> low = bits.range(63, 0);
	return low != 0 ? (ap_uint<7>)tapa_first_set64(low)
		: (ap_uint<7>)(64 + tapa_first_set64(bits.range(127, 64)));
}

static void tapa_store_row_item(
		ap_uint<64> row_output[8][kTapaDenseCapacity / 8],
		id_t position, ap_uint<64> item) {
#pragma HLS INLINE
	row_output[position & 7][position >> 3] = item;
}

// Store one compact MERGE packet without presenting eight dynamic writes to
// every physical row-output bank.  Although consecutive positions necessarily
// map to distinct banks, calling tapa_store_row_item from an unrolled loop
// makes Vitis HLS conservatively see eight possible writes to the same RAM and
// raises the loop II to eight.  Specializing the rotation makes each bank
// index compile-time constant; only its address/data mux remains dynamic.
template <int START_BANK>
static void tapa_store_row_packet_rotated(
		ap_uint<64> row_output[8][kTapaDenseCapacity / 8],
		id_t base_word, const ap_uint<512>& items, ap_uint<8> write_mask) {
#pragma HLS INLINE
	for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
		const int bank = (START_BANK + lane) & 7;
		const id_t word = base_word + ((START_BANK + lane) >> 3);
		if (write_mask[lane])
			row_output[bank][word]
				= items.range(lane * 64 + 63, lane * 64);
	}
}

static void tapa_store_row_packet(
		ap_uint<64> row_output[8][kTapaDenseCapacity / 8],
		id_t position, const ap_uint<512>& items, ap_uint<8> write_mask) {
#pragma HLS INLINE
	const id_t base_word = position >> 3;
	switch ((unsigned)(position & 7)) {
	case 0: tapa_store_row_packet_rotated<0>(row_output, base_word, items,
		write_mask); break;
	case 1: tapa_store_row_packet_rotated<1>(row_output, base_word, items,
		write_mask); break;
	case 2: tapa_store_row_packet_rotated<2>(row_output, base_word, items,
		write_mask); break;
	case 3: tapa_store_row_packet_rotated<3>(row_output, base_word, items,
		write_mask); break;
	case 4: tapa_store_row_packet_rotated<4>(row_output, base_word, items,
		write_mask); break;
	case 5: tapa_store_row_packet_rotated<5>(row_output, base_word, items,
		write_mask); break;
	case 6: tapa_store_row_packet_rotated<6>(row_output, base_word, items,
		write_mask); break;
	default: tapa_store_row_packet_rotated<7>(row_output, base_word, items,
		write_mask); break;
	}
}

#include "adaptive_hbm_tapa_merge_fabric.hpp"

// A three-stage butterfly implements the flexible merge/forward network.
// Every stage first selects one destination bit and then merges two sorted
// subsequences.  This uses 24 two-way reducers instead of the former 8x8
// crossbar plus 56 reducers, while leaving eight independently draining roots.
#define TAPA_INVOKE_FLEX_SPLIT0(I) \
	.invoke(tapa_merge_flexible_split0, merge_source[I], \
		merge_split0[2 * (I)], merge_split0[2 * (I) + 1])
#define TAPA_INVOKE_FLEX_SPLIT1(I) \
	.invoke(tapa_merge_flexible_split1, merge_stage0[I], \
		merge_split1[2 * (I)], merge_split1[2 * (I) + 1])
#define TAPA_INVOKE_FLEX_SPLIT2(I) \
	.invoke(tapa_merge_flexible_split2, merge_stage1[I], \
		merge_split2[2 * (I)], merge_split2[2 * (I) + 1])
#define TAPA_INVOKE_FLEX_MERGE0(A, B, O) \
	.invoke(tapa_merge_reduce2, merge_split0[A], merge_split0[B], \
		merge_stage0[O])
#define TAPA_INVOKE_FLEX_MERGE1(A, B, O) \
	.invoke(tapa_merge_reduce2, merge_split1[A], merge_split1[B], \
		merge_stage1[O])
#define TAPA_INVOKE_FLEX_MERGE2(A, B, O) \
	.invoke(tapa_merge_reduce2, merge_split2[A], merge_split2[B], \
		merge_stage2[O])
#define TAPA_INVOKE_FLEX_FILTER(I) \
	.invoke(tapa_merge_flexible_filter_padding, merge_stage2[I], \
		merge_reduced[I])

static ap_uint<16> tapa_min8_local16(ap_uint<16> value0, ap_uint<16> value1,
		ap_uint<16> value2, ap_uint<16> value3, ap_uint<16> value4,
		ap_uint<16> value5, ap_uint<16> value6, ap_uint<16> value7) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#pragma HLS LATENCY min=1 max=1
	// The Host translates an arbitrary global column index into a column-local
	// key in [0, ADAPT_FP32_DENSE_CAPACITY).  Thus the local key fits in 16 bits
	// even when the original matrix has millions of columns; 65536 is physical
	// invocation capacity, not an accepted-matrix-size or selector threshold.
	//
	// Compare only the physical fragment-local key.  Carrying a zero-extended
	// 32-bit key through this tournament needlessly doubled the comparator carry
	// chains and became the final 160-MHz routed critical path after local MERGE
	// items were compacted to 48 bits.  The command's 32-bit base is restored
	// only after the global reduction.  Lower-numbered inputs win ties,
	// preserving the duplicate-consumption semantics of MERGE.
	const bool le01 = value0 <= value1;
	const bool le02 = value0 <= value2;
	const bool le03 = value0 <= value3;
	const bool le04 = value0 <= value4;
	const bool le05 = value0 <= value5;
	const bool le06 = value0 <= value6;
	const bool le07 = value0 <= value7;
	const bool le12 = value1 <= value2;
	const bool le13 = value1 <= value3;
	const bool le14 = value1 <= value4;
	const bool le15 = value1 <= value5;
	const bool le16 = value1 <= value6;
	const bool le17 = value1 <= value7;
	const bool le23 = value2 <= value3;
	const bool le24 = value2 <= value4;
	const bool le25 = value2 <= value5;
	const bool le26 = value2 <= value6;
	const bool le27 = value2 <= value7;
	const bool le34 = value3 <= value4;
	const bool le35 = value3 <= value5;
	const bool le36 = value3 <= value6;
	const bool le37 = value3 <= value7;
	const bool le45 = value4 <= value5;
	const bool le46 = value4 <= value6;
	const bool le47 = value4 <= value7;
	const bool le56 = value5 <= value6;
	const bool le57 = value5 <= value7;
	const bool le67 = value6 <= value7;

	const bool winner0 = le01 && le02 && le03 && le04
		&& le05 && le06 && le07;
	const bool winner1 = !le01 && le12 && le13 && le14
		&& le15 && le16 && le17;
	const bool winner2 = !le02 && !le12 && le23 && le24
		&& le25 && le26 && le27;
	const bool winner3 = !le03 && !le13 && !le23 && le34
		&& le35 && le36 && le37;
	const bool winner4 = !le04 && !le14 && !le24 && !le34
		&& le45 && le46 && le47;
	const bool winner5 = !le05 && !le15 && !le25 && !le35
		&& !le45 && le56 && le57;
	const bool winner6 = !le06 && !le16 && !le26 && !le36
		&& !le46 && !le56 && le67;
	const bool winner7 = !le07 && !le17 && !le27 && !le37
		&& !le47 && !le57 && !le67;

	const ap_uint<16> selected0 = winner0 ? value0 : ap_uint<16>(0);
	const ap_uint<16> selected1 = winner1 ? value1 : ap_uint<16>(0);
	const ap_uint<16> selected2 = winner2 ? value2 : ap_uint<16>(0);
	const ap_uint<16> selected3 = winner3 ? value3 : ap_uint<16>(0);
	const ap_uint<16> selected4 = winner4 ? value4 : ap_uint<16>(0);
	const ap_uint<16> selected5 = winner5 ? value5 : ap_uint<16>(0);
	const ap_uint<16> selected6 = winner6 ? value6 : ap_uint<16>(0);
	const ap_uint<16> selected7 = winner7 ? value7 : ap_uint<16>(0);
	const ap_uint<16> selected01 = selected0 | selected1;
	const ap_uint<16> selected23 = selected2 | selected3;
	const ap_uint<16> selected45 = selected4 | selected5;
	const ap_uint<16> selected67 = selected6 | selected7;
	return (selected01 | selected23) | (selected45 | selected67);
}

static ap_uint<32> tapa_min16(const ap_uint<32> value[16]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=value complete
	const ap_uint<32> min01 = value[0] < value[1] ? value[0] : value[1];
	const ap_uint<32> min23 = value[2] < value[3] ? value[2] : value[3];
	const ap_uint<32> min45 = value[4] < value[5] ? value[4] : value[5];
	const ap_uint<32> min67 = value[6] < value[7] ? value[6] : value[7];
	const ap_uint<32> min89 = value[8] < value[9] ? value[8] : value[9];
	const ap_uint<32> min_ab = value[10] < value[11] ? value[10] : value[11];
	const ap_uint<32> min_cd = value[12] < value[13] ? value[12] : value[13];
	const ap_uint<32> min_ef = value[14] < value[15] ? value[14] : value[15];
	const ap_uint<32> min03 = min01 < min23 ? min01 : min23;
	const ap_uint<32> min47 = min45 < min67 ? min45 : min67;
	const ap_uint<32> min8b = min89 < min_ab ? min89 : min_ab;
	const ap_uint<32> min_cf = min_cd < min_ef ? min_cd : min_ef;
	const ap_uint<32> min07 = min03 < min47 ? min03 : min47;
	const ap_uint<32> min8f = min8b < min_cf ? min8b : min_cf;
	return min07 < min8f ? min07 : min8f;
}

static ap_uint<2> tapa_select_mode(ap_uint<2> routed, id_t task_count,
		id_t force_mode, id_t N, bool merge_capacity_locked) {
#pragma HLS INLINE
	ap_uint<2> selected = routed;
	if (task_count == 0) selected = ADAPT_FP32_EMPTY;
	else if (task_count == 1) selected = ADAPT_FP32_DIRECT;
	else if (force_mode == ADAPT_FP32_FORCE_MERGE && !merge_capacity_locked)
		selected = ADAPT_FP32_MERGE;
	else if (force_mode == ADAPT_FP32_FORCE_DENSE
			&& N <= kTapaDenseCapacity)
		selected = ADAPT_FP32_DENSE;
	if (selected == ADAPT_FP32_DENSE && N > kTapaDenseCapacity)
		selected = ADAPT_FP32_MERGE;
	return selected;
}

static TapaReaderCommand tapa_make_command(ap_uint<128> descriptor,
		id_t row, ap_uint<2> mode, id_t dense_base, id_t N,
		bool merge_heavy) {
#pragma HLS INLINE
	TapaReaderCommand command = 0;
	command.range(127, 0) = descriptor;
	command.range(159, 128) = row;
	command.range(161, 160) = mode;
	command.range(195, 164) = dense_base;
	command.range(227, 196) = N;
	command[228] = merge_heavy;
	return command;
}

static void tapa_dispatch_tile(tapa::mmap<const ap_uint<512> > task_words,
		tapa::mmap<const id_t> row_task_ptr,
		tapa::mmap<const ap_uint<32> > route,
		id_t task_base, id_t rowptr_base, id_t route_base,
		id_t virtual_row_base, id_t M, id_t N, id_t dense_base,
		bool logical_row_last, id_t force_mode,
		bool emit_global_done,
		tapa::ostream<TapaRowHeader>& row_headers,
		tapa::ostream<TapaReaderCommand>& command0,
		tapa::ostream<TapaReaderCommand>& command1,
		tapa::ostream<TapaReaderCommand>& command2,
		tapa::ostream<TapaReaderCommand>& command3,
		tapa::ostream<TapaReaderCommand>& command4,
		tapa::ostream<TapaReaderCommand>& command5,
		tapa::ostream<TapaReaderCommand>& command6,
		tapa::ostream<TapaReaderCommand>& command7) {
	id_t base_row = 0;
	while (base_row < M) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=32768 avg=4096
		const id_t rows_in_group = base_row + 1 < M ? 2 : 1;
		id_t task_begin[2];
		id_t task_end[2];
		id_t next_task[2];
		ap_uint<2> mode[2];
		bool merge_heavy[2];
		ap_uint<512> cached_word[2];
		id_t cached_index[2];
#pragma HLS ARRAY_PARTITION variable=task_begin complete
#pragma HLS ARRAY_PARTITION variable=task_end complete
#pragma HLS ARRAY_PARTITION variable=next_task complete
#pragma HLS ARRAY_PARTITION variable=mode complete
#pragma HLS ARRAY_PARTITION variable=merge_heavy complete
#pragma HLS ARRAY_PARTITION variable=cached_word complete
#pragma HLS ARRAY_PARTITION variable=cached_index complete
		for (int context = 0; context < 2; ++context) {
#pragma HLS UNROLL
			const bool active = context < rows_in_group;
			const id_t row = base_row + context;
			task_begin[context] = active
				? task_base + row_task_ptr[rowptr_base + row] : 0;
			task_end[context] = active
				? task_base + row_task_ptr[rowptr_base + row + 1] : 0;
			next_task[context] = task_begin[context];
			const id_t task_count = task_end[context] - task_begin[context];
			const ap_uint<32> route_word = active
				? route[route_base + row] : (ap_uint<32>)0;
			mode[context] = active ? tapa_select_mode(
				route_word.range(1, 0), task_count, force_mode, N,
				route_word[31])
				: (ap_uint<2>)ADAPT_FP32_EMPTY;
			merge_heavy[context] = active
				&& mode[context] == ADAPT_FP32_MERGE
				&& (id_t)route_word.range(30, 2)
					>= kTapaMergeHeavyProductThreshold;
			cached_word[context] = 0;
			cached_index[context] = ~id_t(0);
			if (active) {
				TapaRowHeader header = 0;
				header.range(31, 0) = virtual_row_base + row;
				header.range(63, 32) = task_count;
				header.range(65, 64) = mode[context];
				// Bit 31 is the scalable Host's capacity lock.  It prevents a
				// diagnostic force-MERGE run from sending a provably oversized row
				// into local MERGE, and is not part of the product-count metadata.
				header.range(98, 67) = route_word.range(30, 2);
				header[99] = row + 1 == M;
				header.range(131, 100) = N;
				header.range(163, 132) = dense_base;
				// Row-persistent commands currently contain one fragment.  Legacy
				// multi-row commands pass true and therefore keep one CSR row per
				// header exactly as before.
				header[164] = logical_row_last;
				row_headers.write(header);
			}
		}

		// For an aligned pair of DENSE rows, alternate complete B-row tasks.
		// Every reader can advance to the other row while a busy shard finishes
		// its current burst, exposing both row contexts to the shared bank router.
		const bool dense_pair = rows_in_group == 2 && N <= 32768
			&& mode[0] == ADAPT_FP32_DENSE
			&& mode[1] == ADAPT_FP32_DENSE;
		if (dense_pair) {
			ap_uint<1> turn = 0;
			while (next_task[0] < task_end[0]
					|| next_task[1] < task_end[1]) {
#pragma HLS PIPELINE II=1
				ap_uint<1> context = 0;
				const bool turn_ready = next_task[(unsigned)turn]
					< task_end[(unsigned)turn];
				context = turn_ready ? turn : (ap_uint<1>)(turn ^ 1);
				const id_t task = next_task[(unsigned)context]++;
				const id_t word_index = task >> 2;
				if (word_index != cached_index[(unsigned)context]) {
					cached_word[(unsigned)context] = task_words[word_index];
					cached_index[(unsigned)context] = word_index;
				}
				const ap_uint<2> lane = task & 3;
				const ap_uint<128> descriptor
					= cached_word[(unsigned)context].range(
						(unsigned)lane * 128 + 127, (unsigned)lane * 128);
				const TapaReaderCommand command = tapa_make_command(descriptor,
					virtual_row_base + base_row + context,
					mode[(unsigned)context], dense_base, N,
					merge_heavy[(unsigned)context]);
				switch ((unsigned)descriptor.range(58, 56)) {
				case 0: command0.write(command); break;
				case 1: command1.write(command); break;
				case 2: command2.write(command); break;
				case 3: command3.write(command); break;
				case 4: command4.write(command); break;
				case 5: command5.write(command); break;
				case 6: command6.write(command); break;
				default: command7.write(command); break;
				}
				turn = context ^ 1;
			}
			// Both boundaries follow all interleaved products; every shard can
			// therefore retain the second row's count until its second boundary.
			for (int context = 0; context < 2; ++context) {
				TapaReaderCommand boundary = tapa_make_command(
					0, virtual_row_base + base_row + context, mode[context],
					dense_base, N, merge_heavy[context]);
				boundary[162] = 1;
				command0.write(boundary);
				command1.write(boundary);
				command2.write(boundary);
				command3.write(boundary);
				command4.write(boundary);
				command5.write(boundary);
				command6.write(boundary);
				command7.write(boundary);
			}
		} else {
			// All other combinations preserve strict row serialization.  In
			// particular, a MERGE row must see its boundary before any packet from
			// the following row reaches its local sorter.
			for (int context = 0; context < 2; ++context) {
				if (context >= rows_in_group) continue;
				while (next_task[context] < task_end[context]) {
#pragma HLS PIPELINE II=1
					const id_t task = next_task[context]++;
					const id_t word_index = task >> 2;
					if (word_index != cached_index[context]) {
						cached_word[context] = task_words[word_index];
						cached_index[context] = word_index;
					}
					const ap_uint<2> lane = task & 3;
					const ap_uint<128> descriptor
						= cached_word[context].range(
							(unsigned)lane * 128 + 127,
							(unsigned)lane * 128);
					const TapaReaderCommand command = tapa_make_command(
						descriptor, virtual_row_base + base_row + context,
						mode[context], dense_base, N, merge_heavy[context]);
					switch ((unsigned)descriptor.range(58, 56)) {
					case 0: command0.write(command); break;
					case 1: command1.write(command); break;
					case 2: command2.write(command); break;
					case 3: command3.write(command); break;
					case 4: command4.write(command); break;
					case 5: command5.write(command); break;
					case 6: command6.write(command); break;
					default: command7.write(command); break;
					}
				}
				TapaReaderCommand boundary = tapa_make_command(
					0, virtual_row_base + base_row + context, mode[context],
					dense_base, N, merge_heavy[context]);
				boundary[162] = 1;
				command0.write(boundary);
				command1.write(boundary);
				command2.write(boundary);
				command3.write(boundary);
				command4.write(boundary);
				command5.write(boundary);
				command6.write(boundary);
				command7.write(boundary);
			}
		}
		base_row += rows_in_group;
	}

	if (emit_global_done) {
		TapaRowHeader header_done = 0;
		header_done[66] = 1;
		row_headers.write(header_done);
		TapaReaderCommand command_done = 0;
		command_done[163] = 1;
		command0.write(command_done);
		command1.write(command_done);
		command2.write(command_done);
		command3.write(command_done);
		command4.write(command_done);
		command5.write(command_done);
		command6.write(command_done);
		command7.write(command_done);
	}
}

void tapa_dispatch_rows(tapa::mmap<const ap_uint<512> > task_words,
		tapa::mmap<const id_t> row_task_ptr,
		tapa::mmap<const ap_uint<32> > route,
		id_t M, id_t N, id_t force_mode,
		tapa::ostream<TapaRowHeader>& row_headers,
		tapa::ostream<TapaReaderCommand>& command0,
		tapa::ostream<TapaReaderCommand>& command1,
		tapa::ostream<TapaReaderCommand>& command2,
		tapa::ostream<TapaReaderCommand>& command3,
		tapa::ostream<TapaReaderCommand>& command4,
		tapa::ostream<TapaReaderCommand>& command5,
		tapa::ostream<TapaReaderCommand>& command6,
		tapa::ostream<TapaReaderCommand>& command7) {
	tapa_dispatch_tile(task_words, row_task_ptr, route,
		0, 0, 0, 0, M, N, 0, true, force_mode, true,
		row_headers, command0, command1, command2, command3,
		command4, command5, command6, command7);
}

// Leda-style persistent dispatcher.  Commands describe dimension tiles, but
// all numerical work still enters the existing unified MERGE/DENSE graph.
// Virtual row bases are supplied by the Host and kept even-aligned so a DENSE
// pair never crosses a tile boundary.
void tapa_dispatch_commands(tapa::mmap<const ap_uint<512> > tile_commands,
		tapa::mmap<const ap_uint<512> > task_words,
		tapa::mmap<const id_t> row_task_ptr,
		tapa::mmap<const ap_uint<32> > route,
		id_t command_count, id_t force_mode,
		tapa::ostream<TapaRowHeader>& row_headers,
		tapa::ostream<TapaReaderCommand>& command0,
		tapa::ostream<TapaReaderCommand>& command1,
		tapa::ostream<TapaReaderCommand>& command2,
		tapa::ostream<TapaReaderCommand>& command3,
		tapa::ostream<TapaReaderCommand>& command4,
		tapa::ostream<TapaReaderCommand>& command5,
		tapa::ostream<TapaReaderCommand>& command6,
		tapa::ostream<TapaReaderCommand>& command7) {
	if (command_count == 0) {
		tapa_dispatch_tile(task_words, row_task_ptr, route,
			0, 0, 0, 0, 0, 1, 0, true, force_mode, true,
			row_headers, command0, command1, command2, command3,
			command4, command5, command6, command7);
		return;
	}
	for (id_t command_index = 0; command_index < command_count;
			++command_index) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256 avg=4
		const ap_uint<512> descriptor = tile_commands[command_index];
		const id_t task_base = descriptor.range(31, 0);
		const id_t rowptr_base = descriptor.range(63, 32);
		const id_t route_base = descriptor.range(95, 64);
		const id_t tile_rows = descriptor.range(127, 96);
		const id_t tile_columns = descriptor.range(159, 128);
		const id_t virtual_row_base = descriptor.range(191, 160);
		const bool row_persistent
			= descriptor.range(511, 480) == (ap_uint<32>)0x52504331U;
		const id_t dense_base = row_persistent
			? (id_t)descriptor.range(223, 192) : (id_t)0;
		const bool logical_row_last = row_persistent
			? (bool)descriptor[224] : true;
		tapa_dispatch_tile(task_words, row_task_ptr, route,
			task_base, rowptr_base, route_base, virtual_row_base,
			tile_rows, tile_columns, dense_base, logical_row_last, force_mode,
			command_index + 1 == command_count,
			row_headers, command0, command1, command2, command3,
			command4, command5, command6, command7);
	}
}

// Fetch/cache stage for one physical HBM reader.  It deliberately contains no
// FP arithmetic; every successful iteration emits one raw 512-bit B beat.
void tapa_b_reader_fetch(tapa::mmap<const ap_uint<512> > B,
		id_t B_beats,
		tapa::istream<TapaReaderCommand>& commands,
		tapa::ostream<TapaBReaderRawPacket>& raw_packets) {
	ap_uint<512> dense_cache_data[kTapaDenseBReaderCacheBeats];
	id_t dense_cache_offset[kTapaDenseBReaderCacheRows];
	ap_uint<4> dense_cache_beats[kTapaDenseBReaderCacheRows];
	bool dense_cache_valid[kTapaDenseBReaderCacheRows];
#pragma HLS BIND_STORAGE variable=dense_cache_data type=ram_t2p impl=bram latency=2
#pragma HLS ARRAY_PARTITION variable=dense_cache_offset complete dim=1
#pragma HLS ARRAY_PARTITION variable=dense_cache_beats complete dim=1
#pragma HLS ARRAY_PARTITION variable=dense_cache_valid complete dim=1
	for (int entry = 0; entry < kTapaDenseBReaderCacheRows; ++entry) {
#pragma HLS UNROLL
		dense_cache_offset[entry] = 0;
		dense_cache_beats[entry] = 0;
		dense_cache_valid[entry] = false;
	}
	ap_uint<4> dense_cache_replacement = 0;
	bool done = false;
	while (!done) {
		const TapaReaderCommand command = commands.read();
		if (command[163]) {
			TapaBReaderRawPacket packet = 0;
			packet[552] = 1;
			raw_packets.write(packet);
			done = true;
		} else if (command[162]) {
			TapaBReaderRawPacket packet = 0;
			packet.range(547, 516) = command.range(159, 128);
			packet.range(549, 548) = command.range(161, 160);
			packet[551] = 1;
			packet[649] = command[228];
			raw_packets.write(packet);
		} else {
			const ap_uint<128> descriptor = command.range(127, 0);
			const id_t offset = descriptor.range(31, 0);
			const id_t length = descriptor.range(55, 32);
			const id_t beats = (length + 7) >> 3;
			const bool dense_mode
				= command.range(161, 160) == ADAPT_FP32_DENSE;
			const bool cacheable = dense_mode && beats != 0
				&& beats <= kTapaDenseBReaderCacheRowBeats;
			bool cache_hit = false;
			ap_uint<4> cache_hit_row = 0;
			for (int entry = 0; entry < kTapaDenseBReaderCacheRows; ++entry) {
#pragma HLS UNROLL
				if (cacheable && dense_cache_valid[entry]
						&& dense_cache_offset[entry] == offset
						&& dense_cache_beats[entry] == beats) {
					cache_hit = true;
					cache_hit_row = entry;
				}
			}
			const ap_uint<4> cache_fill_row = dense_cache_replacement;
			// A command keeps its selected cache row constant for the complete beat
			// loop.  Metadata is committed only after the loop, eliminating the old
			// tag store-to-load recurrence from this II=1 pipeline.
#pragma HLS DEPENDENCE variable=dense_cache_data inter false
			for (id_t beat = 0; beat < beats; ++beat) {
#pragma HLS PIPELINE II=1
				const id_t address = offset + beat;
				ap_uint<512> source = 0;
				if (address < B_beats) {
					if (cache_hit) {
						const id_t cache_index
							= ((id_t)cache_hit_row << 3) + beat;
						source = dense_cache_data[cache_index];
					} else {
						source = B[address];
						if (cacheable) {
							const id_t cache_index
								= ((id_t)cache_fill_row << 3) + beat;
							dense_cache_data[cache_index] = source;
						}
					}
				}
				const id_t remaining = length - (beat << 3);
				const ap_uint<4> valid = remaining > 8 ? 8 : remaining;
				TapaBReaderRawPacket packet = 0;
				packet.range(511, 0) = source;
				packet.range(515, 512) = valid;
				packet.range(547, 516) = command.range(159, 128);
				packet.range(549, 548) = command.range(161, 160);
				packet[550] = beat + 1 == beats
					|| (command.range(161, 160) == ADAPT_FP32_MERGE
						&& (beat + 1) % kTapaMergeRunPackets == 0);
				packet.range(584, 553) = descriptor.range(127, 96);
				packet.range(616, 585) = command.range(195, 164);
				packet.range(648, 617) = command.range(227, 196);
				packet[649] = command[228];
				raw_packets.write(packet);
			}
			if (cacheable && !cache_hit) {
				dense_cache_offset[cache_fill_row] = offset;
				dense_cache_beats[cache_fill_row] = beats;
				dense_cache_valid[cache_fill_row] = true;
				dense_cache_replacement = dense_cache_replacement + 1;
			}
		}
	}
}

// Fixed-width arithmetic stage.  Keeping the eight independent multiplies in
// their own II=1 loop prevents the fetch/cache controller from sharing them or
// stretching the HBM service loop.  Boundary/done tokens pass through without
// arithmetic and retain the deployed TapaProductPacket protocol.
void tapa_b_reader_scale(
		tapa::istream<TapaBReaderRawPacket>& raw_packets,
		tapa::ostream<TapaProductPacket>& products) {
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		// Do not make an empty input FIFO a stall for every stage of this
		// six-cycle FP32 pipeline.  A short row can end before enough packets
		// arrive to fill the pipeline; a blocking read then prevents its already
		// accepted boundary from draining, while the allocator waits for that
		// boundary before dispatching the next row.  Polling injects an idle
		// iteration instead, so in-flight products and boundaries keep advancing.
		TapaBReaderRawPacket raw;
		if (!raw_packets.try_read(raw)) continue;
		TapaProductPacket packet = 0;
		packet.range(552, 512) = raw.range(552, 512);
		packet[554] = raw[649];
		if (!raw[552] && !raw[551]) {
			const ap_uint<4> valid = raw.range(515, 512);
			const float scale = tapa_bits_to_float(raw.range(584, 553));
			const id_t dense_base = raw.range(616, 585);
			const id_t row_N = raw.range(648, 617);
			bool range_error = false;
			ap_uint<512> scaled = 0;
			for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
				if (lane < valid) {
					const ap_uint<64> item = raw.range(
						lane * 64 + 63, lane * 64);
					const id_t global_col = item.range(31, 0);
					const id_t col = global_col - dense_base;
					if (global_col < dense_base || col >= row_N)
						range_error = true;
					scaled.range(lane * 64 + 63, lane * 64)
						= tapa_pack_item(col,
							tapa_bits_to_float(item.range(63, 32)) * scale);
				}
			}
			packet.range(511, 0) = scaled;
			packet[553] = range_error;
		}
		products.write(packet);
		done = raw[552];
	}
}

// Register every local-MERGE/reducer boundary in its own physical task.  A
// deep inter-task FIFO is kept with the producing shard, while this one-entry
// elastic stage is placed beside the central reducer.  Besides preserving one
// packet/cycle, the explicit held register prevents reducer read/back-pressure
// logic from forming a combinational round trip through a remote shard and
// back into the same FIFO control RAM.  The payload is deliberately not reset:
// held is observed only when held_valid is true.
void tapa_product_register_slice(
		tapa::istream<TapaProductPacket>& input,
		tapa::ostream<TapaProductPacket>& output) {
	TapaProductPacket held;
	bool held_valid = false;
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=131072 avg=512
		bool wrote = false;
		if (held_valid) {
			wrote = output.try_write(held);
			if (wrote && held[552]) done = true;
		}

		// Simultaneously replace a consumed packet so an unstalled stream still
		// transfers one packet every cycle after the initial register latency.
		if (!done && (!held_valid || wrote)) {
			TapaProductPacket next;
			if (input.try_read(next)) {
				held = next;
				held_valid = true;
			} else if (wrote) {
				held_valid = false;
			}
		} else if (wrote) {
			held_valid = false;
		}
	}
}

static bool tapa_any_local(const bool active[kTapaLocalMergeWays]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=active complete
	return active[0] || active[1] || active[2] || active[3]
		|| active[4] || active[5] || active[6] || active[7];
}

// Keep the internal packet-to-item selector explicitly at the 48-bit local
// MERGE item width.  A
// dynamic slice of the complete packet is represented by Vitis HLS as a wide
// barrel shifter.  There is one selector per local merge way, so that
// form costs eight large shifters in every HBM shard.  Fixed slices followed
// by this 8:1 mux preserve the same II=1 access while limiting both the logic
// and the routed fanout to the item width.
static TapaLocalMergeItem tapa_select_packet_item(const TapaMergePacket packet,
		ap_uint<3> lane) {
#pragma HLS INLINE
	switch ((unsigned)lane) {
	case 0: return packet.range(47, 0);
	case 1: return packet.range(95, 48);
	case 2: return packet.range(143, 96);
	case 3: return packet.range(191, 144);
	case 4: return packet.range(239, 192);
	case 5: return packet.range(287, 240);
	case 6: return packet.range(335, 288);
	default: return packet.range(383, 336);
	}
}

static TapaMergePacket tapa_insert_packet_item(TapaMergePacket packet,
		ap_uint<3> lane, TapaLocalMergeItem item) {
#pragma HLS INLINE
	switch ((unsigned)lane) {
	case 0: packet.range(47, 0) = item; break;
	case 1: packet.range(95, 48) = item; break;
	case 2: packet.range(143, 96) = item; break;
	case 3: packet.range(191, 144) = item; break;
	case 4: packet.range(239, 192) = item; break;
	case 5: packet.range(287, 240) = item; break;
	case 6: packet.range(335, 288) = item; break;
	default: packet.range(383, 336) = item; break;
	}
	return packet;
}

static float tapa_sum_local(const float value[kTapaLocalMergeWays]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=value complete
	const float sum01 = value[0] + value[1];
	const float sum23 = value[2] + value[3];
	const float sum45 = value[4] + value[5];
	const float sum67 = value[6] + value[7];
	const float sum03 = sum01 + sum23;
	const float sum47 = sum45 + sum67;
	const float sum = sum03 + sum47;
	// Match the timing-safe FP32 tree used by the central MERGE and DENSE
	// reducers.  The extra operator stage is pipeline fill only: all seven
	// adders remain fully pipelined and the enclosing local merge stays II=1.
	// Without the explicit latency Vitis compresses min-key qualification and
	// the first fadd input stage into one 4.885-ns HLS path.
#pragma HLS BIND_OP variable=sum01 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum23 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum45 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum67 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum03 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum47 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum op=fadd impl=fulldsp latency=5
	return sum;
}

// Merge up to eight sorted runs.  Consecutive runs reside in distinct
// packet banks, allowing all heads to be read independently.  One unique
// column is produced per cycle; a base-8 hierarchy therefore needs at most
// three passes for the current 256-run bound instead of up to eight binary
// passes.  Zero cancellations remain explicit until the central global merge.
// The ping-pong stores use different implementations.  The role swaps after
// each pass, therefore keep that role compile-time-visible to HLS; otherwise
// it merges the formal arrays and loses the intended BRAM/URAM split.
template <bool kSourceIsBuffer0>
static id_t tapa_local_merge_emit(
		const TapaMergePacket source[kTapaLocalMergeWays]
			[kTapaMergePacketsPerBank],
		const TapaMergePacketIndex start_packet[kTapaLocalMergeWays],
		const TapaMergeItemCount length[kTapaLocalMergeWays], id_t run_count,
		TapaMergePacket destination[kTapaLocalMergeWays]
			[kTapaMergePacketsPerBank], ap_uint<4> destination_bank,
		id_t destination_start_packet, bool emit_stream, id_t row,
		ap_uint<2> mode, bool merge_heavy,
		tapa::ostream<TapaProductPacket>& output,
		id_t* destination_packets) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=source complete dim=1
#pragma HLS ARRAY_PARTITION variable=destination complete dim=1
#pragma HLS ARRAY_PARTITION variable=start_packet complete
#pragma HLS ARRAY_PARTITION variable=length complete
	id_t position[kTapaLocalMergeWays];
	bool active[kTapaLocalMergeWays];
	TapaMergePacket head_packet[kTapaLocalMergeWays];
	TapaMergePacket prefetched_packet[kTapaLocalMergeWays];
	ap_uint<16> head_col[kTapaLocalMergeWays];
	float head_value[kTapaLocalMergeWays];
#pragma HLS ARRAY_PARTITION variable=position complete
#pragma HLS ARRAY_PARTITION variable=active complete
#pragma HLS ARRAY_PARTITION variable=head_packet complete
#pragma HLS ARRAY_PARTITION variable=prefetched_packet complete
#pragma HLS ARRAY_PARTITION variable=head_col complete
#pragma HLS ARRAY_PARTITION variable=head_value complete
	for (int lane = 0; lane < kTapaLocalMergeWays; ++lane) {
#pragma HLS UNROLL
		position[lane] = 0;
		active[lane] = lane < run_count && length[lane] != 0;
		head_packet[lane] = active[lane]
			? source[lane][start_packet[lane]] : TapaMergePacket(0);
		// Do not reset the wide prefetch register.  It is committed only on
		// item 7, and every such commit is preceded by the item-6 read below.
		// Resetting all eight packet registers made one loop-init control net
		// drive 3072 data bits in each shard and was a dominant physical
		// high-fanout/congestion source.  The active/head state remains fully
		// initialized, so this changes neither protocol nor visible data.
		head_col[lane] = active[lane]
			? head_packet[lane].range(15, 0) : ~ap_uint<16>(0);
		head_value[lane] = active[lane]
			? tapa_bits_to_float(head_packet[lane].range(47, 16)) : 0.0f;
	}

	id_t output_length = 0;
	id_t output_packets = 0;
	TapaMergePacket pending = 0;
	ap_uint<4> pending_count = 0;
	TapaProductPacket held_packet = 0;
	bool held_valid = false;
	while (tapa_any_local(active)
			&& output_length < kTapaLocalItemsPerShard) {
#pragma HLS PIPELINE II=1
		ap_uint<16> candidate[kTapaLocalMergeWays];
		bool consume[kTapaLocalMergeWays];
		float contribution[kTapaLocalMergeWays];
#pragma HLS ARRAY_PARTITION variable=candidate complete
#pragma HLS ARRAY_PARTITION variable=consume complete
#pragma HLS ARRAY_PARTITION variable=contribution complete
		for (int lane = 0; lane < kTapaLocalMergeWays; ++lane) {
#pragma HLS UNROLL
			candidate[lane] = active[lane]
				? head_col[lane] : ~ap_uint<16>(0);
		}
		const ap_uint<16> min_key = tapa_min8_local16(candidate[0],
			candidate[1], candidate[2], candidate[3], candidate[4],
			candidate[5], candidate[6], candidate[7]);
		for (int lane = 0; lane < kTapaLocalMergeWays; ++lane) {
#pragma HLS UNROLL
			consume[lane] = active[lane] && head_col[lane] == min_key;
			contribution[lane] = consume[lane] ? head_value[lane] : 0.0f;
		}
		pending = tapa_insert_packet_item(pending, pending_count,
			tapa_pack_merge_item((ap_uint<16>)min_key,
				tapa_sum_local(contribution)));
		if (pending_count == 7) {
			if (emit_stream) {
				if (held_valid) output.write(held_packet);
				held_packet = 0;
				held_packet.range(511, 0) = tapa_expand_merge_packet(pending);
				held_packet.range(515, 512) = 8;
				held_packet.range(547, 516) = row;
				held_packet.range(549, 548) = mode;
				held_packet[554] = merge_heavy;
				held_valid = true;
			} else {
				destination[destination_bank]
					[destination_start_packet + output_packets] = pending;
				output_packets++;
			}
			pending = 0;
			pending_count = 0;
		} else {
			pending_count++;
		}
		output_length++;

		for (int lane = 0; lane < kTapaLocalMergeWays; ++lane) {
#pragma HLS UNROLL
			if (consume[lane]) {
				const id_t next = position[lane] + 1;
				position[lane] = next;
				if (next == length[lane]) {
					active[lane] = false;
					head_col[lane] = ~ap_uint<16>(0);
				} else {
					const ap_uint<3> next_lane = next & 7;
					// Always extract from the registered resident packet.  The
					// following packet is committed to head_packet only after item
					// 7 has been extracted below; consequently item 0 of the next
					// packet cannot see source_q0 through an HLS next-state bypass.
					const TapaMergePacket resident_packet = head_packet[lane];
					const TapaLocalMergeItem next_item =
#if defined(ADAPT_TAPA_LEGACY_DYNAMIC_PACKET_SLICE)
						resident_packet.range(
							(unsigned)next_lane * 48 + 47,
							(unsigned)next_lane * 48);
#else
						tapa_select_packet_item(resident_packet, next_lane);
#endif
					head_col[lane] = next_item.range(15, 0);
					head_value[lane]
						= tapa_bits_to_float(next_item.range(47, 16));
					// Start the following packet read while item 6 becomes the
					// registered head.  Item 7 supplies one complete capture
					// cycle before the following packet is selected at item 0.
					if (next_lane == 6 && next + 2 < length[lane]) {
						prefetched_packet[lane] = source[lane][
							start_packet[lane] + ((next + 2) >> 3)];
					}
					// At lane 7 the prefetched BRAM output may feed this register's
					// D input, but the current min8 recurrence still uses the old
					// resident packet.  The next iteration observes head_packet Q.
					if (next_lane == 7 && next + 1 < length[lane])
						head_packet[lane] = prefetched_packet[lane];
				}
			}
		}
	}
	if (pending_count != 0) {
		if (emit_stream) {
			if (held_valid) output.write(held_packet);
			held_packet = 0;
			held_packet.range(511, 0) = tapa_expand_merge_packet(pending);
			held_packet.range(515, 512) = pending_count;
			held_packet.range(547, 516) = row;
			held_packet.range(549, 548) = mode;
			held_packet[554] = merge_heavy;
			held_valid = true;
		} else {
			destination[destination_bank]
				[destination_start_packet + output_packets] = pending;
			output_packets++;
		}
	}
	if (emit_stream && held_valid) {
		held_packet[550] = 1;
		output.write(held_packet);
	}
	*destination_packets = output_packets;
	return output_length;
}

// One instance is placed after every HBM reader.  DENSE and DIRECT rows pass
// through unchanged; MERGE rows use a packet-banked 8-way hierarchy.  The
// eight instances execute concurrently, and DMSA's distributed-local idea
// remains an optimization inside the original MERGE path.
void tapa_shard_local_merge(tapa::istream<TapaProductPacket>& input,
		tapa::ostream<TapaProductPacket>& output) {
	TapaMergePacket merge_buffer0[kTapaLocalMergeWays]
		[kTapaMergePacketsPerBank];
	TapaMergePacket merge_buffer1[kTapaLocalMergeWays]
		[kTapaMergePacketsPerBank];
	TapaMergePacketIndex run_start[kTapaRunsPerShard];
	TapaMergeItemCount run_length[kTapaRunsPerShard];
	TapaMergePacketIndex next_run_start[kTapaRunsPerShard];
	TapaMergeItemCount next_run_length[kTapaRunsPerShard];
#pragma HLS ARRAY_PARTITION variable=merge_buffer0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=merge_buffer1 complete dim=1
#pragma HLS BIND_STORAGE variable=merge_buffer0 type=ram_2p impl=uram latency=2
#pragma HLS BIND_STORAGE variable=merge_buffer1 type=ram_2p impl=bram
// Buffer0 is URAM and buffer1 remains BRAM.  Both retain the full 8x256
// packet capacity and eight independently addressed banks.  The helper is
// role-specialized solely to preserve this physical split in HLS.
// The four run-descriptor arrays are small and accessed only between merge
// passes.  Moving them to URAM releases eight BRAM18s per shard-local merger
// without changing packet-bank capacity or its eight-way II=1 datapath.
#pragma HLS BIND_STORAGE variable=run_start type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=run_length type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=next_run_start type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=next_run_length type=ram_2p impl=uram

	id_t deferred_dense_row = 0;
	id_t deferred_dense_products = 0;
	bool deferred_dense_valid = false;
	bool done = false;
	while (!done) {
		TapaProductPacket packet = input.read();
		if (packet[552]) {
			output.write(packet);
			done = true;
			break;
		}
		const id_t row = packet.range(547, 516);
		const ap_uint<2> mode = packet.range(549, 548);
		const bool merge_heavy = packet[554];
		id_t source_products = 0;
		if (mode == ADAPT_FP32_DENSE) {
			// DENSE row pairs can be interleaved at task granularity.  Boundaries
			// remain ordered row0,row1, so retain the second row's count across the
			// first boundary instead of materializing either row locally.
			const id_t pair_base = row & ~id_t(1);
			id_t pair_products[2] = {0, 0};
#pragma HLS ARRAY_PARTITION variable=pair_products complete
			if (deferred_dense_valid
					&& (deferred_dense_row & ~id_t(1)) == pair_base) {
				pair_products[deferred_dense_row & 1] = deferred_dense_products;
				deferred_dense_valid = false;
			}
			bool row_done = false;
			while (!row_done) {
#pragma HLS PIPELINE II=1
				if (packet[552]) {
					packet[553] = 1;
					packet[551] = 1;
					row_done = true;
				} else {
					const id_t packet_row = packet.range(547, 516);
					const bool row_valid
						= (packet_row & ~id_t(1)) == pair_base;
					if (!row_valid || packet.range(549, 548) != mode)
						packet[553] = 1;
					const ap_uint<1> context = packet_row & 1;
					if (packet[551]) {
						packet.range(31, 0) = row_valid
							? pair_products[(unsigned)context] : (id_t)0;
						const ap_uint<1> other = context ^ 1;
						if (pair_products[(unsigned)other] != 0) {
							deferred_dense_row = pair_base + other;
							deferred_dense_products
								= pair_products[(unsigned)other];
							deferred_dense_valid = true;
						}
						row_done = true;
					} else if (row_valid) {
						pair_products[(unsigned)context]
							+= packet.range(515, 512);
					}
				}
				output.write(packet);
				if (!row_done) packet = input.read();
			}
			continue;
		} else if (mode != ADAPT_FP32_MERGE) {
			bool row_done = false;
			while (!row_done) {
#pragma HLS PIPELINE II=1
				if (packet[552]) {
					packet[553] = 1;
					packet[551] = 1;
					row_done = true;
				} else if (packet.range(547, 516) != row
						|| packet.range(549, 548) != mode) {
					packet[553] = 1;
				}
				if (packet[551]) {
					packet.range(31, 0) = source_products;
					row_done = true;
				} else {
					source_products += packet.range(515, 512);
				}
				output.write(packet);
				if (!row_done) packet = input.read();
			}
			continue;
		}

		id_t bank_packet_offset[kTapaLocalMergeWays];
#pragma HLS ARRAY_PARTITION variable=bank_packet_offset complete
		for (int bank = 0; bank < kTapaLocalMergeWays; ++bank) {
#pragma HLS UNROLL
			bank_packet_offset[bank] = 0;
		}
		id_t run_count = 0;
		// One register drives each packet bank directly.  The old binary
		// run_count[2:0] decoder crossed the shard and fed every BRAM write-enable;
		// it was the worst routed path (5.811 ns data delay, 85.6% routing).
		ap_uint<kTapaLocalMergeWays> run_bank_onehot = 1;
		// A shard stores at most 16384 products, so the open-run recurrence needs
		// 15 bits.  Keeping this at id_t made every packet pay a 32-bit carry
		// chain before writing the run descriptor.
		ap_uint<15> current_run_length = 0;
		bool run_open = false;
		bool local_error = false;
		bool row_done = false;
		while (!row_done) {
#pragma HLS PIPELINE II=1
			if (packet[552]) {
				local_error = true;
				row_done = true;
			} else if (packet[551]) {
				// Reader/dispatcher streams are row-serialized by construction.  Check
				// the 32-bit row tag once at the boundary instead of placing that wide
				// equality comparator on every data beat's run-length write path.
				if (run_open || packet.range(547, 516) != row
						|| packet.range(549, 548) != mode
						|| packet[554] != merge_heavy)
					local_error = true;
				row_done = true;
			} else {
				if (packet[553] || packet[554] != merge_heavy)
					local_error = true;
				const id_t valid = packet.range(515, 512);
				source_products += valid;
				if (!run_open) {
					if (run_count >= kTapaRunsPerShard) {
						local_error = true;
					} else {
						id_t selected_offset = 0;
						for (int bank = 0; bank < kTapaLocalMergeWays;
								++bank) {
#pragma HLS UNROLL
							if (run_bank_onehot[bank])
								selected_offset = bank_packet_offset[bank];
						}
						run_start[run_count] = selected_offset;
						current_run_length = 0;
						run_open = true;
					}
				}
				if (run_open) {
					ap_uint<kTapaLocalMergeWays> bank_full_mask = 0;
					for (int bank = 0; bank < kTapaLocalMergeWays; ++bank) {
#pragma HLS UNROLL
						const bool selected = run_bank_onehot[bank];
						const bool bank_full = bank_packet_offset[bank]
							>= kTapaMergePacketsPerBank;
						bank_full_mask[bank] = selected && bank_full;
						if (selected && !bank_full) {
							merge_buffer0[bank][bank_packet_offset[bank]]
								= tapa_compress_merge_packet(packet.range(511, 0));
							bank_packet_offset[bank]++;
						}
					}
					// Update the recurrence only once per packet.  Placing this add in
					// every unrolled bank branch made HLS build eight cascaded
					// conditional adders (10.8 ns) even though the selector is one-hot.
					if (bank_full_mask != 0)
						local_error = true;
					else
						current_run_length += valid;
					if (packet[550]) {
						run_length[run_count++] = current_run_length;
						run_bank_onehot
							= (run_bank_onehot << 1)
								| (run_bank_onehot
									>> (kTapaLocalMergeWays - 1));
						run_open = false;
					}
				}
			}
			if (!row_done) packet = input.read();
		}

		if (!local_error) {
			id_t current_run_count = run_count;
			bool source_is_buffer0 = true;
			while (current_run_count > kTapaLocalMergeWays
					&& !local_error) {
				id_t destination_packet_offset[kTapaLocalMergeWays];
#pragma HLS ARRAY_PARTITION variable=destination_packet_offset complete
				for (int bank = 0; bank < kTapaLocalMergeWays; ++bank) {
#pragma HLS UNROLL
					destination_packet_offset[bank] = 0;
				}
				id_t next_count = 0;
				for (id_t run = 0; run < current_run_count;) {
					TapaMergePacketIndex group_start[kTapaLocalMergeWays];
					TapaMergeItemCount group_length[kTapaLocalMergeWays];
#pragma HLS ARRAY_PARTITION variable=group_start complete
#pragma HLS ARRAY_PARTITION variable=group_length complete
					id_t group_items = 0;
					id_t group_runs = 0;
					bool group_accepting = true;
					for (int lane = 0; lane < kTapaLocalMergeWays; ++lane) {
#pragma HLS PIPELINE II=1
						const id_t index = run + lane;
						const id_t candidate_length
							= index < current_run_count
								? (id_t)run_length[index] : (id_t)0;
						// Greedily stop before an intermediate run would exceed one
						// packet bank.  Besides fixing c-52's last hierarchy level,
						// this keeps the decision local to the eight descriptors that
						// are already read by this merge group.
						const bool valid_run = group_accepting
							&& index < current_run_count
							&& group_items + candidate_length
								<= kTapaMergePacketsPerBank * 8;
						if (index < current_run_count && !valid_run)
							group_accepting = false;
						// Runs are packet-banked by their global descriptor index, not
						// by their position inside this capacity-limited group.  A
						// previous group may contain fewer than eight runs, so `run`
						// need not be a multiple of eight.  Rotate the descriptor back
						// to its physical source bank; otherwise every later group reads
						// the wrong BRAM bank (c-52's 1,714-run final row exposed this).
						const ap_uint<3> source_bank
							= (ap_uint<3>)(run + lane);
						group_start[(unsigned)source_bank]
							= valid_run ? run_start[index]
								: TapaMergePacketIndex(0);
						group_length[(unsigned)source_bank]
							= valid_run ? candidate_length : 0;
						if (valid_run) {
							group_items += candidate_length;
							group_runs++;
						}
					}
					if (group_runs == 0) {
						local_error = true;
						break;
					}
					const ap_uint<4> destination_bank
						= next_count & (kTapaLocalMergeWays - 1);
					const id_t destination_start
						= destination_packet_offset[destination_bank];
					const id_t maximum_packets = (group_items + 7) >> 3;
					if (destination_start + maximum_packets
							> kTapaMergePacketsPerBank) {
						local_error = true;
						break;
					}
					id_t merged_packets = 0;
					const id_t merged_length = source_is_buffer0
						? tapa_local_merge_emit<true>(merge_buffer0, group_start,
							group_length, kTapaLocalMergeWays, merge_buffer1,
							destination_bank, destination_start, false, row, mode,
							merge_heavy, output, &merged_packets)
						: tapa_local_merge_emit<false>(merge_buffer1, group_start,
							group_length, kTapaLocalMergeWays, merge_buffer0,
							destination_bank, destination_start, false, row, mode,
							merge_heavy, output, &merged_packets);
					next_run_start[next_count] = destination_start;
					next_run_length[next_count] = merged_length;
					destination_packet_offset[destination_bank]
						+= merged_packets;
					next_count++;
					run += group_runs;
				}
				if (!local_error) {
					for (id_t run = 0; run < next_count; ++run) {
#pragma HLS PIPELINE II=1
						run_start[run] = next_run_start[run];
						run_length[run] = next_run_length[run];
					}
					current_run_count = next_count;
					source_is_buffer0 = !source_is_buffer0;
				}
			}

			if (!local_error && current_run_count == 1) {
				const id_t final_length = run_length[0];
				const id_t final_packets = (final_length + 7) >> 3;
				for (id_t packet_index = 0; packet_index < final_packets;
						++packet_index) {
#pragma HLS PIPELINE II=1
					TapaProductPacket result = 0;
					const TapaMergePacket compact = source_is_buffer0
						? merge_buffer0[0][run_start[0] + packet_index]
						: merge_buffer1[0][run_start[0] + packet_index];
					result.range(511, 0) = tapa_expand_merge_packet(compact);
					const id_t remaining = final_length - (packet_index << 3);
					result.range(515, 512) = remaining > 8 ? 8 : remaining;
					result.range(547, 516) = row;
					result.range(549, 548) = mode;
					result[550] = packet_index + 1 == final_packets;
					result[554] = merge_heavy;
					output.write(result);
				}
			} else if (!local_error && current_run_count > 1) {
				// Keep the run tables in small RAMs.  Only the final
				// eight descriptors need parallel access by the merge network;
				// staging them here avoids a 256-way fully-partitioned mux in the
				// local task while preserving the same merge throughput.
				TapaMergePacketIndex final_start[kTapaLocalMergeWays];
				TapaMergeItemCount final_length[kTapaLocalMergeWays];
#pragma HLS ARRAY_PARTITION variable=final_start complete
#pragma HLS ARRAY_PARTITION variable=final_length complete
				for (int lane = 0; lane < kTapaLocalMergeWays; ++lane) {
#pragma HLS PIPELINE II=1
					const bool valid_run = lane < current_run_count;
					final_start[lane] = valid_run ? run_start[lane]
						: TapaMergePacketIndex(0);
					final_length[lane] = valid_run ? run_length[lane]
						: TapaMergeItemCount(0);
				}
				id_t ignored_packets = 0;
				if (source_is_buffer0)
					tapa_local_merge_emit<true>(merge_buffer0, final_start, final_length,
						current_run_count, merge_buffer1, 0, 0, true, row, mode,
						merge_heavy, output, &ignored_packets);
				else
					tapa_local_merge_emit<false>(merge_buffer1, final_start, final_length,
						current_run_count, merge_buffer0, 0, 0, true, row, mode,
						merge_heavy, output, &ignored_packets);
			}
		}

		TapaProductPacket boundary = 0;
		boundary.range(31, 0) = source_products;
		boundary.range(547, 516) = row;
		boundary.range(549, 548) = mode;
		boundary[551] = 1;
		boundary[553] = local_error;
		boundary[554] = merge_heavy;
		output.write(boundary);
	}
}

template <int SHARD>
static void tapa_accept_packet(tapa::istream<TapaProductPacket>& input,
		id_t expected_row, ap_uint<2> expected_mode,
		bool row_ended[kTapaNumShards],
		TapaProductPacket packet_store[kTapaNumShards][kTapaPacketsPerShard],
		id_t packet_count[kTapaNumShards],
		id_t source_product_count[kTapaNumShards],
		bool shard_overflow[kTapaNumShards],
		id_t shard_protocol_errors[kTapaNumShards]) {
#pragma HLS INLINE
	TapaProductPacket packet;
	if (!row_ended[SHARD] && input.try_read(packet)) {
		if (packet[552]) {
			shard_protocol_errors[SHARD]++;
			row_ended[SHARD] = true;
		} else if (packet.range(547, 516) != expected_row
				|| packet.range(549, 548) != expected_mode) {
			shard_protocol_errors[SHARD]++;
		} else if (packet[551]) {
			source_product_count[SHARD] = packet.range(31, 0);
			if (packet[553]) shard_overflow[SHARD] = true;
			row_ended[SHARD] = true;
		} else {
			if (packet_count[SHARD] >= kTapaPacketsPerShard) {
				shard_overflow[SHARD] = true;
			} else {
				packet_store[SHARD][packet_count[SHARD]++] = packet;
			}
		}
	}
}

template <int SOURCE>
static void tapa_merge_stream_refill(
		tapa::istream<TapaProductPacket>& input, id_t expected_row,
		ap_uint<2> expected_mode,
		ap_uint<512> source_payload[kTapaNumShards],
		ap_uint<4> source_remaining[kTapaNumShards],
		bool source_ended[kTapaNumShards],
		id_t source_reported_products[kTapaNumShards],
		id_t source_protocol_errors[kTapaNumShards],
		bool source_overflow[kTapaNumShards]) {
#pragma HLS INLINE
	if (source_remaining[SOURCE] != 0 || source_ended[SOURCE]) return;
	TapaProductPacket packet;
	if (!input.try_read(packet)) return;
	if (packet[552]) {
		source_protocol_errors[SOURCE]++;
		source_ended[SOURCE] = true;
		return;
	}
	if (packet.range(547, 516) != expected_row
			|| packet.range(549, 548) != expected_mode)
		source_protocol_errors[SOURCE]++;
	if (packet[551]) {
		source_reported_products[SOURCE] = packet.range(31, 0);
		source_overflow[SOURCE] = packet[553];
		source_ended[SOURCE] = true;
		return;
	}
	const ap_uint<4> valid = packet.range(515, 512);
	if (valid == 0 || valid > 8) {
		source_protocol_errors[SOURCE]++;
		return;
	}
	source_payload[SOURCE] = packet.range(511, 0);
	source_remaining[SOURCE] = valid;
}

// The shard-local MERGE tasks already produce one sorted, duplicate-free run
// per shard.  Consume those eight runs directly instead of first materializing
// them in the central packet_store.  Refilling packet heads and the global
// eight-way reduction share one II=1 loop, so local output/next-row work can
// overlap the current row's global merge through the inter-task FIFOs.
static void tapa_merge_stream_select(
		tapa::istream<TapaProductPacket>& product0,
		tapa::istream<TapaProductPacket>& product1,
		tapa::istream<TapaProductPacket>& product2,
		tapa::istream<TapaProductPacket>& product3,
		tapa::istream<TapaProductPacket>& product4,
		tapa::istream<TapaProductPacket>& product5,
		tapa::istream<TapaProductPacket>& product6,
		tapa::istream<TapaProductPacket>& product7,
		id_t expected_row, ap_uint<2> expected_mode,
		id_t expected_products,
		hls::stream<TapaMergeContributions>& selected_items,
		id_t* stream_cycles, id_t* reported_products,
		id_t* stream_protocol_errors, bool* stream_overflow) {
#pragma HLS INLINE off
	ap_uint<512> source_payload[kTapaNumShards];
	ap_uint<4> source_remaining[kTapaNumShards];
	bool source_ended[kTapaNumShards];
	id_t source_reported_products[kTapaNumShards];
	id_t source_protocol_errors[kTapaNumShards];
	bool source_overflow[kTapaNumShards];
#pragma HLS ARRAY_PARTITION variable=source_payload complete dim=1
#pragma HLS ARRAY_PARTITION variable=source_remaining complete dim=1
#pragma HLS ARRAY_PARTITION variable=source_ended complete dim=1
#pragma HLS ARRAY_PARTITION variable=source_reported_products complete dim=1
#pragma HLS ARRAY_PARTITION variable=source_protocol_errors complete dim=1
#pragma HLS ARRAY_PARTITION variable=source_overflow complete dim=1
	for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
		// Payload is observed only while source_remaining is nonzero.  Leaving
		// the wide data register don't-care here removes 4096 unnecessary
		// loop-init sinks; the validity, accounting and error state below still
		// receive deterministic per-row initialization.
		source_remaining[source] = 0;
		source_ended[source] = false;
		source_reported_products[source] = 0;
		source_protocol_errors[source] = 0;
		source_overflow[source] = false;
	}

	id_t cycles = 0;
	while (!(source_ended[0] && source_ended[1] && source_ended[2]
			&& source_ended[3] && source_ended[4] && source_ended[5]
			&& source_ended[6] && source_ended[7])) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=65536 avg=512
		cycles++;
		tapa_merge_stream_refill<0>(product0, expected_row, expected_mode,
			source_payload, source_remaining, source_ended,
			source_reported_products, source_protocol_errors, source_overflow);
		tapa_merge_stream_refill<1>(product1, expected_row, expected_mode,
			source_payload, source_remaining, source_ended,
			source_reported_products, source_protocol_errors, source_overflow);
		tapa_merge_stream_refill<2>(product2, expected_row, expected_mode,
			source_payload, source_remaining, source_ended,
			source_reported_products, source_protocol_errors, source_overflow);
		tapa_merge_stream_refill<3>(product3, expected_row, expected_mode,
			source_payload, source_remaining, source_ended,
			source_reported_products, source_protocol_errors, source_overflow);
		tapa_merge_stream_refill<4>(product4, expected_row, expected_mode,
			source_payload, source_remaining, source_ended,
			source_reported_products, source_protocol_errors, source_overflow);
		tapa_merge_stream_refill<5>(product5, expected_row, expected_mode,
			source_payload, source_remaining, source_ended,
			source_reported_products, source_protocol_errors, source_overflow);
		tapa_merge_stream_refill<6>(product6, expected_row, expected_mode,
			source_payload, source_remaining, source_ended,
			source_reported_products, source_protocol_errors, source_overflow);
		tapa_merge_stream_refill<7>(product7, expected_row, expected_mode,
			source_payload, source_remaining, source_ended,
			source_reported_products, source_protocol_errors, source_overflow);

		const bool ready = (source_remaining[0] != 0 || source_ended[0])
			&& (source_remaining[1] != 0 || source_ended[1])
			&& (source_remaining[2] != 0 || source_ended[2])
			&& (source_remaining[3] != 0 || source_ended[3])
			&& (source_remaining[4] != 0 || source_ended[4])
			&& (source_remaining[5] != 0 || source_ended[5])
			&& (source_remaining[6] != 0 || source_ended[6])
			&& (source_remaining[7] != 0 || source_ended[7]);
		const bool any_active = source_remaining[0] != 0
			|| source_remaining[1] != 0 || source_remaining[2] != 0
			|| source_remaining[3] != 0 || source_remaining[4] != 0
			|| source_remaining[5] != 0 || source_remaining[6] != 0
			|| source_remaining[7] != 0;
		if (ready && any_active) {
			ap_uint<16> candidate[kTapaNumShards];
			bool consume[kTapaNumShards];
#pragma HLS ARRAY_PARTITION variable=candidate complete
#pragma HLS ARRAY_PARTITION variable=consume complete
			for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
				candidate[source] = source_remaining[source] != 0
					? source_payload[source].range(15, 0)
					: ~ap_uint<16>(0);
			}
			const ap_uint<16> min_col = tapa_min8_local16(
				candidate[0], candidate[1],
				candidate[2], candidate[3], candidate[4], candidate[5],
				candidate[6], candidate[7]);
			for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
				consume[source] = source_remaining[source] != 0
					&& source_payload[source].range(15, 0) == min_col;
			}
			TapaMergeContributions token = 0;
			token.range(31, 0) = (ap_uint<32>)min_col;
			for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
				token.range(32 + source * 32 + 31, 32 + source * 32)
					= consume[source]
					? source_payload[source].range(63, 32)
					: (ap_uint<32>)0;
				if (consume[source]) {
					source_payload[source] >>= 64;
					source_remaining[source]--;
				}
			}
			selected_items.write(token);
		}
	}
	TapaMergeContributions done = 0;
	done[288] = 1;
	selected_items.write(done);

	const id_t products01
		= source_reported_products[0] + source_reported_products[1];
	const id_t products23
		= source_reported_products[2] + source_reported_products[3];
	const id_t products45
		= source_reported_products[4] + source_reported_products[5];
	const id_t products67
		= source_reported_products[6] + source_reported_products[7];
	const id_t products = (products01 + products23) + (products45 + products67);
	const id_t errors01
		= source_protocol_errors[0] + source_protocol_errors[1];
	const id_t errors23
		= source_protocol_errors[2] + source_protocol_errors[3];
	const id_t errors45
		= source_protocol_errors[4] + source_protocol_errors[5];
	const id_t errors67
		= source_protocol_errors[6] + source_protocol_errors[7];
	*stream_cycles = cycles;
	*reported_products = products;
	*stream_protocol_errors = (errors01 + errors23) + (errors45 + errors67)
		+ (products != expected_products ? 1 : 0);
	*stream_overflow = source_overflow[0]
		|| source_overflow[1] || source_overflow[2] || source_overflow[3]
		|| source_overflow[4] || source_overflow[5] || source_overflow[6]
		|| source_overflow[7];
}

static void tapa_merge_stream_sum(
		hls::stream<TapaMergeContributions>& selected_items, id_t N,
		id_t column_base,
		hls::stream<TapaMergeItem>& merge_items) {
#pragma HLS INLINE off
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=65537 avg=512
		const TapaMergeContributions token = selected_items.read();
		if (token[288]) {
			done = true;
		} else {
			float contribution[kTapaNumShards];
#pragma HLS ARRAY_PARTITION variable=contribution complete
			for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
				contribution[source] = tapa_bits_to_float(token.range(
					32 + source * 32 + 31, 32 + source * 32));
			}
			const float sum01 = contribution[0] + contribution[1];
			const float sum23 = contribution[2] + contribution[3];
			const float sum45 = contribution[4] + contribution[5];
			const float sum67 = contribution[6] + contribution[7];
			const float sum03 = sum01 + sum23;
			const float sum47 = sum45 + sum67;
			const float sum = sum03 + sum47;
#pragma HLS BIND_OP variable=sum01 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum23 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum45 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum67 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum03 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum47 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum op=fadd impl=fulldsp latency=5
			const ap_uint<32> col = token.range(31, 0);
			if (col < N && tapa_nonzero(sum)) {
				TapaMergeItem item = 0;
				item.range(63, 0) = tapa_pack_item(column_base + col, sum);
				merge_items.write(item);
			}
		}
	}
	TapaMergeItem output_done = 0;
	output_done[64] = 1;
	merge_items.write(output_done);
}

static void tapa_merge_stream_store(
		hls::stream<TapaMergeItem>& merge_items,
		ap_uint<64> row_output[8][kTapaDenseCapacity / 8],
		bool* output_overflow, id_t* result_row_nnz) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=row_output complete dim=1
	id_t row_nnz = 0;
	bool overflow = false;
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=65537 avg=512
		const TapaMergeItem token = merge_items.read();
		if (token[64]) {
			done = true;
		} else if (row_nnz < kTapaDenseCapacity) {
			tapa_store_row_item(row_output, row_nnz, token.range(63, 0));
			row_nnz++;
		} else {
			overflow = true;
		}
	}
	*output_overflow = overflow;
	*result_row_nnz = row_nnz;
}

static void tapa_merge_stream_reduce(
		tapa::istream<TapaProductPacket>& product0,
		tapa::istream<TapaProductPacket>& product1,
		tapa::istream<TapaProductPacket>& product2,
		tapa::istream<TapaProductPacket>& product3,
		tapa::istream<TapaProductPacket>& product4,
		tapa::istream<TapaProductPacket>& product5,
		tapa::istream<TapaProductPacket>& product6,
		tapa::istream<TapaProductPacket>& product7,
		id_t expected_row, ap_uint<2> expected_mode,
		id_t expected_products, id_t N, id_t column_base,
		ap_uint<64> row_output[8][kTapaDenseCapacity / 8],
		id_t* stream_cycles, id_t* reported_products,
		id_t* stream_protocol_errors, bool* stream_overflow,
		id_t* result_row_nnz) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
	hls::stream<TapaMergeItem> merge_items("merge_items");
	hls::stream<TapaMergeContributions> selected_items("selected_items");
#pragma HLS STREAM variable=selected_items depth=64
#pragma HLS STREAM variable=merge_items depth=64
	bool source_overflow = false;
	bool output_overflow = false;
	tapa_merge_stream_select(product0, product1, product2, product3,
		product4, product5, product6, product7, expected_row, expected_mode,
		expected_products, selected_items, stream_cycles, reported_products,
		stream_protocol_errors, &source_overflow);
	tapa_merge_stream_sum(selected_items, N, column_base, merge_items);
	tapa_merge_stream_store(merge_items, row_output, &output_overflow,
		result_row_nnz);
	*stream_overflow = source_overflow || output_overflow;
}

// Consume one row from the persistent eight-root MERGE fabric.  The fabric
// has already reduced duplicate columns across all shard-local streams and
// may supply up to eight consecutive local columns per token for a heavy row.
// Restore the 32-bit row-local base only here, after all 16-bit comparisons.
static void tapa_merge_fabric_store(
		tapa::istream<TapaMergeResultPacket>& merge_results,
		id_t expected_row, id_t expected_products, id_t N, id_t column_base,
		ap_uint<64> row_output[8][kTapaDenseCapacity / 8],
		id_t* stream_cycles, id_t* reported_products,
		id_t* stream_protocol_errors, bool* stream_overflow,
		id_t* result_row_nnz) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=row_output complete dim=1
	id_t cycles = 0;
	id_t row_nnz = 0;
	id_t products = 0;
	id_t errors = 0;
	bool overflow = false;
	bool row_done = false;
	while (!row_done) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=65537 avg=512
		const TapaMergeResultPacket packet = merge_results.read();
		cycles++;
		if (packet[549]) {
			errors++;
			row_done = true;
			continue;
		}
		if ((id_t)packet.range(547, 516) != expected_row) errors++;
		if (packet[548]) {
			products = packet.range(584, 553);
			errors += packet.range(551, 550);
			row_done = true;
			continue;
		}
		const ap_uint<4> valid = packet.range(515, 512);
		if (valid == 0 || valid > 8) {
			errors++;
			continue;
		}
		ap_uint<512> restored_items = 0;
		ap_uint<8> write_mask = 0;
		ap_uint<8> column_error_mask = 0;
		ap_uint<8> overflow_mask = 0;
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			if (lane < valid) {
				const ap_uint<16> local_col
					= packet.range(lane * 64 + 15, lane * 64);
				const ap_uint<32> value
					= packet.range(lane * 64 + 63, lane * 64 + 32);
				if (local_col >= N) {
					column_error_mask[lane] = 1;
				} else if (row_nnz + lane < kTapaDenseCapacity) {
					ap_uint<64> item = 0;
					item.range(31, 0) = column_base + local_col;
					item.range(63, 32) = value;
					restored_items.range(lane * 64 + 63, lane * 64) = item;
					write_mask[lane] = 1;
				} else {
					overflow_mask[lane] = 1;
				}
			}
		}
		// Do not let eight unrolled `errors++` operations become one serial
		// eight-adder recurrence.  The balanced popcount keeps the II=1 packet
		// writer below the 160 MHz combinational budget.
		errors += tapa_popcount8(column_error_mask);
		overflow |= overflow_mask != 0;
		tapa_store_row_packet(row_output, row_nnz, restored_items, write_mask);
		row_nnz += valid;
	}
	if (products != expected_products) errors++;
	*stream_cycles = cycles;
	*reported_products = products;
	*stream_protocol_errors = errors;
	*stream_overflow = overflow;
	*result_row_nnz = row_nnz > kTapaDenseCapacity
		? (id_t)kTapaDenseCapacity : row_nnz;
}

// Scan the epoch-tagged bitmap words and stream globally sorted columns.
// The terminating token lets the consumer remain one persistent flat pipeline
// for the complete row instead of restarting at every active address group.
template <int COPY>
static ap_uint<kTapaDenseEpochBits> tapa_dense_capture_extract_epoch(
		ap_uint<kTapaDenseEpochBits> epoch, bool route_complete) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#pragma HLS LATENCY min=1 max=1
	// Keep two consumer-local copies of the epoch at the route/extract boundary.
	// route_complete is deliberately a real data dependency on the completed
	// route/accumulate dataflow.  Without it, HLS 2022.2 launched extraction in
	// parallel with accumulation and the first DENSE group read epoch-zero RAM.
	// A direct scalar connection made the bank-reducer epoch register drive both
	// wide extract trees across SLR1; routed paths were about 90% interconnect
	// delay.  The template parameter forces two distinct one-cycle RTL stages so
	// candidate generation and accumulator extraction do not share that fanout.
	return route_complete ? epoch : (ap_uint<kTapaDenseEpochBits>)0;
}

template <int COPY>
static id_t tapa_dense_pipeline_sum8(id_t value0, id_t value1,
		id_t value2, id_t value3, id_t value4, id_t value5,
		id_t value6, id_t value7) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#pragma HLS LATENCY min=3 max=3
	// This helper is called only after a row has left the II=1 bank router.  Its
	// explicit three-cycle boundary keeps the balanced 8-to-1 counter reduction
	// out of both the route-loop exit and the parent scalar-result register.
	const id_t sum01 = value0 + value1;
	const id_t sum23 = value2 + value3;
	const id_t sum45 = value4 + value5;
	const id_t sum67 = value6 + value7;
	const id_t sum03 = sum01 + sum23;
	const id_t sum47 = sum45 + sum67;
	return sum03 + sum47;
}

static void tapa_dense_generate_candidates(
		const ap_uint<64> leaf_bits[kTapaDenseBanks][kTapaDensePhases]
			[kTapaDenseLeafWords],
		const ap_uint<kTapaDenseEpochBits>
			leaf_word_tag[kTapaDenseBanks][kTapaDensePhases]
				[kTapaDenseLeafWords],
		ap_uint<kTapaDenseEpochBits> epoch, id_t N,
		ap_uint<16> virtual_base,
		hls::stream<TapaDenseCandidateGroup>& candidate_stream,
		id_t* bitmap_word_scans, id_t* nonempty_words,
		id_t* active_address_groups, id_t* candidate_columns) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=leaf_bits complete dim=1
#pragma HLS ARRAY_PARTITION variable=leaf_bits complete dim=2
#pragma HLS ARRAY_PARTITION variable=leaf_word_tag complete dim=1
#pragma HLS ARRAY_PARTITION variable=leaf_word_tag complete dim=2
	id_t local_word_scans = 0;
	id_t local_nonempty_words = 0;
	id_t local_active_groups = 0;
	id_t local_candidates = 0;
	// The former route loop maintained an 8x11x128 phase-summary register and
	// dynamically set one bit on every accumulator update.  That feedback was
	// the routed critical path.  Epoch-tagged leaf words already contain the
	// same information, so scan only the 0..63 word range belonging to this row
	// context here, outside the accumulation recurrence.
	const id_t row_word_count = (N + 511) >> 9;
	const ap_uint<7> first_word = virtual_base >> 9;
	dense_candidate_words:
	for (id_t local_word = 0; local_word < row_word_count; ++local_word) {
#pragma HLS LOOP_FLATTEN off
		local_word_scans++;
		const ap_uint<7> word = first_word + local_word;
		ap_uint<64> bank_word[kTapaDenseBanks];
#pragma HLS ARRAY_PARTITION variable=bank_word complete dim=1
		ap_uint<64> active_addresses = 0;
		for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
			ap_uint<64> phase_union = 0;
			for (int phase = 0; phase < kTapaDensePhases; ++phase) {
#pragma HLS UNROLL
				if (leaf_word_tag[bank][phase][word] == epoch)
					phase_union |= leaf_bits[bank][phase][word];
			}
			bank_word[bank] = phase_union;
			active_addresses |= bank_word[bank];
		}
		if (active_addresses != 0) local_nonempty_words++;
		// Keep address traversal in one explicitly pipelined state machine.
		// With a pipelined inner loop nested below the address while,
		// Vitis HLS 2022.2 statically expanded all 64 possible address iterations:
		// 4,608 64-bit OR expressions (about 150k LUTs) in this function alone.
		// The three setup stages below are intentional.  Byte selection, the
		// byte-local priority encoder, and the bank permutation are separated by
		// registers.  Keeping the whole 64-bit first-set operation in one stage
		// produced a 4.904 ns HLS path, above the 4.5625 ns effective budget for
		// 160 MHz.  Unlike the former serial extractor, one token now preserves
		// all eight low-column positions of the selected address.
		ap_uint<13> address = 0;
		ap_uint<64> pending_addresses = 0;
		ap_uint<3> pending_byte = 0;
		ap_uint<6> pending_bit = 0;
		bool bit_pending = false;
		bool address_pending = false;
		dense_candidate_addresses:
		while (active_addresses != 0 || bit_pending || address_pending) {
#pragma HLS PIPELINE II=1
			if (address_pending) {
				// Materialize the bank permutation one cycle after priority
				// encoding.  Without this boundary, the first-set encoder, dynamic
				// bank selection and writeback formed a 6.358 ns HLS path.
				ap_uint<8> next_low = 0;
				for (int low = 0; low < 8; ++low) {
#pragma HLS UNROLL
					const id_t virtual_col = ((id_t)address << 3) | low;
					const id_t col = virtual_col - virtual_base;
					const ap_uint<3> bank = tapa_dense_bank(col);
					next_low[low] = col < N
						&& bank_word[(unsigned)bank][pending_bit];
				}
				// A row contributes at most N distinct columns and the Host
				// contract keeps N <= kTapaDenseCapacity.  The former per-lane
				// capacity guard therefore could never reject a legal candidate,
				// but its unrolled running counter formed an eight-step
				// compare/add recurrence (15.83 ns).  Count the complete mask with
				// the balanced tree instead.
				if (next_low != 0) {
					TapaDenseCandidateGroup token = 0;
					token.range(12, 0) = address;
					token.range(20, 13) = next_low;
					candidate_stream.write(token);
					local_candidates += tapa_popcount8(next_low);
				}
				address_pending = false;
			} else if (bit_pending) {
				// Decode only the selected byte in this stage.  The dynamic 64-to-8
				// selection and the eight-bit priority encoder are now isolated from
				// the preceding byte-nonzero tree and the following bank mux.
				const ap_uint<8> octet = pending_addresses
					>> ((ap_uint<6>)pending_byte << 3);
				const ap_uint<3> bit_in_byte = tapa_first_set8(octet);
				const ap_uint<6> bit
					= ((ap_uint<6>)pending_byte << 3) | bit_in_byte;
				address = ((ap_uint<13>)word << 6) | bit;
				pending_bit = bit;
				bit_pending = false;
				address_pending = true;
			} else {
				local_active_groups++;
				pending_addresses = active_addresses;
				ap_uint<3> byte = 0;
				if (active_addresses.range(7, 0) != 0) byte = 0;
				else if (active_addresses.range(15, 8) != 0) byte = 1;
				else if (active_addresses.range(23, 16) != 0) byte = 2;
				else if (active_addresses.range(31, 24) != 0) byte = 3;
				else if (active_addresses.range(39, 32) != 0) byte = 4;
				else if (active_addresses.range(47, 40) != 0) byte = 5;
				else if (active_addresses.range(55, 48) != 0) byte = 6;
				else byte = 7;
				active_addresses = active_addresses
					& (ap_uint<64>)(active_addresses - (ap_uint<64>)1);
				pending_byte = byte;
				bit_pending = true;
			}
		}
	}
	TapaDenseCandidateGroup done = 0;
	done[21] = 1;
	candidate_stream.write(done);
	*bitmap_word_scans = local_word_scans;
	*nonempty_words = local_nonempty_words;
	*active_address_groups = local_active_groups;
	*candidate_columns = local_candidates;
}

static void tapa_dense_reduce_candidate_groups(
		const TapaDenseCell accumulator[kTapaDenseBanks][kTapaDensePhases]
			[kTapaDenseBankCapacity],
		ap_uint<kTapaDenseEpochBits> epoch, ap_uint<16> virtual_base,
		id_t dense_base,
		hls::stream<TapaDenseCandidateGroup>& candidate_stream,
		hls::stream<TapaDenseReducedGroup>& reduced_stream) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=1
#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=2
	bool done = false;
	dense_reduce_groups:
	while (!done) {
#pragma HLS PIPELINE II=1
		const TapaDenseCandidateGroup token = candidate_stream.read();
		if (token[21]) {
			TapaDenseReducedGroup result = 0;
			result[531] = 1;
			reduced_stream.write(result);
			done = true;
		} else {
			const ap_uint<13> address = token.range(12, 0);
			const ap_uint<8> active_low = token.range(20, 13);
			const ap_uint<13> local_address
				= address - ((ap_uint<13>)virtual_base >> 3);
			const ap_uint<3> permutation
				= (local_address ^ (local_address >> 3)).range(2, 0);
			ap_uint<512> bank_items = 0;
			ap_uint<8> candidate_bank_mask = 0;
			ap_uint<8> nonzero_bank_mask = 0;
			for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
				const ap_uint<3> low = (ap_uint<3>)bank ^ permutation;
				const bool candidate = active_low[(unsigned)low];
				const id_t virtual_col = ((id_t)address << 3) | low;
				const id_t col = virtual_col - virtual_base;
				float phase_value[16];
#pragma HLS ARRAY_PARTITION variable=phase_value complete dim=1
				for (int phase = 0; phase < 16; ++phase) {
#pragma HLS UNROLL
					TapaDenseCell cell = 0;
					if (phase < kTapaDensePhases)
						cell = accumulator[bank][phase][address];
					phase_value[phase] = phase < kTapaDensePhases
						&& cell.range(kTapaDenseEpochBits + 31, 32) == epoch
						? tapa_bits_to_float(cell.range(31, 0)) : 0.0f;
				}
				const float sum01 = phase_value[0] + phase_value[1];
				const float sum23 = phase_value[2] + phase_value[3];
				const float sum45 = phase_value[4] + phase_value[5];
				const float sum67 = phase_value[6] + phase_value[7];
				const float sum89 = phase_value[8] + phase_value[9];
				const float sum_ab = phase_value[10] + phase_value[11];
				const float sum_cd = phase_value[12] + phase_value[13];
				const float sum_ef = phase_value[14] + phase_value[15];
				const float sum03 = sum01 + sum23;
				const float sum47 = sum45 + sum67;
				const float sum8b = sum89 + sum_ab;
				const float sum_cf = sum_cd + sum_ef;
				const float value = (sum03 + sum47) + (sum8b + sum_cf);
				bank_items.range(bank * 64 + 63, bank * 64)
					= tapa_pack_item(dense_base + col, value);
				candidate_bank_mask[bank] = candidate;
				nonzero_bank_mask[bank] = candidate && tapa_nonzero(value);
			}
			TapaDenseReducedGroup result = 0;
			result.range(511, 0) = bank_items;
			result.range(519, 512) = candidate_bank_mask;
			result.range(527, 520) = nonzero_bank_mask;
			result.range(530, 528) = permutation;
			reduced_stream.write(result);
		}
	}
}

static ap_uint<64> tapa_dense_select_physical_bank(
		ap_uint<512> bank_items, ap_uint<3> bank) {
#pragma HLS INLINE
	switch ((unsigned)bank) {
	case 0: return bank_items.range(63, 0);
	case 1: return bank_items.range(127, 64);
	case 2: return bank_items.range(191, 128);
	case 3: return bank_items.range(255, 192);
	case 4: return bank_items.range(319, 256);
	case 5: return bank_items.range(383, 320);
	case 6: return bank_items.range(447, 384);
	default: return bank_items.range(511, 448);
	}
}

static void tapa_dense_compact_reduced_groups(
		hls::stream<TapaDenseReducedGroup>& reduced_stream,
		ap_uint<64> row_output[8][kTapaDenseCapacity / 8],
		id_t output_offset,
		id_t* extract_iterations, id_t* output_packets,
		id_t* result_row_nnz,
		id_t bank_candidate_columns[kTapaDenseBanks]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=row_output complete dim=1
#pragma HLS ARRAY_PARTITION variable=bank_candidate_columns complete dim=1
	id_t row_nnz = 0;
	id_t consumed = 0;
	id_t local_bank_candidates[kTapaDenseBanks];
#pragma HLS ARRAY_PARTITION variable=local_bank_candidates complete dim=1
	for (int candidate_bank = 0; candidate_bank < kTapaDenseBanks;
			++candidate_bank) {
#pragma HLS UNROLL
		local_bank_candidates[candidate_bank] = 0;
	}
	bool done = false;
	dense_compact_groups:
	while (!done) {
#pragma HLS PIPELINE II=1
		const TapaDenseReducedGroup token = reduced_stream.read();
		if (token[531]) {
			done = true;
		} else {
			const ap_uint<512> bank_items = token.range(511, 0);
			const ap_uint<8> candidate_bank_mask = token.range(519, 512);
			const ap_uint<8> nonzero_bank_mask = token.range(527, 520);
			const ap_uint<3> permutation = token.range(530, 528);
			for (int candidate_bank = 0; candidate_bank < kTapaDenseBanks;
					++candidate_bank) {
#pragma HLS UNROLL
				if (candidate_bank_mask[candidate_bank])
					local_bank_candidates[candidate_bank]++;
			}
			ap_uint<64> logical_items[8];
			ap_uint<8> logical_valid = 0;
#pragma HLS ARRAY_PARTITION variable=logical_items complete dim=1
			for (int low = 0; low < 8; ++low) {
#pragma HLS UNROLL
				const ap_uint<3> bank = (ap_uint<3>)low ^ permutation;
				logical_items[low]
					= tapa_dense_select_physical_bank(bank_items, bank);
				logical_valid[low] = nonzero_bank_mask[(unsigned)bank];
			}
			ap_uint<4> prefix[8];
#pragma HLS ARRAY_PARTITION variable=prefix complete dim=1
			prefix[0] = 0;
			for (int low = 1; low < 8; ++low) {
#pragma HLS UNROLL
				prefix[low] = prefix[low - 1] + logical_valid[low - 1];
			}
			for (int output_bank = 0; output_bank < 8; ++output_bank) {
#pragma HLS UNROLL
				bool write_enable = false;
				ap_uint<64> write_item = 0;
				id_t write_word = 0;
				for (int low = 0; low < 8; ++low) {
#pragma HLS UNROLL
					const id_t position
						= output_offset + row_nnz + prefix[low];
					if (logical_valid[low] && (position & 7) == output_bank) {
						write_enable = true;
						write_item = logical_items[low];
						write_word = position >> 3;
					}
				}
				if (write_enable)
					row_output[output_bank][write_word] = write_item;
			}
			consumed += tapa_popcount8(candidate_bank_mask);
			row_nnz += tapa_popcount8(logical_valid);
		}
	}
	*extract_iterations = consumed;
	*output_packets = (row_nnz + 7) >> 3;
	*result_row_nnz = row_nnz;
	for (int candidate_bank = 0; candidate_bank < kTapaDenseBanks;
			++candidate_bank) {
#pragma HLS UNROLL
		bank_candidate_columns[candidate_bank]
			= local_bank_candidates[candidate_bank];
	}
}

static void tapa_dense_stream_extract(
		const ap_uint<64> leaf_bits[kTapaDenseBanks][kTapaDensePhases]
			[kTapaDenseLeafWords],
		const ap_uint<kTapaDenseEpochBits>
			leaf_word_tag[kTapaDenseBanks][kTapaDensePhases]
				[kTapaDenseLeafWords],
		const TapaDenseCell accumulator[kTapaDenseBanks][kTapaDensePhases]
			[kTapaDenseBankCapacity],
		ap_uint<kTapaDenseEpochBits> candidate_epoch,
		ap_uint<kTapaDenseEpochBits> accumulator_epoch, id_t N,
		ap_uint<16> virtual_base, id_t dense_base, id_t output_offset,
		ap_uint<64> row_output[8][kTapaDenseCapacity / 8],
		id_t* bitmap_word_scans, id_t* nonempty_words,
		id_t* active_address_groups, id_t* candidate_columns,
		id_t* extract_iterations, id_t* output_packets,
		id_t* result_row_nnz,
		id_t bank_candidate_columns[kTapaDenseBanks]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
	hls::stream<TapaDenseCandidateGroup> candidate_stream("dense_candidates");
	hls::stream<TapaDenseReducedGroup> reduced_stream("dense_reduced_groups");
#pragma HLS STREAM variable=candidate_stream depth=64
#pragma HLS STREAM variable=reduced_stream depth=64
	tapa_dense_generate_candidates(leaf_bits, leaf_word_tag, candidate_epoch, N,
		virtual_base,
		candidate_stream, bitmap_word_scans,
		nonempty_words, active_address_groups, candidate_columns);
	tapa_dense_reduce_candidate_groups(accumulator, accumulator_epoch,
		virtual_base, dense_base, candidate_stream, reduced_stream);
	tapa_dense_compact_reduced_groups(reduced_stream, row_output, output_offset,
		extract_iterations,
		output_packets, result_row_nnz, bank_candidate_columns);
}

template <int SOURCE>
static void tapa_dense_stream_refill(
		tapa::istream<TapaProductPacket>& input, id_t expected_row,
		ap_uint<2> expected_mode, bool dual_row, ap_uint<3> route_slot,
		TapaProductPacket
			source_packet[kTapaDenseRouteSlots][kTapaNumShards],
		ap_uint<24> source_bank_meta[kTapaDenseRouteSlots][kTapaNumShards],
		ap_uint<8> source_valid[kTapaDenseRouteSlots][kTapaNumShards],
		ap_uint<1> source_context[kTapaDenseRouteSlots][kTapaNumShards],
		bool source_ended[2][kTapaNumShards],
		id_t source_reported_products[2][kTapaNumShards],
		id_t source_protocol_errors[kTapaNumShards]) {
#pragma HLS INLINE
#pragma HLS DEPENDENCE variable=source_packet inter false
#pragma HLS DEPENDENCE variable=source_bank_meta inter false
#pragma HLS DEPENDENCE variable=source_valid inter false
#pragma HLS DEPENDENCE variable=source_context inter false
	if (source_valid[(unsigned)route_slot][SOURCE] != 0
			|| (source_ended[0][SOURCE] && source_ended[1][SOURCE])) return;
	TapaProductPacket packet;
	if (!input.try_read(packet)) return;
	if (packet[552]) {
		source_protocol_errors[SOURCE]++;
		source_ended[0][SOURCE] = true;
		source_ended[1][SOURCE] = true;
		return;
	}
	const id_t packet_row = packet.range(547, 516);
	const bool first_row = packet_row == expected_row;
	const bool second_row = dual_row && packet_row == expected_row + 1;
	const bool tag_error = (!first_row && !second_row)
		|| packet.range(549, 548) != expected_mode;
	if (tag_error) source_protocol_errors[SOURCE]++;
	const ap_uint<1> context = second_row ? (ap_uint<1>)1 : (ap_uint<1>)0;
	if (packet[551]) {
		source_reported_products[(unsigned)context][SOURCE]
			= packet.range(31, 0);
		if (packet[553]) source_protocol_errors[SOURCE]++;
		source_ended[(unsigned)context][SOURCE] = true;
		return;
	}

	const ap_uint<4> valid = packet.range(515, 512);
	ap_uint<24> banks = 0;
	for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
		const ap_uint<64> item = packet.range(lane * 64 + 63, lane * 64);
		banks.range(lane * 3 + 2, lane * 3)
			= tapa_dense_bank(item.range(31, 0));
	}
	source_packet[(unsigned)route_slot][SOURCE] = packet;
	source_bank_meta[(unsigned)route_slot][SOURCE] = banks;
	source_context[(unsigned)route_slot][SOURCE] = context;
	source_valid[(unsigned)route_slot][SOURCE]
		= valid == 8 ? (ap_uint<8>)0xff
		: (ap_uint<8>)(((ap_uint<9>)1 << valid) - 1);
}

template <int SOURCE>
static TapaDenseRefill tapa_dense_stream_try_refill(
		tapa::istream<TapaProductPacket>& input, id_t expected_row,
		ap_uint<2> expected_mode, bool dual_row, ap_uint<8> slot_valid,
		bool first_ended, bool second_ended) {
#pragma HLS INLINE
	TapaDenseRefill refill = 0;
	if (slot_valid != 0 || (first_ended && second_ended)) return refill;
	TapaProductPacket packet;
	if (!input.try_read(packet)) return refill;
	if (packet[552]) {
		refill.range(623, 622) = 1;
		refill[624] = 1;
		return refill;
	}
	const id_t packet_row = packet.range(547, 516);
	const bool first_row = packet_row == expected_row;
	const bool second_row = dual_row && packet_row == expected_row + 1;
	const ap_uint<1> context = second_row ? (ap_uint<1>)1 : (ap_uint<1>)0;
	ap_uint<2> errors = ((!first_row && !second_row)
		|| packet.range(549, 548) != expected_mode)
		? (ap_uint<2>)1 : (ap_uint<2>)0;
	refill[587] = context;
	if (packet[551]) {
		refill[589] = 1;
		refill.range(621, 590) = packet.range(31, 0);
		if (packet[553]) errors++;
	} else {
		const ap_uint<4> valid = packet.range(515, 512);
		if (packet[553]) errors = 1;
		ap_uint<24> banks = 0;
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			const ap_uint<64> item
				= packet.range(lane * 64 + 63, lane * 64);
			banks.range(lane * 3 + 2, lane * 3)
				= tapa_dense_bank(item.range(31, 0));
		}
		refill.range(554, 0) = packet;
		refill.range(578, 555) = banks;
		refill.range(586, 579) = valid == 8 ? (ap_uint<8>)0xff
			: (ap_uint<8>)(((ap_uint<9>)1 << valid) - 1);
		refill[588] = 1;
	}
	refill.range(623, 622) = errors;
	return refill;
}

template <int SOURCE>
static void tapa_dense_stream_drain_boundary(
		tapa::istream<TapaProductPacket>& input, id_t expected_row,
		ap_uint<2> expected_mode, bool dual_row,
		bool source_ended[2][kTapaNumShards],
		id_t source_reported_products[2][kTapaNumShards],
		id_t source_protocol_errors[kTapaNumShards]) {
#pragma HLS INLINE
	if (source_ended[0][SOURCE] && source_ended[1][SOURCE]) return;
	TapaProductPacket packet;
	if (!input.try_read(packet)) return;
	if (packet[552]) {
		source_protocol_errors[SOURCE]++;
		source_ended[0][SOURCE] = true;
		source_ended[1][SOURCE] = true;
		return;
	}
	const id_t packet_row = packet.range(547, 516);
	const bool first_row = packet_row == expected_row;
	const bool second_row = dual_row && packet_row == expected_row + 1;
	if ((!first_row && !second_row)
			|| packet.range(549, 548) != expected_mode)
		source_protocol_errors[SOURCE]++;
	const ap_uint<1> context = second_row ? (ap_uint<1>)1 : (ap_uint<1>)0;
	if (packet[551]) {
		source_reported_products[(unsigned)context][SOURCE]
			= packet.range(31, 0);
		if (packet[553]) source_protocol_errors[SOURCE]++;
		source_ended[(unsigned)context][SOURCE] = true;
	} else {
		// Exact route metadata should make every data lane enter the bank router.
		// Discarding an unexpected surplus packet prevents malformed metadata from
		// deadlocking the following row and turns it into a visible protocol error.
		source_protocol_errors[SOURCE]++;
	}
}

static void tapa_dense_stream_select(
		tapa::istream<TapaProductPacket>& product0,
		tapa::istream<TapaProductPacket>& product1,
		tapa::istream<TapaProductPacket>& product2,
		tapa::istream<TapaProductPacket>& product3,
		tapa::istream<TapaProductPacket>& product4,
		tapa::istream<TapaProductPacket>& product5,
		tapa::istream<TapaProductPacket>& product6,
		tapa::istream<TapaProductPacket>& product7,
		id_t expected_row, ap_uint<2> expected_mode, bool dual_row,
		id_t expected_products0, id_t expected_products1,
		hls::stream<TapaDenseContributions>& contribution_stream,
		id_t* stream_cycles, id_t* route_beats, id_t* coalesced_products,
		id_t bank_products[kTapaDenseBanks], id_t* reported_products,
		id_t* stream_protocol_errors) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=bank_products complete dim=1
	TapaProductPacket source_packet[kTapaDenseRouteSlots][kTapaNumShards];
	ap_uint<24> source_bank_meta[kTapaDenseRouteSlots][kTapaNumShards];
	ap_uint<8> source_valid[kTapaDenseRouteSlots][kTapaNumShards];
	ap_uint<1> source_context[kTapaDenseRouteSlots][kTapaNumShards];
	bool source_bank_head_valid[kTapaDenseHeadSlots][kTapaNumShards]
		[kTapaDenseBanks];
	ap_uint<8> source_bank_head_lane_mask[kTapaDenseHeadSlots]
		[kTapaNumShards][kTapaDenseBanks];
	ap_uint<64> source_bank_head_item[kTapaDenseHeadSlots][kTapaNumShards]
		[kTapaDenseBanks];
	bool source_ended[2][kTapaNumShards];
	id_t source_reported_products[2][kTapaNumShards];
	id_t source_protocol_errors[kTapaNumShards];
#pragma HLS ARRAY_PARTITION variable=source_packet complete dim=0
#pragma HLS ARRAY_PARTITION variable=source_bank_meta complete dim=0
#pragma HLS ARRAY_PARTITION variable=source_valid complete dim=0
#pragma HLS ARRAY_PARTITION variable=source_context complete dim=0
#pragma HLS ARRAY_PARTITION variable=source_bank_head_valid complete dim=0
#pragma HLS ARRAY_PARTITION variable=source_bank_head_lane_mask complete dim=0
#pragma HLS ARRAY_PARTITION variable=source_bank_head_item complete dim=0
#pragma HLS ARRAY_PARTITION variable=source_ended complete dim=0
#pragma HLS ARRAY_PARTITION variable=source_reported_products complete dim=0
#pragma HLS ARRAY_PARTITION variable=source_protocol_errors complete dim=1
	for (int slot = 0; slot < kTapaDenseRouteSlots; ++slot) {
#pragma HLS UNROLL
		for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
			source_valid[slot][source] = 0;
		}
	}
	for (int slot = 0; slot < kTapaDenseHeadSlots; ++slot) {
#pragma HLS UNROLL
		for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
			for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
				source_bank_head_valid[slot][source][bank] = false;
			}
		}
	}
	// Packet, bank metadata, context and decoded head data are all guarded by
	// source_valid/source_bank_head_valid.  Initializing those wide data arrays
	// made the pipelined loop-init net drive 17k+ loads after placement.  Keep
	// only the validity reset above; no don't-care data can be issued.
	for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
		source_ended[0][source] = false;
		source_ended[1][source] = !dual_row;
		source_reported_products[0][source] = 0;
		source_reported_products[1][source] = 0;
		source_protocol_errors[source] = 0;
	}
	for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
		bank_products[bank] = 0;
	}
	id_t bank_writes[kTapaDenseBanks];
#pragma HLS ARRAY_PARTITION variable=bank_writes complete dim=1
	for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
		bank_writes[bank] = 0;
	}

	const id_t expected_products = expected_products0
		+ (dual_row ? expected_products1 : (id_t)0);
	id_t cycles = 0;
	id_t beats = 0;
	bool routing_done = false;
	bool boundaries_seen = false;
	ap_uint<5> drain_quiet_cycles = 0;
	ap_uint<8> issued_mask_delay[kTapaDenseIssueDelay][kTapaNumShards];
#pragma HLS ARRAY_PARTITION variable=issued_mask_delay complete dim=0
	for (int stage = 0; stage < kTapaDenseIssueDelay; ++stage) {
#pragma HLS UNROLL
		for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
			issued_mask_delay[stage][source] = 0;
		}
	}
	// Eight packet contexts make the packet-head recurrence distance eight.
	// This lets HLS register the lane arbitration/equal-column match path while
	// still accepting one context every cycle.  A slot is refilled only after
	// its old head has been consumed, so each source stream remains in order.
#pragma HLS DEPENDENCE variable=source_packet inter false
#pragma HLS DEPENDENCE variable=source_bank_meta inter false
#pragma HLS DEPENDENCE variable=source_valid inter false
#pragma HLS DEPENDENCE variable=source_context inter false
#pragma HLS DEPENDENCE variable=source_bank_head_valid inter false
#pragma HLS DEPENDENCE variable=source_bank_head_lane_mask inter false
#pragma HLS DEPENDENCE variable=source_bank_head_item inter false
	// Terminate from the lightweight per-source row-boundary state.  The old
	// accepted-products termination made the complete 64-head arbitration,
	// equality matching, popcount tree and 32-bit counter one loop-carried
	// combinational path (10.485 ns in HLS).  Per-bank counters retain the exact
	// product accounting without putting the cross-bank reduction in the loop.
	while (!routing_done) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=131072 avg=512
		cycles++;

		TapaDenseContributions token = 0;
		ap_uint<4> issued_by_bank[kTapaDenseBanks];
		ap_uint<8> bank_source_mask[kTapaDenseBanks];
#pragma HLS ARRAY_PARTITION variable=issued_by_bank complete dim=1
#pragma HLS ARRAY_PARTITION variable=bank_source_mask complete dim=1
		for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
			ap_uint<8> source_match = 0;
			for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
				source_match[source]
					= source_bank_head_valid[0][source][bank];
			}
			const ap_uint<8> onehot
				= source_match & (ap_uint<8>)(~source_match + 1);
			const bool bank_issue = source_match != 0;
			const ap_uint<64> winning_item = onehot[0]
				? source_bank_head_item[0][0][bank]
				: (onehot[1] ? source_bank_head_item[0][1][bank]
				: (onehot[2] ? source_bank_head_item[0][2][bank]
				: (onehot[3] ? source_bank_head_item[0][3][bank]
				: (onehot[4] ? source_bank_head_item[0][4][bank]
				: (onehot[5] ? source_bank_head_item[0][5][bank]
				: (onehot[6] ? source_bank_head_item[0][6][bank]
					: source_bank_head_item[0][7][bank]))))));
			const ap_uint<1> winning_context = onehot[0]
				? source_context[0][0]
				: (onehot[1] ? source_context[0][1]
				: (onehot[2] ? source_context[0][2]
				: (onehot[3] ? source_context[0][3]
				: (onehot[4] ? source_context[0][4]
				: (onehot[5] ? source_context[0][5]
				: (onehot[6] ? source_context[0][6]
					: source_context[0][7]))))));
			const ap_uint<32> winning_col = winning_item.range(31, 0);
			ap_uint<8> matching_sources = 0;
			const int base = bank * kTapaDenseContributionBankBits;
			token[base] = bank_issue;
			token[base + 1] = winning_context;
			token.range(base + 33, base + 2) = winning_col;
			for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
				const ap_uint<64> item
					= source_bank_head_item[0][source][bank];
				const bool matches = bank_issue
					&& source_bank_head_valid[0][source][bank]
					&& item.range(31, 0) == winning_col
					&& source_context[0][source]
						== winning_context;
				matching_sources[source] = matches;
				token.range(base + 42 + source * 32 + 31,
					base + 42 + source * 32) = matches
					? item.range(63, 32) : (ap_uint<32>)0;
			}
			token.range(base + 41, base + 34) = matching_sources;
			bank_source_mask[bank] = matching_sources;
			issued_by_bank[bank] = tapa_popcount8(matching_sources);
			bank_products[bank] += issued_by_bank[bank];
			if (source_match != 0) bank_writes[bank]++;
		}

		ap_uint<8> issued_bank_mask = 0;
		for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
			issued_bank_mask[bank] = issued_by_bank[bank] != 0;
		}
		if (issued_bank_mask != 0) {
			beats++;
			contribution_stream.write(token);
		}
		ap_uint<8> current_issued_mask[kTapaNumShards];
#pragma HLS ARRAY_PARTITION variable=current_issued_mask complete dim=1
		for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
			ap_uint<8> issued_mask = 0;
			for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
				if (bank_source_mask[bank][source])
					issued_mask
						|= source_bank_head_lane_mask[0][source][bank];
			}
			current_issued_mask[source] = issued_mask;
		}
		// Delay each arbitration result until its packet state has rotated from
		// active slot 0 to refill slot 5.  Three registers break the remaining
		// arbitration -> valid-write path while preserving exact packet
		// association and one new arbitration every cycle.
		for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
			const ap_uint<8> retiring_mask = issued_mask_delay[0][source];
			for (int stage = 0; stage < kTapaDenseIssueDelay - 1; ++stage) {
#pragma HLS UNROLL
				issued_mask_delay[stage][source]
					= issued_mask_delay[stage + 1][source];
			}
			issued_mask_delay[kTapaDenseIssueDelay - 1][source]
				= current_issued_mask[source];
			source_valid[kTapaDenseRefillSlot][source] &= ~retiring_mask;
		}
		// Refill at slot 5 after applying its aligned issued mask.  The packet then
		// rotates to slot 4, where bank heads are decoded in the next cycle; this
		// keeps FIFO/row validation and lane-to-bank decode on separate paths.
		TapaDenseRefill refill[kTapaNumShards];
#pragma HLS ARRAY_PARTITION variable=refill complete dim=1
		refill[0] = tapa_dense_stream_try_refill<0>(product0, expected_row,
			expected_mode, dual_row,
			source_valid[kTapaDenseRefillSlot][0], source_ended[0][0],
			source_ended[1][0]);
		refill[1] = tapa_dense_stream_try_refill<1>(product1, expected_row,
			expected_mode, dual_row,
			source_valid[kTapaDenseRefillSlot][1], source_ended[0][1],
			source_ended[1][1]);
		refill[2] = tapa_dense_stream_try_refill<2>(product2, expected_row,
			expected_mode, dual_row,
			source_valid[kTapaDenseRefillSlot][2], source_ended[0][2],
			source_ended[1][2]);
		refill[3] = tapa_dense_stream_try_refill<3>(product3, expected_row,
			expected_mode, dual_row,
			source_valid[kTapaDenseRefillSlot][3], source_ended[0][3],
			source_ended[1][3]);
		refill[4] = tapa_dense_stream_try_refill<4>(product4, expected_row,
			expected_mode, dual_row,
			source_valid[kTapaDenseRefillSlot][4], source_ended[0][4],
			source_ended[1][4]);
		refill[5] = tapa_dense_stream_try_refill<5>(product5, expected_row,
			expected_mode, dual_row,
			source_valid[kTapaDenseRefillSlot][5], source_ended[0][5],
			source_ended[1][5]);
		refill[6] = tapa_dense_stream_try_refill<6>(product6, expected_row,
			expected_mode, dual_row,
			source_valid[kTapaDenseRefillSlot][6], source_ended[0][6],
			source_ended[1][6]);
		refill[7] = tapa_dense_stream_try_refill<7>(product7, expected_row,
			expected_mode, dual_row,
			source_valid[kTapaDenseRefillSlot][7], source_ended[0][7],
			source_ended[1][7]);
		for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
			const TapaDenseRefill loaded = refill[source];
			source_protocol_errors[source] += loaded.range(623, 622);
			if (loaded[624]) {
				source_ended[0][source] = true;
				source_ended[1][source] = true;
			} else if (loaded[589]) {
				const ap_uint<1> context = loaded[587];
				source_reported_products[(unsigned)context][source]
					= loaded.range(621, 590);
				source_ended[(unsigned)context][source] = true;
			} else if (loaded[588]) {
				source_packet[kTapaDenseRefillSlot][source]
					= loaded.range(554, 0);
				source_bank_meta[kTapaDenseRefillSlot][source]
					= loaded.range(578, 555);
				source_valid[kTapaDenseRefillSlot][source]
					= loaded.range(586, 579);
				source_context[kTapaDenseRefillSlot][source] = loaded[587];
			}
		}
		// Predecode the next bank heads in the look-ahead slot.  Arbitration at
		// slot 0 then reads one cached valid/item/lane-mask triple per source and
		// bank instead of scanning and multiplexing all eight packet lanes.
		for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
			for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
				ap_uint<8> match = 0;
				ap_uint<64> lane_item[8];
#pragma HLS ARRAY_PARTITION variable=lane_item complete dim=1
				for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
					lane_item[lane]
						= source_packet[kTapaDenseDecodeSlot][source].range(
							lane * 64 + 63, lane * 64);
					const ap_uint<3> lane_bank
						= source_bank_meta[kTapaDenseDecodeSlot][source].range(
							lane * 3 + 2, lane * 3);
					match[lane]
						= source_valid[kTapaDenseDecodeSlot][source][lane]
						&& lane_bank == (ap_uint<3>)bank;
				}
				const ap_uint<8> onehot
					= match & (ap_uint<8>)(~match + 1);
				source_bank_head_valid[kTapaDenseDecodeSlot][source][bank]
					= match != 0;
				source_bank_head_lane_mask[kTapaDenseDecodeSlot][source][bank]
					= onehot;
				source_bank_head_item[kTapaDenseDecodeSlot][source][bank]
					= onehot[0] ? lane_item[0]
					: (onehot[1] ? lane_item[1]
					: (onehot[2] ? lane_item[2]
					: (onehot[3] ? lane_item[3]
					: (onehot[4] ? lane_item[4]
					: (onehot[5] ? lane_item[5]
					: (onehot[6] ? lane_item[6] : lane_item[7]))))));
			}
		}
		// Rotate the packet contexts through all eight recurrence slots.  Decoded
		// heads use a separate one-way window below: after arbitration the old head
		// is dead, and the packet is decoded again before its next active turn.
		for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
			const TapaProductPacket recycled_packet = source_packet[0][source];
			const ap_uint<24> recycled_bank_meta = source_bank_meta[0][source];
			const ap_uint<8> recycled_valid = source_valid[0][source];
			const ap_uint<1> recycled_context = source_context[0][source];
			for (int slot = 0; slot < kTapaDenseRouteSlots - 1; ++slot) {
#pragma HLS UNROLL
				source_packet[slot][source] = source_packet[slot + 1][source];
				source_bank_meta[slot][source]
					= source_bank_meta[slot + 1][source];
				source_valid[slot][source] = source_valid[slot + 1][source];
				source_context[slot][source] = source_context[slot + 1][source];
			}
			source_packet[kTapaDenseRouteSlots - 1][source] = recycled_packet;
			source_bank_meta[kTapaDenseRouteSlots - 1][source]
				= recycled_bank_meta;
			source_valid[kTapaDenseRouteSlots - 1][source] = recycled_valid;
			source_context[kTapaDenseRouteSlots - 1][source] = recycled_context;
			for (int slot = 0; slot < kTapaDenseHeadSlots - 1; ++slot) {
#pragma HLS UNROLL
				for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
					source_bank_head_valid[slot][source][bank]
						= source_bank_head_valid[slot + 1][source][bank];
					source_bank_head_lane_mask[slot][source][bank]
						= source_bank_head_lane_mask[slot + 1][source][bank];
					source_bank_head_item[slot][source][bank]
						= source_bank_head_item[slot + 1][source][bank];
				}
			}
			for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
				source_bank_head_valid[kTapaDenseHeadSlots - 1][source][bank]
					= false;
			}
		}

		const bool all_boundaries = source_ended[0][0] && source_ended[0][1]
			&& source_ended[0][2] && source_ended[0][3]
			&& source_ended[0][4] && source_ended[0][5]
			&& source_ended[0][6] && source_ended[0][7]
			&& source_ended[1][0] && source_ended[1][1]
			&& source_ended[1][2] && source_ended[1][3]
			&& source_ended[1][4] && source_ended[1][5]
			&& source_ended[1][6] && source_ended[1][7];
		if (!boundaries_seen && all_boundaries) {
			boundaries_seen = true;
			drain_quiet_cycles = 0;
		} else if (boundaries_seen) {
			// A pending lane must rotate to slot zero and issue within one complete
			// slot turn.  Include the delayed retiring mask and decode stages in the
			// quiet window.  Any actual issue restarts the proof-of-empty window.
			if (issued_bank_mask != 0) {
				drain_quiet_cycles = 0;
			} else if (drain_quiet_cycles
					== kTapaDenseDrainQuietCycles - 1) {
				routing_done = true;
			} else {
				drain_quiet_cycles++;
			}
		}
	}

	TapaDenseContributions done = 0;
	done[kTapaDenseContributionsBits - 1] = 1;
	contribution_stream.write(done);
	// These are end-of-row diagnostics, not part of the product routing
	// recurrence.  The former combinational reductions put up to six CARRY8s
	// between the last route-loop register and the scalar result registers.  Two
	// dedicated three-cycle trees preserve exact counts without making HLS retain
	// the complete wide route-loop state across a following serial loop.
	const id_t accepted_products = tapa_dense_pipeline_sum8<0>(
		bank_products[0], bank_products[1], bank_products[2], bank_products[3],
		bank_products[4], bank_products[5], bank_products[6], bank_products[7]);
	const id_t total_bank_writes = tapa_dense_pipeline_sum8<1>(
		bank_writes[0], bank_writes[1], bank_writes[2], bank_writes[3],
		bank_writes[4], bank_writes[5], bank_writes[6], bank_writes[7]);
	const id_t reported0 = (source_reported_products[0][0]
		+ source_reported_products[0][1] + source_reported_products[0][2]
		+ source_reported_products[0][3]) + (source_reported_products[0][4]
		+ source_reported_products[0][5] + source_reported_products[0][6]
		+ source_reported_products[0][7]);
	const id_t reported1 = (source_reported_products[1][0]
		+ source_reported_products[1][1] + source_reported_products[1][2]
		+ source_reported_products[1][3]) + (source_reported_products[1][4]
		+ source_reported_products[1][5] + source_reported_products[1][6]
		+ source_reported_products[1][7]);
	const id_t errors = (source_protocol_errors[0] + source_protocol_errors[1]
		+ source_protocol_errors[2] + source_protocol_errors[3])
		+ (source_protocol_errors[4] + source_protocol_errors[5]
		+ source_protocol_errors[6] + source_protocol_errors[7]);
	*stream_cycles = cycles;
	*route_beats = beats;
	*coalesced_products = accepted_products - total_bank_writes;
	*reported_products = reported0 + reported1;
	*stream_protocol_errors = errors
		+ (reported0 != expected_products0 ? 1 : 0)
		+ (reported1 != (dual_row ? expected_products1 : (id_t)0) ? 1 : 0)
		+ (accepted_products != expected_products ? 1 : 0);
}

static void tapa_dense_stream_sum(
		hls::stream<TapaDenseContributions>& contribution_stream,
		hls::stream<TapaDenseUpdates>& update_stream) {
#pragma HLS INLINE off
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=131073 avg=512
		const TapaDenseContributions token = contribution_stream.read();
		if (token[kTapaDenseContributionsBits - 1]) {
			done = true;
		} else {
			TapaDenseUpdates updates = 0;
			for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
				const int input_base = bank * kTapaDenseContributionBankBits;
				const int output_base = bank * kTapaDenseUpdateBankBits;
				float contribution[kTapaNumShards];
#pragma HLS ARRAY_PARTITION variable=contribution complete dim=1
				for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
					contribution[source] = tapa_bits_to_float(token.range(
						input_base + 42 + source * 32 + 31,
						input_base + 42 + source * 32));
				}
				const float sum = tapa_sum8_fp32(contribution[0], contribution[1],
					contribution[2], contribution[3], contribution[4], contribution[5],
					contribution[6], contribution[7]);
				updates[output_base] = token[input_base];
				updates[output_base + 1] = token[input_base + 1];
				updates.range(output_base + 33, output_base + 2)
					= token.range(input_base + 33, input_base + 2);
				updates.range(output_base + 65, output_base + 34)
					= tapa_float_to_bits(sum);
				updates.range(output_base + 69, output_base + 66)
					= tapa_popcount8(token.range(input_base + 41, input_base + 34));
			}
			update_stream.write(updates);
		}
	}
	TapaDenseUpdates output_done = 0;
	output_done[kTapaDenseUpdatesBits - 1] = 1;
	update_stream.write(output_done);
}

static void tapa_dense_stream_accumulate(
		hls::stream<TapaDenseUpdates>& update_stream, id_t N0, id_t N1,
		bool dual_row,
		ap_uint<kTapaDenseEpochBits> epoch,
		TapaDenseCell accumulator[kTapaDenseBanks][kTapaDensePhases]
			[kTapaDenseBankCapacity],
		ap_uint<64> leaf_bits[kTapaDenseBanks][kTapaDensePhases]
			[kTapaDenseLeafWords],
		ap_uint<kTapaDenseEpochBits>
			leaf_word_tag[kTapaDenseBanks][kTapaDensePhases]
				[kTapaDenseLeafWords],
		id_t raw_by_bank[kTapaDenseBanks]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=1
#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=2
#pragma HLS ARRAY_PARTITION variable=leaf_bits complete dim=1
#pragma HLS ARRAY_PARTITION variable=leaf_bits complete dim=2
#pragma HLS ARRAY_PARTITION variable=leaf_word_tag complete dim=1
#pragma HLS ARRAY_PARTITION variable=leaf_word_tag complete dim=2
#pragma HLS ARRAY_PARTITION variable=raw_by_bank complete dim=1
#pragma HLS DEPENDENCE variable=accumulator inter false
#pragma HLS DEPENDENCE variable=leaf_bits inter false
#pragma HLS DEPENDENCE variable=leaf_word_tag inter false
	for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
		raw_by_bank[bank] = 0;
	}
	ap_uint<4> accumulation_phase = 0;
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=131073 avg=512
		const TapaDenseUpdates updates = update_stream.read();
		if (updates[kTapaDenseUpdatesBits - 1]) {
			done = true;
		} else {
			for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
				const int base = bank * kTapaDenseUpdateBankBits;
				if (updates[base]) {
					const id_t col = updates.range(base + 33, base + 2);
					const ap_uint<1> context = updates[base + 1];
					const id_t row_N = context ? N1 : N0;
					if (col < row_N && col < kTapaDenseCapacity) {
						const ap_uint<16> virtual_col = dual_row
							? (ap_uint<16>)(((ap_uint<16>)context << 15)
								| (ap_uint<15>)col)
							: (ap_uint<16>)col;
						const ap_uint<13> address = virtual_col >> 3;
						const TapaDenseCell cell
							= accumulator[bank][accumulation_phase][address];
						const bool phase_seen = cell.range(
							kTapaDenseEpochBits + 31, 32) == epoch;
						const float previous = phase_seen
							? tapa_bits_to_float(cell.range(31, 0)) : 0.0f;
						const float value = tapa_bits_to_float(
							updates.range(base + 65, base + 34));
						const float updated_value = previous + value;
#pragma HLS BIND_OP variable=updated_value op=fadd impl=fulldsp latency=5
						TapaDenseCell updated = 0;
						updated.range(31, 0) = tapa_float_to_bits(updated_value);
						updated.range(kTapaDenseEpochBits + 31, 32) = epoch;
						accumulator[bank][accumulation_phase][address] = updated;
						const ap_uint<7> leaf_word = address >> 6;
						const ap_uint<6> leaf_bit = address.range(5, 0);
						ap_uint<64> updated_leaf_bits
							= leaf_word_tag[bank][accumulation_phase][leaf_word]
								== epoch
							? leaf_bits[bank][accumulation_phase][leaf_word]
							: (ap_uint<64>)0;
						updated_leaf_bits[leaf_bit] = 1;
						leaf_bits[bank][accumulation_phase][leaf_word]
							= updated_leaf_bits;
						leaf_word_tag[bank][accumulation_phase][leaf_word] = epoch;
						if (phase_seen) raw_by_bank[bank]++;
					}
				}
			}
			accumulation_phase = accumulation_phase == kTapaDensePhases - 1
				? (ap_uint<4>)0 : (ap_uint<4>)(accumulation_phase + 1);
		}
	}
}

static void tapa_dense_stream_route(
		tapa::istream<TapaProductPacket>& product0,
		tapa::istream<TapaProductPacket>& product1,
		tapa::istream<TapaProductPacket>& product2,
		tapa::istream<TapaProductPacket>& product3,
		tapa::istream<TapaProductPacket>& product4,
		tapa::istream<TapaProductPacket>& product5,
		tapa::istream<TapaProductPacket>& product6,
		tapa::istream<TapaProductPacket>& product7,
		id_t expected_row, ap_uint<2> expected_mode, bool dual_row,
		id_t expected_products0, id_t expected_products1, id_t N0, id_t N1,
		id_t dense_base0, id_t dense_base1,
		ap_uint<kTapaDenseEpochBits> epoch,
		TapaDenseCell accumulator[kTapaDenseBanks][kTapaDensePhases]
			[kTapaDenseBankCapacity],
		ap_uint<64> leaf_bits[kTapaDenseBanks][kTapaDensePhases]
			[kTapaDenseLeafWords],
		ap_uint<kTapaDenseEpochBits>
			leaf_word_tag[kTapaDenseBanks][kTapaDensePhases]
				[kTapaDenseLeafWords],
		id_t* stream_cycles, id_t* route_beats,
		id_t* coalesced_products, id_t bank_products[kTapaDenseBanks],
		id_t raw_by_bank[kTapaDenseBanks], id_t* reported_products,
		id_t* stream_protocol_errors) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
	hls::stream<TapaDenseContributions> contribution_stream(
		"dense_contribution_stream");
	hls::stream<TapaDenseUpdates> update_stream("dense_update_stream");
#pragma HLS STREAM variable=contribution_stream depth=64
#pragma HLS STREAM variable=update_stream depth=64
	tapa_dense_stream_select(product0, product1, product2, product3, product4,
		product5, product6, product7, expected_row, expected_mode, dual_row,
		expected_products0, expected_products1, contribution_stream,
		stream_cycles, route_beats, coalesced_products, bank_products,
		reported_products, stream_protocol_errors);
	tapa_dense_stream_sum(contribution_stream, update_stream);
	tapa_dense_stream_accumulate(update_stream, N0, N1, dual_row, epoch, accumulator,
		leaf_bits, leaf_word_tag, raw_by_bank);
}

// Bank-routed streaming DENSE reducer.  Eight source packet heads now refill
// directly from the local-merge pass-through streams.  Bank arbitration and
// input collection therefore overlap for the complete row; packet_store is
// retained only by DIRECT/MERGE until their own streaming conversion.
static void tapa_dense_stream_bank_reduce(
		tapa::istream<TapaProductPacket>& product0,
		tapa::istream<TapaProductPacket>& product1,
		tapa::istream<TapaProductPacket>& product2,
		tapa::istream<TapaProductPacket>& product3,
		tapa::istream<TapaProductPacket>& product4,
		tapa::istream<TapaProductPacket>& product5,
		tapa::istream<TapaProductPacket>& product6,
		tapa::istream<TapaProductPacket>& product7,
		id_t expected_row, ap_uint<2> expected_mode, bool dual_row,
		id_t expected_products0, id_t expected_products1, id_t N0, id_t N1,
		id_t dense_base0, id_t dense_base1,
		ap_uint<64> row_output[8][kTapaDenseCapacity / 8],
		TapaDenseDiagnostics* diagnostics, id_t* reported_products,
		id_t* stream_protocol_errors, id_t result_row_nnz[2]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=row_output complete dim=1
	static TapaDenseCell accumulator[kTapaDenseBanks][kTapaDensePhases]
		[kTapaDenseBankCapacity];
	static ap_uint<64> leaf_bits[kTapaDenseBanks][kTapaDensePhases]
		[kTapaDenseLeafWords];
	static ap_uint<kTapaDenseEpochBits>
		leaf_word_tag[kTapaDenseBanks][kTapaDensePhases]
			[kTapaDenseLeafWords];
	static ap_uint<kTapaDenseEpochBits> epoch_state = 0;
	static bool state_initialized = false;
#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=1
#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=2
#pragma HLS ARRAY_PARTITION variable=leaf_bits complete dim=1
#pragma HLS ARRAY_PARTITION variable=leaf_bits complete dim=2
#pragma HLS ARRAY_PARTITION variable=leaf_word_tag complete dim=1
#pragma HLS ARRAY_PARTITION variable=leaf_word_tag complete dim=2
#pragma HLS BIND_STORAGE variable=accumulator type=ram_t2p impl=uram latency=2
#pragma HLS BIND_STORAGE variable=leaf_bits type=ram_t2p impl=bram latency=2
#pragma HLS BIND_STORAGE variable=leaf_word_tag type=ram_t2p impl=bram latency=2

	diagnostics->stream_cycles = 0;
	diagnostics->route_cycles = 0;
	diagnostics->replay_cycles = 0;
	diagnostics->raw_dependencies = 0;
	diagnostics->epoch_clear_cycles = 0;
	diagnostics->touched_addresses = 0;
	diagnostics->bitmap_clear_cycles = 0;
	diagnostics->bitmap_build_cycles = 0;
	diagnostics->bitmap_word_scans = 0;
	diagnostics->nonempty_words = 0;
	diagnostics->active_address_groups = 0;
	diagnostics->candidate_columns = 0;
	diagnostics->extract_iterations = 0;
	diagnostics->output_packets = 0;
	diagnostics->coalesced_products = 0;
	for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
		diagnostics->bank_products[bank] = 0;
		diagnostics->bank_unique_addresses[bank] = 0;
	}

	if (!state_initialized || epoch_state
			== (ap_uint<kTapaDenseEpochBits>)0xffffU) {
		for (id_t address = 0; address < kTapaDenseBankCapacity;
			++address) {
#pragma HLS PIPELINE II=1
			for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
				for (int phase = 0; phase < kTapaDensePhases; ++phase) {
#pragma HLS UNROLL
					accumulator[bank][phase][address] = 0;
					if (address < kTapaDenseLeafWords)
						leaf_word_tag[bank][phase][address] = 0;
				}
			}
		}
		diagnostics->epoch_clear_cycles = kTapaDenseBankCapacity;
		epoch_state = 0;
		state_initialized = true;
	}
	const ap_uint<kTapaDenseEpochBits> epoch = epoch_state + 1;
	epoch_state = epoch;

	id_t route_stream_cycles = 0;
	id_t route_beats_result = 0;
	id_t coalesced_products_result = 0;
	id_t route_reported_products = 0;
	id_t route_protocol_errors = 0;
	id_t route_bank_products[kTapaDenseBanks];
	id_t route_raw_by_bank[kTapaDenseBanks];
#pragma HLS ARRAY_PARTITION variable=route_bank_products complete dim=1
#pragma HLS ARRAY_PARTITION variable=route_raw_by_bank complete dim=1
	tapa_dense_stream_route(product0, product1, product2, product3, product4,
		product5, product6, product7, expected_row, expected_mode, dual_row,
		expected_products0, expected_products1, N0, N1,
		dense_base0, dense_base1, epoch, accumulator, leaf_bits,
		leaf_word_tag, &route_stream_cycles,
		&route_beats_result, &coalesced_products_result, route_bank_products,
		route_raw_by_bank, &route_reported_products, &route_protocol_errors);
	*reported_products = route_reported_products;
	*stream_protocol_errors = route_protocol_errors;
	diagnostics->stream_cycles = route_stream_cycles;
	diagnostics->route_cycles = route_beats_result;
	diagnostics->coalesced_products = coalesced_products_result;
	const id_t expected_products_result = expected_products0
		+ (dual_row ? expected_products1 : (id_t)0);
	const id_t ideal_width_result = kTapaDenseBanks * kTapaNumShards;
	const id_t ideal_beats_result = (expected_products_result
		+ ideal_width_result - 1) / ideal_width_result;
	diagnostics->replay_cycles = route_beats_result > ideal_beats_result
		? route_beats_result - ideal_beats_result : (id_t)0;
	for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
		diagnostics->bank_products[bank] = route_bank_products[bank];
	}
	const id_t raw01_result = route_raw_by_bank[0] + route_raw_by_bank[1];
	const id_t raw23_result = route_raw_by_bank[2] + route_raw_by_bank[3];
	const id_t raw45_result = route_raw_by_bank[4] + route_raw_by_bank[5];
	const id_t raw67_result = route_raw_by_bank[6] + route_raw_by_bank[7];
	diagnostics->raw_dependencies = (raw01_result + raw23_result)
		+ (raw45_result + raw67_result);

#if 0
	// Route metadata is exact and available before the first product.  Keep the
	// accepted counter at the external 32-bit width now that DENSE is no longer
	// artificially bounded by the 8x1024 packet store.
	id_t accepted_products = 0;
	id_t route_beats = 0;
	id_t coalesced_products = 0;
	id_t raw_by_bank[kTapaDenseBanks];
#pragma HLS ARRAY_PARTITION variable=raw_by_bank complete dim=1
	for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
		raw_by_bank[bank] = 0;
	}
	ap_uint<4> accumulation_phase = 0;
	// Keep one packet head per HBM shard.  The former implementation drained
	// an entire shard before looking at the next one, so a same-bank cluster in
	// one packet replayed even when the other seven shards had products for idle
	// banks.  Each bank now also coalesces equal-column heads from every source,
	// accepting up to eight repeated contributions with one accumulator update.
	TapaProductPacket source_packet[kTapaNumShards];
	ap_uint<24> source_bank_meta[kTapaNumShards];
	ap_uint<8> source_valid[kTapaNumShards];
	ap_uint<1> source_context[kTapaNumShards];
	bool source_ended[2][kTapaNumShards];
	id_t source_reported_products[2][kTapaNumShards];
	id_t source_protocol_errors[kTapaNumShards];
#pragma HLS ARRAY_PARTITION variable=source_packet complete dim=1
#pragma HLS ARRAY_PARTITION variable=source_bank_meta complete dim=1
#pragma HLS ARRAY_PARTITION variable=source_valid complete dim=1
#pragma HLS ARRAY_PARTITION variable=source_context complete dim=1
#pragma HLS ARRAY_PARTITION variable=source_ended complete dim=0
#pragma HLS ARRAY_PARTITION variable=source_reported_products complete dim=0
#pragma HLS ARRAY_PARTITION variable=source_protocol_errors complete dim=1
	for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
		source_packet[source] = 0;
		source_bank_meta[source] = 0;
		source_valid[source] = 0;
		source_context[source] = 0;
		source_ended[0][source] = false;
		source_ended[1][source] = !dual_row;
		source_reported_products[0][source] = 0;
		source_reported_products[1][source] = 0;
		source_protocol_errors[source] = 0;
	}
	const id_t expected_products = expected_products0
		+ (dual_row ? expected_products1 : (id_t)0);

	// Terminate from the Host-packed product count rather than a wide all-stream
	// empty reduction.  accepted_products remains a balanced popcount recurrence;
	// stream availability can stall an iteration without becoming loop control.
	while (accepted_products < expected_products) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=131072 avg=512
		diagnostics->stream_cycles++;
		tapa_dense_stream_refill<0>(product0, expected_row, expected_mode, dual_row,
			source_packet, source_bank_meta, source_valid, source_context, source_ended,
			source_reported_products, source_protocol_errors);
		tapa_dense_stream_refill<1>(product1, expected_row, expected_mode, dual_row,
			source_packet, source_bank_meta, source_valid, source_context, source_ended,
			source_reported_products, source_protocol_errors);
		tapa_dense_stream_refill<2>(product2, expected_row, expected_mode, dual_row,
			source_packet, source_bank_meta, source_valid, source_context, source_ended,
			source_reported_products, source_protocol_errors);
		tapa_dense_stream_refill<3>(product3, expected_row, expected_mode, dual_row,
			source_packet, source_bank_meta, source_valid, source_context, source_ended,
			source_reported_products, source_protocol_errors);
		tapa_dense_stream_refill<4>(product4, expected_row, expected_mode, dual_row,
			source_packet, source_bank_meta, source_valid, source_context, source_ended,
			source_reported_products, source_protocol_errors);
		tapa_dense_stream_refill<5>(product5, expected_row, expected_mode, dual_row,
			source_packet, source_bank_meta, source_valid, source_context, source_ended,
			source_reported_products, source_protocol_errors);
		tapa_dense_stream_refill<6>(product6, expected_row, expected_mode, dual_row,
			source_packet, source_bank_meta, source_valid, source_context, source_ended,
			source_reported_products, source_protocol_errors);
		tapa_dense_stream_refill<7>(product7, expected_row, expected_mode, dual_row,
			source_packet, source_bank_meta, source_valid, source_context, source_ended,
			source_reported_products, source_protocol_errors);
		// First choose one lane per {source,bank}, then one reference source per
		// bank whose equal-column peers can be coalesced.  Keep selection one-hot
		// and fixed-width.  This avoids variable source/lane
		// indexing, which otherwise synthesizes a 64-way 554-bit crossbar and makes
		// HLS scheduling intractable even though the logical arbitration is small.
		bool source_bank_valid[kTapaNumShards][kTapaDenseBanks];
		ap_uint<8> source_bank_lane_mask[kTapaNumShards][kTapaDenseBanks];
		ap_uint<64> source_bank_item[kTapaNumShards][kTapaDenseBanks];
#pragma HLS ARRAY_PARTITION variable=source_bank_valid complete dim=0
#pragma HLS ARRAY_PARTITION variable=source_bank_lane_mask complete dim=0
#pragma HLS ARRAY_PARTITION variable=source_bank_item complete dim=0
		for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
			for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
				ap_uint<8> match = 0;
				ap_uint<64> lane_item[8];
#pragma HLS ARRAY_PARTITION variable=lane_item complete dim=1
				for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
					lane_item[lane] = source_packet[source].range(
						lane * 64 + 63, lane * 64);
					const ap_uint<3> lane_bank
						= source_bank_meta[source].range(
							lane * 3 + 2, lane * 3);
					match[lane] = source_valid[source][lane]
						&& lane_bank == (ap_uint<3>)bank;
				}
				const ap_uint<8> onehot
					= match & (ap_uint<8>)(~match + 1);
				source_bank_valid[source][bank] = match != 0;
				source_bank_lane_mask[source][bank] = onehot;
				source_bank_item[source][bank] = onehot[0] ? lane_item[0]
					: (onehot[1] ? lane_item[1]
					: (onehot[2] ? lane_item[2]
					: (onehot[3] ? lane_item[3]
					: (onehot[4] ? lane_item[4]
					: (onehot[5] ? lane_item[5]
					: (onehot[6] ? lane_item[6] : lane_item[7]))))));
			}
		}

		bool bank_issue[kTapaDenseBanks];
		ap_uint<8> bank_source_mask[kTapaDenseBanks];
		ap_uint<64> winning_item[kTapaDenseBanks];
		ap_uint<1> winning_context[kTapaDenseBanks];
		float bank_combined_value[kTapaDenseBanks];
#pragma HLS ARRAY_PARTITION variable=bank_issue complete dim=1
#pragma HLS ARRAY_PARTITION variable=bank_source_mask complete dim=1
#pragma HLS ARRAY_PARTITION variable=winning_item complete dim=1
#pragma HLS ARRAY_PARTITION variable=winning_context complete dim=1
#pragma HLS ARRAY_PARTITION variable=bank_combined_value complete dim=1
		for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
			ap_uint<8> source_match = 0;
			for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
				source_match[source] = source_bank_valid[source][bank];
			}
			const ap_uint<8> onehot
				= source_match & (ap_uint<8>)(~source_match + 1);
			bank_issue[bank] = source_match != 0;
			winning_item[bank] = onehot[0] ? source_bank_item[0][bank]
				: (onehot[1] ? source_bank_item[1][bank]
				: (onehot[2] ? source_bank_item[2][bank]
				: (onehot[3] ? source_bank_item[3][bank]
				: (onehot[4] ? source_bank_item[4][bank]
				: (onehot[5] ? source_bank_item[5][bank]
				: (onehot[6] ? source_bank_item[6][bank]
					: source_bank_item[7][bank]))))));
			winning_context[bank] = onehot[0] ? source_context[0]
				: (onehot[1] ? source_context[1]
				: (onehot[2] ? source_context[2]
				: (onehot[3] ? source_context[3]
				: (onehot[4] ? source_context[4]
				: (onehot[5] ? source_context[5]
				: (onehot[6] ? source_context[6]
					: source_context[7]))))));
		}
		// Equal logical columns from different HBM shards are independent input
		// tokens but target the same accumulator address.  Combine them before the
		// phase memory so one bank update can retire up to eight contributions.
		// Non-equal heads remain queued and are considered again next cycle.
		for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
			ap_uint<8> matching_sources = 0;
			float contribution[kTapaNumShards];
#pragma HLS ARRAY_PARTITION variable=contribution complete dim=1
			const ap_uint<32> winning_col
				= winning_item[bank].range(31, 0);
			for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
				const ap_uint<64> item = source_bank_item[source][bank];
				const bool matches = bank_issue[bank]
					&& source_bank_valid[source][bank]
					&& item.range(31, 0) == winning_col
					&& source_context[source] == winning_context[bank];
				matching_sources[source] = matches;
				contribution[source] = matches
					? tapa_bits_to_float(item.range(63, 32)) : 0.0f;
			}
			bank_source_mask[bank] = matching_sources;
			bank_combined_value[bank] = tapa_sum8_fp32(
				contribution[0], contribution[1], contribution[2], contribution[3],
				contribution[4], contribution[5], contribution[6], contribution[7]);
		}

		ap_uint<4> issued_by_bank[kTapaDenseBanks];
#pragma HLS ARRAY_PARTITION variable=issued_by_bank complete dim=1
		for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
			issued_by_bank[bank] = tapa_popcount8(bank_source_mask[bank]);
			diagnostics->bank_products[bank] += issued_by_bank[bank];
			if (bank_issue[bank]) {
				const ap_uint<64> item = winning_item[bank];
				const id_t col = item.range(31, 0);
				if (col < N && col < kTapaDenseCapacity) {
					const ap_uint<16> virtual_col = dual_row
						? (ap_uint<16>)(((ap_uint<16>)winning_context[bank]
							<< 15) | (ap_uint<15>)col)
						: (ap_uint<16>)col;
					const ap_uint<13> address = virtual_col >> 3;
					const TapaDenseCell cell
						= accumulator[bank][accumulation_phase][address];
					const bool phase_seen = cell.range(
						kTapaDenseEpochBits + 31, 32) == epoch;
					const float previous = phase_seen
						? tapa_bits_to_float(cell.range(31, 0)) : 0.0f;
					const float value = bank_combined_value[bank];
					// The counted router removes packet arbitration from the loop
					// termination path, exposing URAM -> phase mux -> FP add as the next
					// timing limiter.  Give the fully pipelined FP32 adder one extra
					// stage; the eleven physical accumulation phases still exceed the
					// resulting loop pipeline depth and preserve RAW separation.
					const float updated_value = previous + value;
#pragma HLS BIND_OP variable=updated_value op=fadd impl=fulldsp latency=5
					TapaDenseCell updated = 0;
					updated.range(31, 0)
						= tapa_float_to_bits(updated_value);
					updated.range(kTapaDenseEpochBits + 31, 32) = epoch;
					accumulator[bank][accumulation_phase][address] = updated;
					const ap_uint<7> leaf_word = address >> 6;
					const ap_uint<6> leaf_bit = address.range(5, 0);
					ap_uint<64> updated_leaf_bits
						= leaf_word_tag[bank][accumulation_phase][leaf_word]
							== epoch
						? leaf_bits[bank][accumulation_phase][leaf_word]
						: (ap_uint<64>)0;
					updated_leaf_bits[leaf_bit] = 1;
					leaf_bits[bank][accumulation_phase][leaf_word]
						= updated_leaf_bits;
					leaf_word_tag[bank][accumulation_phase][leaf_word] = epoch;
					if (phase_seen) raw_by_bank[bank]++;
				}
			}
		}
		// The balanced tree counts retired source contributions, not accumulator
		// writes.  With cross-source coalescing one bank write may retire 1--8
		// products while retaining one phase-separated FP32 state update.
		const ap_uint<5> issued01 = issued_by_bank[0] + issued_by_bank[1];
		const ap_uint<5> issued23 = issued_by_bank[2] + issued_by_bank[3];
		const ap_uint<5> issued45 = issued_by_bank[4] + issued_by_bank[5];
		const ap_uint<5> issued67 = issued_by_bank[6] + issued_by_bank[7];
		const ap_uint<6> issued03 = issued01 + issued23;
		const ap_uint<6> issued47 = issued45 + issued67;
		const ap_uint<7> issued_count = issued03 + issued47;
		if (issued_count != 0) {
			ap_uint<8> issued_bank_mask = 0;
			for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
				issued_bank_mask[bank] = bank_issue[bank];
			}
			coalesced_products += issued_count
				- tapa_popcount8(issued_bank_mask);
			accepted_products += issued_count;
			route_beats++;
			accumulation_phase
				= accumulation_phase == kTapaDensePhases - 1
				? (ap_uint<4>)0 : (ap_uint<4>)(accumulation_phase + 1);
		}

		for (int source = 0; source < kTapaNumShards; ++source) {
#pragma HLS UNROLL
			ap_uint<8> issued_mask = 0;
			for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
				if (bank_source_mask[bank][source])
					issued_mask |= source_bank_lane_mask[source][bank];
			}
			source_valid[source] &= ~issued_mask;
		}
	}

	// The last data packet and its row boundary are distinct tokens.  Once all
	// expected lanes have entered the accumulator, consume the eight boundaries
	// without serializing them by source.  In the normal case this is one cycle.
	while (!(source_ended[0][0] && source_ended[0][1]
			&& source_ended[0][2] && source_ended[0][3]
			&& source_ended[0][4] && source_ended[0][5]
			&& source_ended[0][6] && source_ended[0][7]
			&& source_ended[1][0] && source_ended[1][1]
			&& source_ended[1][2] && source_ended[1][3]
			&& source_ended[1][4] && source_ended[1][5]
			&& source_ended[1][6] && source_ended[1][7])) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1024 avg=2
		diagnostics->stream_cycles++;
		tapa_dense_stream_drain_boundary<0>(product0, expected_row,
			expected_mode, dual_row, source_ended, source_reported_products,
			source_protocol_errors);
		tapa_dense_stream_drain_boundary<1>(product1, expected_row,
			expected_mode, dual_row, source_ended, source_reported_products,
			source_protocol_errors);
		tapa_dense_stream_drain_boundary<2>(product2, expected_row,
			expected_mode, dual_row, source_ended, source_reported_products,
			source_protocol_errors);
		tapa_dense_stream_drain_boundary<3>(product3, expected_row,
			expected_mode, dual_row, source_ended, source_reported_products,
			source_protocol_errors);
		tapa_dense_stream_drain_boundary<4>(product4, expected_row,
			expected_mode, dual_row, source_ended, source_reported_products,
			source_protocol_errors);
		tapa_dense_stream_drain_boundary<5>(product5, expected_row,
			expected_mode, dual_row, source_ended, source_reported_products,
			source_protocol_errors);
		tapa_dense_stream_drain_boundary<6>(product6, expected_row,
			expected_mode, dual_row, source_ended, source_reported_products,
			source_protocol_errors);
		tapa_dense_stream_drain_boundary<7>(product7, expected_row,
			expected_mode, dual_row, source_ended, source_reported_products,
			source_protocol_errors);
	}

	const id_t reported0_01
		= source_reported_products[0][0] + source_reported_products[0][1];
	const id_t reported0_23
		= source_reported_products[0][2] + source_reported_products[0][3];
	const id_t reported0_45
		= source_reported_products[0][4] + source_reported_products[0][5];
	const id_t reported0_67
		= source_reported_products[0][6] + source_reported_products[0][7];
	const id_t reported1_01
		= source_reported_products[1][0] + source_reported_products[1][1];
	const id_t reported1_23
		= source_reported_products[1][2] + source_reported_products[1][3];
	const id_t reported1_45
		= source_reported_products[1][4] + source_reported_products[1][5];
	const id_t reported1_67
		= source_reported_products[1][6] + source_reported_products[1][7];
	const id_t reported0 = (reported0_01 + reported0_23)
		+ (reported0_45 + reported0_67);
	const id_t reported1 = (reported1_01 + reported1_23)
		+ (reported1_45 + reported1_67);
	*reported_products = reported0 + reported1;
	const id_t errors01
		= source_protocol_errors[0] + source_protocol_errors[1];
	const id_t errors23
		= source_protocol_errors[2] + source_protocol_errors[3];
	const id_t errors45
		= source_protocol_errors[4] + source_protocol_errors[5];
	const id_t errors67
		= source_protocol_errors[6] + source_protocol_errors[7];
	*stream_protocol_errors = (errors01 + errors23) + (errors45 + errors67);
	if (reported0 != expected_products0
			|| reported1 != (dual_row ? expected_products1 : (id_t)0)
			|| accepted_products != expected_products)
		(*stream_protocol_errors)++;

	const id_t ideal_width = kTapaDenseBanks * kTapaNumShards;
	const id_t ideal_beats
		= (accepted_products + ideal_width - 1) / ideal_width;
	diagnostics->route_cycles = route_beats;
	diagnostics->coalesced_products = coalesced_products;
	diagnostics->replay_cycles = route_beats > ideal_beats
		? route_beats - ideal_beats : (id_t)0;
	const id_t raw01 = raw_by_bank[0] + raw_by_bank[1];
	const id_t raw23 = raw_by_bank[2] + raw_by_bank[3];
	const id_t raw45 = raw_by_bank[4] + raw_by_bank[5];
	const id_t raw67 = raw_by_bank[6] + raw_by_bank[7];
	diagnostics->raw_dependencies = (raw01 + raw23) + (raw45 + raw67);
#endif

	// Candidate generation and the persistent FP32 reduction pipeline now run
	// concurrently.  This removes the 65,536-entry cascaded BRAM list and turns
	// the former serial generate-then-extract latency into producer/consumer
	// latency with FIFO backpressure.
	result_row_nnz[0] = 0;
	result_row_nnz[1] = 0;
	id_t first_bank_candidates[kTapaDenseBanks];
	id_t second_bank_candidates[kTapaDenseBanks];
#pragma HLS ARRAY_PARTITION variable=first_bank_candidates complete dim=1
#pragma HLS ARRAY_PARTITION variable=second_bank_candidates complete dim=1
	for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
		first_bank_candidates[bank] = 0;
		second_bank_candidates[bank] = 0;
	}
	const bool route_complete = route_protocol_errors == 0
		&& route_reported_products == expected_products_result;
	const ap_uint<kTapaDenseEpochBits> candidate_epoch
		= tapa_dense_capture_extract_epoch<0>(epoch, route_complete);
	const ap_uint<kTapaDenseEpochBits> accumulator_epoch
		= tapa_dense_capture_extract_epoch<1>(epoch, route_complete);
	tapa_dense_stream_extract(leaf_bits, leaf_word_tag, accumulator,
		candidate_epoch, accumulator_epoch, N0, 0, dense_base0, 0, row_output,
		&diagnostics->bitmap_word_scans, &diagnostics->nonempty_words,
		&diagnostics->active_address_groups,
		&diagnostics->candidate_columns,
		&diagnostics->extract_iterations, &diagnostics->output_packets,
		&result_row_nnz[0], first_bank_candidates);
	if (dual_row) {
		id_t word_scans = 0;
		id_t nonempty_words = 0;
		id_t active_groups = 0;
		id_t candidates = 0;
		id_t extract_iterations = 0;
		id_t output_packets = 0;
		const id_t second_output_offset = (result_row_nnz[0] + 7) & ~id_t(7);
		tapa_dense_stream_extract(leaf_bits, leaf_word_tag, accumulator,
			candidate_epoch, accumulator_epoch, N1, (ap_uint<16>)32768,
			dense_base1, second_output_offset,
			row_output, &word_scans, &nonempty_words,
			&active_groups, &candidates, &extract_iterations,
			&output_packets, &result_row_nnz[1], second_bank_candidates);
		diagnostics->bitmap_word_scans += word_scans;
		diagnostics->nonempty_words += nonempty_words;
		diagnostics->active_address_groups += active_groups;
		diagnostics->candidate_columns += candidates;
		diagnostics->extract_iterations += extract_iterations;
		diagnostics->output_packets += output_packets;
	}
	for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
		diagnostics->bank_unique_addresses[bank]
			= first_bank_candidates[bank] + second_bank_candidates[bank];
		diagnostics->touched_addresses
			+= diagnostics->bank_unique_addresses[bank];
	}
}

static void tapa_emit_buffered_row(
		const ap_uint<64> row_output[8][kTapaDenseCapacity / 8],
		id_t base_item, id_t row_nnz, bool logical_row_last,
		tapa::ostream<TapaRowLength>& row_lengths,
		tapa::ostream<TapaOutputPacket>& output_packets) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=row_output complete dim=1
	TapaRowLength length_token = row_nnz;
	length_token[33] = logical_row_last;
	row_lengths.write(length_token);
	const id_t base_beat = base_item >> 3;
	for (id_t beat = 0; beat < ((row_nnz + 7) >> 3); ++beat) {
#pragma HLS PIPELINE II=1
		TapaOutputPacket packet = 0;
		const id_t remaining = row_nnz - (beat << 3);
		const ap_uint<4> valid = remaining > 8 ? 8 : remaining;
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			if (lane < valid)
				packet.range(lane * 64 + 63, lane * 64)
					= row_output[lane][base_beat + beat];
		}
		packet.range(515, 512) = valid;
		output_packets.write(packet);
	}
}

void tapa_adaptive_reduce(tapa::istream<TapaRowHeader>& row_headers,
		tapa::istream<TapaMergeResultPacket>& merge_results,
		tapa::istream<TapaProductPacket>& product0,
		tapa::istream<TapaProductPacket>& product1,
		tapa::istream<TapaProductPacket>& product2,
		tapa::istream<TapaProductPacket>& product3,
		tapa::istream<TapaProductPacket>& product4,
		tapa::istream<TapaProductPacket>& product5,
		tapa::istream<TapaProductPacket>& product6,
		tapa::istream<TapaProductPacket>& product7,
		id_t M, id_t N, tapa::ostream<TapaRowLength>& row_lengths,
		tapa::ostream<TapaOutputPacket>& output_packets,
		tapa::ostream<id_t>& stat_values) {
	// Keep the large buffers static in software simulation so that the TAPA
	// coroutine does not reserve them on its split stack.  In synthesis they
	// must remain ordinary function-local memories: Vitis HLS 2022.2 otherwise
	// mangles the complete reducer signature into the static-memory module name.
	// That name exceeds the filesystem component limit and the generated XO
	// silently omits the RAM module, which later fails Vivado RTL elaboration.
#ifdef __SYNTHESIS__
	TapaProductPacket
		packet_store[kTapaNumShards][kTapaPacketsPerShard];
	ap_uint<64> row_output[8][kTapaDenseCapacity / 8];
#else
	static TapaProductPacket
		packet_store[kTapaNumShards][kTapaPacketsPerShard];
	static ap_uint<64> row_output[8][kTapaDenseCapacity / 8];
#endif
	#pragma HLS ARRAY_PARTITION variable=packet_store complete dim=1
		#pragma HLS ARRAY_PARTITION variable=row_output complete dim=1
	#pragma HLS BIND_STORAGE variable=packet_store type=ram_2p impl=uram
		#pragma HLS BIND_STORAGE variable=row_output type=ram_2p impl=uram

	id_t stats[ADAPT_FP32_STAT_COUNT];
#pragma HLS ARRAY_PARTITION variable=stats complete dim=1
	for (int index = 0; index < ADAPT_FP32_STAT_COUNT; ++index)
		stats[index] = 0;
	stats[ADAPT_FP32_STAT_DIAGNOSTIC_VERSION]
		= ADAPT_FP32_DIAGNOSTIC_VERSION;
	id_t protocol_errors = 0;

	bool matrix_done = false;
	bool have_pending_header = false;
	TapaRowHeader pending_header = 0;
	while (!matrix_done) {
		const TapaRowHeader header = have_pending_header
			? pending_header : row_headers.read();
		have_pending_header = false;
		if (header[66]) {
			// Drain the explicit termination token from both persistent routes.
			const TapaMergeResultPacket merge_done = merge_results.read();
			const TapaProductPacket done0 = product0.read();
			const TapaProductPacket done1 = product1.read();
			const TapaProductPacket done2 = product2.read();
			const TapaProductPacket done3 = product3.read();
			const TapaProductPacket done4 = product4.read();
			const TapaProductPacket done5 = product5.read();
			const TapaProductPacket done6 = product6.read();
			const TapaProductPacket done7 = product7.read();
			if (!merge_done[549]
					|| !done0[552] || !done1[552] || !done2[552] || !done3[552]
					|| !done4[552] || !done5[552] || !done6[552] || !done7[552])
				protocol_errors++;
			matrix_done = true;
			break;
		}
		const id_t row = header.range(31, 0);
		const id_t task_count = header.range(63, 32);
		const ap_uint<2> mode = header.range(65, 64);
		const id_t expected_products = header.range(98, 67);
		const id_t commanded_N = header.range(131, 100);
		const id_t dense_base = header.range(163, 132);
		const bool logical_row_last = header[164];
		const bool commanded_row = commanded_N != 0;
		const id_t row_N = commanded_row ? commanded_N : N;
		TapaRowHeader second_header = 0;
		bool dual_dense = false;
		if (mode == ADAPT_FP32_DENSE && (row & 1) == 0
				&& !header[99]
				&& (commanded_row || row + 1 < M) && row_N <= 32768) {
			second_header = row_headers.read();
			dual_dense = !second_header[66]
				&& second_header.range(31, 0) == row + 1
				&& second_header.range(65, 64) == ADAPT_FP32_DENSE
				&& (!commanded_row
					|| (id_t)second_header.range(131, 100) <= 32768);
			if (!dual_dense) {
				pending_header = second_header;
				have_pending_header = true;
			}
		}
		const id_t second_expected_products = dual_dense
			? (id_t)second_header.range(98, 67) : (id_t)0;
		const id_t second_row_N = dual_dense
			? (id_t)second_header.range(131, 100) : (id_t)0;
		const id_t second_dense_base = dual_dense
			? (id_t)second_header.range(163, 132) : (id_t)0;
		const bool second_logical_row_last = dual_dense
			? (bool)second_header[164] : false;
		id_t packet_count[kTapaNumShards];
		id_t source_product_count[kTapaNumShards];
		bool row_ended[kTapaNumShards];
		bool shard_overflow[kTapaNumShards];
		id_t shard_protocol_errors[kTapaNumShards];
	#pragma HLS ARRAY_PARTITION variable=packet_count complete
	#pragma HLS ARRAY_PARTITION variable=source_product_count complete
#pragma HLS ARRAY_PARTITION variable=row_ended complete
#pragma HLS ARRAY_PARTITION variable=shard_overflow complete
#pragma HLS ARRAY_PARTITION variable=shard_protocol_errors complete
		for (int shard = 0; shard < kTapaNumShards; ++shard) {
	#pragma HLS UNROLL
			packet_count[shard] = 0;
			source_product_count[shard] = 0;
			row_ended[shard] = false;
			shard_overflow[shard] = false;
			shard_protocol_errors[shard] = 0;
		}
		id_t collect_cycles = 0;
		id_t row_nnz = 0;
		id_t second_row_nnz = 0;
		TapaDenseDiagnostics dense_diagnostics;
		if (mode == ADAPT_FP32_DENSE) {
			id_t dense_reported_products = 0;
			id_t dense_protocol_errors = 0;
			id_t dense_row_nnz[2] = {0, 0};
#pragma HLS ARRAY_PARTITION variable=dense_row_nnz complete
			tapa_dense_stream_bank_reduce(
				product0, product1, product2, product3,
				product4, product5, product6, product7,
				row, mode, dual_dense, expected_products,
				second_expected_products, row_N, second_row_N,
				dense_base, second_dense_base, row_output,
				&dense_diagnostics, &dense_reported_products,
				&dense_protocol_errors, dense_row_nnz);
			row_nnz = dense_row_nnz[0];
			second_row_nnz = dense_row_nnz[1];
			// Preserve the common row/statistics path below.  The streaming helper
			// has already consumed every source boundary for this DENSE row.
			for (int shard = 0; shard < kTapaNumShards; ++shard) {
#pragma HLS UNROLL
				row_ended[shard] = true;
			}
			source_product_count[0] = dense_reported_products;
			collect_cycles = dense_diagnostics.stream_cycles;
			protocol_errors += dense_protocol_errors;
		} else if (mode == ADAPT_FP32_MERGE
				|| mode == ADAPT_FP32_DIRECT) {
			id_t merge_reported_products = 0;
			id_t merge_protocol_errors = 0;
			bool merge_overflow = false;
			tapa_merge_fabric_store(merge_results,
				row, expected_products, row_N, dense_base, row_output,
				&collect_cycles, &merge_reported_products,
				&merge_protocol_errors, &merge_overflow, &row_nnz);
			// The streaming global merger has consumed all eight boundaries.  DIRECT
			// is a one-run special case of the same sorted stream; using this path
			// avoids the former 8,192-item central packet-store limit on an evil B
			// row without adding another large memory or a separate crossbar.
			for (int shard = 0; shard < kTapaNumShards; ++shard) {
#pragma HLS UNROLL
				row_ended[shard] = true;
			}
			source_product_count[0] = merge_reported_products;
			shard_overflow[0] = merge_overflow;
			protocol_errors += merge_protocol_errors;
		}
		while (!(row_ended[0] && row_ended[1] && row_ended[2]
				&& row_ended[3] && row_ended[4] && row_ended[5]
				&& row_ended[6] && row_ended[7])) {
#pragma HLS PIPELINE II=1
			collect_cycles++;
			tapa_accept_packet<0>(product0, row, mode, row_ended,
				packet_store, packet_count,
				source_product_count,
				shard_overflow, shard_protocol_errors);
			tapa_accept_packet<1>(product1, row, mode, row_ended,
				packet_store, packet_count,
				source_product_count,
				shard_overflow, shard_protocol_errors);
			tapa_accept_packet<2>(product2, row, mode, row_ended,
				packet_store, packet_count,
				source_product_count,
				shard_overflow, shard_protocol_errors);
			tapa_accept_packet<3>(product3, row, mode, row_ended,
				packet_store, packet_count,
				source_product_count,
				shard_overflow, shard_protocol_errors);
			tapa_accept_packet<4>(product4, row, mode, row_ended,
				packet_store, packet_count,
				source_product_count,
				shard_overflow, shard_protocol_errors);
			tapa_accept_packet<5>(product5, row, mode, row_ended,
				packet_store, packet_count,
				source_product_count,
				shard_overflow, shard_protocol_errors);
			tapa_accept_packet<6>(product6, row, mode, row_ended,
				packet_store, packet_count,
				source_product_count,
				shard_overflow, shard_protocol_errors);
			tapa_accept_packet<7>(product7, row, mode, row_ended,
				packet_store, packet_count,
				source_product_count,
				shard_overflow, shard_protocol_errors);
		}

		id_t row_products = 0;
		bool overflow = false;
		for (int shard = 0; shard < kTapaNumShards; ++shard) {
#pragma HLS UNROLL
			row_products += source_product_count[shard];
			overflow = overflow || shard_overflow[shard];
			protocol_errors += shard_protocol_errors[shard];
		}
		const id_t completed_logical_rows
			= (logical_row_last ? (id_t)1 : (id_t)0)
			+ (dual_dense && second_logical_row_last ? (id_t)1 : (id_t)0);
		stats[ADAPT_FP32_STAT_ROWS] += completed_logical_rows;
		stats[ADAPT_FP32_STAT_PARTIAL_PRODUCTS] += row_products;
		if (overflow) {
			stats[ADAPT_FP32_STAT_INPUT_OVERFLOW_ROWS]++;
		} else if (mode == ADAPT_FP32_EMPTY) {
			stats[ADAPT_FP32_STAT_EMPTY_ROWS] += completed_logical_rows;
			if (task_count != 0 || row_products != 0) protocol_errors++;
		} else if (mode == ADAPT_FP32_DIRECT) {
			// A single input run is already sorted and duplicate-free.  Its packets
			// were streamed through the global selector above, which degenerates to
			// a pass-through for the one active shard and has already filled
			// row_output.  No packet materialization or second reduction is needed.
			stats[ADAPT_FP32_STAT_DIRECT_ROWS] += completed_logical_rows;
		} else if (mode == ADAPT_FP32_DENSE) {
			stats[ADAPT_FP32_STAT_DENSE_ROWS] += completed_logical_rows;
			stats[ADAPT_FP32_STAT_DENSE_ROUTE_BEATS]
				+= dense_diagnostics.route_cycles;
			stats[ADAPT_FP32_STAT_BANK_REPLAY_CYCLES]
				+= dense_diagnostics.replay_cycles;
			stats[ADAPT_FP32_STAT_RAW_DEPENDENCIES]
				+= dense_diagnostics.raw_dependencies;
			stats[ADAPT_FP32_STAT_DENSE_COLLECT_CYCLES] += collect_cycles;
			stats[ADAPT_FP32_STAT_DENSE_EPOCH_CLEAR_CYCLES]
				+= dense_diagnostics.epoch_clear_cycles;
			for (int bank = 0; bank < kTapaDenseBanks; ++bank) {
#pragma HLS UNROLL
				stats[ADAPT_FP32_STAT_DENSE_BANK0_PRODUCTS + bank]
					+= dense_diagnostics.bank_products[bank];
				stats[ADAPT_FP32_STAT_DENSE_BANK0_UNIQUE_ADDRESSES + bank]
					+= dense_diagnostics.bank_unique_addresses[bank];
			}
			stats[ADAPT_FP32_STAT_DENSE_TOUCHED_ADDRESSES]
				+= dense_diagnostics.touched_addresses;
			stats[ADAPT_FP32_STAT_DENSE_BITMAP_CLEAR_CYCLES]
				+= dense_diagnostics.bitmap_clear_cycles;
			stats[ADAPT_FP32_STAT_DENSE_BITMAP_BUILD_CYCLES]
				+= dense_diagnostics.bitmap_build_cycles;
			stats[ADAPT_FP32_STAT_DENSE_BITMAP_WORD_SCANS]
				+= dense_diagnostics.bitmap_word_scans;
			stats[ADAPT_FP32_STAT_DENSE_NONEMPTY_WORDS]
				+= dense_diagnostics.nonempty_words;
			stats[ADAPT_FP32_STAT_DENSE_ACTIVE_ADDRESS_GROUPS]
				+= dense_diagnostics.active_address_groups;
			stats[ADAPT_FP32_STAT_DENSE_CANDIDATE_COLUMNS]
				+= dense_diagnostics.candidate_columns;
			stats[ADAPT_FP32_STAT_DENSE_EXTRACT_ITERATIONS]
				+= dense_diagnostics.extract_iterations;
			stats[ADAPT_FP32_STAT_DENSE_OUTPUT_PACKETS]
				+= dense_diagnostics.output_packets;
			stats[ADAPT_FP32_STAT_DENSE_COALESCED_PRODUCTS]
				+= dense_diagnostics.coalesced_products;
		} else {
			stats[ADAPT_FP32_STAT_MERGE_ROWS] += completed_logical_rows;
			stats[ADAPT_FP32_STAT_MERGE_STREAM_CYCLES] += collect_cycles;
			if (task_count > 1)
				stats[ADAPT_FP32_STAT_MERGE_TRANSACTIONS] += task_count - 1;
		}

		tapa_emit_buffered_row(row_output, 0, row_nnz, logical_row_last,
			row_lengths, output_packets);
		if (dual_dense) {
			const id_t second_base = (row_nnz + 7) & ~id_t(7);
			tapa_emit_buffered_row(row_output, second_base, second_row_nnz,
				second_logical_row_last,
				row_lengths, output_packets);
		}
	}
	if (protocol_errors != 0)
		stats[ADAPT_FP32_STAT_INPUT_OVERFLOW_ROWS] += protocol_errors;
	TapaRowLength done = 0;
	done[32] = 1;
	row_lengths.write(done);
	for (int index = 0; index < ADAPT_FP32_STAT_COUNT; ++index)
		stat_values.write(stats[index]);
}

void tapa_write_csr(tapa::istream<TapaRowLength>& row_lengths,
		tapa::istream<TapaOutputPacket>& output_packets,
		tapa::istream<id_t>& stat_values, id_t C_capacity,
		tapa::mmap<id_t> C_rcsr,
		tapa::mmap<ap_uint<512> > C_item0,
		tapa::mmap<ap_uint<512> > C_item1,
		tapa::mmap<ap_uint<512> > C_item2,
		tapa::mmap<ap_uint<512> > C_item3,
		tapa::mmap<id_t> stats) {
	id_t row = 0;
	id_t output_offset = 0;
	id_t overflow_rows = 0;
	id_t word_index = 0;
	ap_uint<512> pending = 0;
	ap_uint<3> pending_count = 0;
	C_rcsr[0] = 0;
	bool done = false;
	while (!done) {
		const TapaRowLength length_token = row_lengths.read();
		if (length_token[32]) {
			done = true;
			break;
		}
		const id_t row_nnz = length_token.range(31, 0);
		const bool logical_row_last = length_token[33];
		const bool keep = output_offset + row_nnz <= C_capacity;
		if (!keep) overflow_rows++;
		for (id_t beat = 0; beat < ((row_nnz + 7) >> 3); ++beat) {
#pragma HLS PIPELINE II=1
			const TapaOutputPacket packet = output_packets.read();
			if (keep) {
				const ap_uint<4> valid = packet.range(515, 512);
				const ap_uint<512> packet_data = packet.range(511, 0);
				const ap_uint<10> shift = (ap_uint<10>)pending_count << 6;
				// One 8-way, 64-bit-granularity barrel shift replaces eight
				// dependent dynamic range writes (the latter formed a 9.17 ns chain).
				const ap_uint<1024> combined = (ap_uint<1024>)pending
					| ((ap_uint<1024>)packet_data << shift);
				const ap_uint<5> combined_count = pending_count + valid;
				if (combined_count >= 8) {
					const ap_uint<512> output_word = combined.range(511, 0);
					const id_t striped_index = word_index >> 2;
					switch (word_index & 3) {
					case 0: C_item0[striped_index] = output_word; break;
					case 1: C_item1[striped_index] = output_word; break;
					case 2: C_item2[striped_index] = output_word; break;
					default: C_item3[striped_index] = output_word; break;
					}
					word_index++;
					pending = combined.range(1023, 512);
					pending_count = combined_count - 8;
				} else {
					pending = combined.range(511, 0);
					pending_count = combined_count;
				}
			}
		}
		if (keep) output_offset += row_nnz;
		if (logical_row_last) C_rcsr[++row] = output_offset;
	}
	if (pending_count != 0) {
		const id_t striped_index = word_index >> 2;
		switch (word_index & 3) {
		case 0: C_item0[striped_index] = pending; break;
		case 1: C_item1[striped_index] = pending; break;
		case 2: C_item2[striped_index] = pending; break;
		default: C_item3[striped_index] = pending; break;
		}
	}
	for (int index = 0; index < ADAPT_FP32_STAT_COUNT; ++index) {
		id_t value = stat_values.read();
		if (index == ADAPT_FP32_STAT_OUTPUT_NNZ) value = output_offset;
		if (index == ADAPT_FP32_STAT_OUTPUT_OVERFLOW_ROWS)
			value += overflow_rows;
		stats[index] = value;
	}
}

void adaptive_hbm_spgemm(
		tapa::mmap<const ap_uint<512> > task_words,
		tapa::mmap<const id_t> row_task_ptr,
		tapa::mmap<const ap_uint<32> > route,
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
		id_t M, id_t N, id_t force_mode, id_t C_capacity,
		tapa::mmap<id_t> C_rcsr,
		tapa::mmap<ap_uint<512> > C_item0,
		tapa::mmap<ap_uint<512> > C_item1,
		tapa::mmap<ap_uint<512> > C_item2,
		tapa::mmap<ap_uint<512> > C_item3,
		tapa::mmap<id_t> stats) {
	// TAPA maps FIFO depths below 128 to distributed SRLs.  Keep the wide,
	// 555-bit producer FIFOs at depth 128 so they use BRAM, but terminate them
	// before the central reducer with registered task boundaries.  The shallow
	// product FIFOs are reducer-local elasticity; they must not absorb the deep
	// buffering or concentrate another 92 BRAM tiles in the central SLR.
	tapa::stream<TapaRowHeader, 128> row_headers("row_headers");
	tapa::stream<TapaReaderCommand, 128> command0("command0");
	tapa::stream<TapaReaderCommand, 128> command1("command1");
	tapa::stream<TapaReaderCommand, 128> command2("command2");
	tapa::stream<TapaReaderCommand, 128> command3("command3");
	tapa::stream<TapaReaderCommand, 128> command4("command4");
	tapa::stream<TapaReaderCommand, 128> command5("command5");
	tapa::stream<TapaReaderCommand, 128> command6("command6");
	tapa::stream<TapaReaderCommand, 128> command7("command7");
	// Fetch and scale are separate TAPA tasks rather than an HLS DATAFLOW
	// wrapper.  A four-entry elastic FIFO is sufficient because both stages
	// target II=1, and it avoids a Vitis HLS 2022.2 Block_entry scheduler crash.
	tapa::stream<TapaBReaderRawPacket, 4> raw_beat0("raw_beat0");
	tapa::stream<TapaBReaderRawPacket, 4> raw_beat1("raw_beat1");
	tapa::stream<TapaBReaderRawPacket, 4> raw_beat2("raw_beat2");
	tapa::stream<TapaBReaderRawPacket, 4> raw_beat3("raw_beat3");
	tapa::stream<TapaBReaderRawPacket, 4> raw_beat4("raw_beat4");
	tapa::stream<TapaBReaderRawPacket, 4> raw_beat5("raw_beat5");
	tapa::stream<TapaBReaderRawPacket, 4> raw_beat6("raw_beat6");
	tapa::stream<TapaBReaderRawPacket, 4> raw_beat7("raw_beat7");
	tapa::stream<TapaProductPacket, 128> raw_product0("raw_product0");
	tapa::stream<TapaProductPacket, 128> raw_product1("raw_product1");
	tapa::stream<TapaProductPacket, 128> raw_product2("raw_product2");
	tapa::stream<TapaProductPacket, 128> raw_product3("raw_product3");
	tapa::stream<TapaProductPacket, 128> raw_product4("raw_product4");
	tapa::stream<TapaProductPacket, 128> raw_product5("raw_product5");
	tapa::stream<TapaProductPacket, 128> raw_product6("raw_product6");
	tapa::stream<TapaProductPacket, 128> raw_product7("raw_product7");
	tapa::stream<TapaProductPacket, 128> shard_product0("shard_product0");
	tapa::stream<TapaProductPacket, 128> shard_product1("shard_product1");
	tapa::stream<TapaProductPacket, 128> shard_product2("shard_product2");
	tapa::stream<TapaProductPacket, 128> shard_product3("shard_product3");
	tapa::stream<TapaProductPacket, 128> shard_product4("shard_product4");
	tapa::stream<TapaProductPacket, 128> shard_product5("shard_product5");
	tapa::stream<TapaProductPacket, 128> shard_product6("shard_product6");
	tapa::stream<TapaProductPacket, 128> shard_product7("shard_product7");
	tapa::stream<TapaProductPacket, 4> product0("product0");
	tapa::stream<TapaProductPacket, 4> product1("product1");
	tapa::stream<TapaProductPacket, 4> product2("product2");
	tapa::stream<TapaProductPacket, 4> product3("product3");
	tapa::stream<TapaProductPacket, 4> product4("product4");
	tapa::stream<TapaProductPacket, 4> product5("product5");
	tapa::stream<TapaProductPacket, 4> product6("product6");
	tapa::stream<TapaProductPacket, 4> product7("product7");
	tapa::stream<TapaProductPacket, 16> reduce_product0("reduce_product0");
	tapa::stream<TapaProductPacket, 16> reduce_product1("reduce_product1");
	tapa::stream<TapaProductPacket, 16> reduce_product2("reduce_product2");
	tapa::stream<TapaProductPacket, 16> reduce_product3("reduce_product3");
	tapa::stream<TapaProductPacket, 16> reduce_product4("reduce_product4");
	tapa::stream<TapaProductPacket, 16> reduce_product5("reduce_product5");
	tapa::stream<TapaProductPacket, 16> reduce_product6("reduce_product6");
	tapa::stream<TapaProductPacket, 16> reduce_product7("reduce_product7");
	tapa::stream<TapaMergeResultPacket, 32> merge_results("merge_results");
	tapa::streams<TapaBankedMergeItem, 8, 16> merge_source("merge_source");
	tapa::streams<TapaBankedMergeItem, 16, 16> merge_split0("merge_split0");
	tapa::streams<TapaBankedMergeItem, 8, 16> merge_stage0("merge_stage0");
	tapa::streams<TapaBankedMergeItem, 16, 16> merge_split1("merge_split1");
	tapa::streams<TapaBankedMergeItem, 8, 16> merge_stage1("merge_stage1");
	tapa::streams<TapaBankedMergeItem, 16, 16> merge_split2("merge_split2");
	tapa::streams<TapaBankedMergeItem, 8, 16> merge_stage2("merge_stage2");
	tapa::streams<TapaBankedMergeItem, 8, 32> merge_reduced("merge_reduced");
	tapa::stream<TapaMergeSparsePacket, 32> merge_sparse("merge_sparse");
	tapa::stream<TapaMergeCompactPacket, 32> merge_compact("merge_compact");
	tapa::stream<TapaRowLength, 128> row_lengths("row_lengths");
	tapa::stream<TapaOutputPacket, 128> output_packets("output_packets");
	tapa::stream<id_t, ADAPT_FP32_STAT_COUNT> stat_values("stat_values");

	tapa::task()
		.invoke(tapa_dispatch_rows, task_words, row_task_ptr, route,
			M, N, force_mode, row_headers,
			command0, command1, command2, command3,
			command4, command5, command6, command7)
		.invoke(tapa_b_reader_fetch, B0, B0_beats, command0, raw_beat0)
		.invoke(tapa_b_reader_fetch, B1, B1_beats, command1, raw_beat1)
		.invoke(tapa_b_reader_fetch, B2, B2_beats, command2, raw_beat2)
		.invoke(tapa_b_reader_fetch, B3, B3_beats, command3, raw_beat3)
		.invoke(tapa_b_reader_fetch, B4, B4_beats, command4, raw_beat4)
		.invoke(tapa_b_reader_fetch, B5, B5_beats, command5, raw_beat5)
		.invoke(tapa_b_reader_fetch, B6, B6_beats, command6, raw_beat6)
		.invoke(tapa_b_reader_fetch, B7, B7_beats, command7, raw_beat7)
		.invoke(tapa_b_reader_scale, raw_beat0, raw_product0)
		.invoke(tapa_b_reader_scale, raw_beat1, raw_product1)
		.invoke(tapa_b_reader_scale, raw_beat2, raw_product2)
		.invoke(tapa_b_reader_scale, raw_beat3, raw_product3)
		.invoke(tapa_b_reader_scale, raw_beat4, raw_product4)
		.invoke(tapa_b_reader_scale, raw_beat5, raw_product5)
		.invoke(tapa_b_reader_scale, raw_beat6, raw_product6)
		.invoke(tapa_b_reader_scale, raw_beat7, raw_product7)
		.invoke(tapa_shard_local_merge, raw_product0, shard_product0)
		.invoke(tapa_shard_local_merge, raw_product1, shard_product1)
		.invoke(tapa_shard_local_merge, raw_product2, shard_product2)
		.invoke(tapa_shard_local_merge, raw_product3, shard_product3)
		.invoke(tapa_shard_local_merge, raw_product4, shard_product4)
		.invoke(tapa_shard_local_merge, raw_product5, shard_product5)
		.invoke(tapa_shard_local_merge, raw_product6, shard_product6)
		.invoke(tapa_shard_local_merge, raw_product7, shard_product7)
		.invoke(tapa_product_register_slice, shard_product0, product0)
		.invoke(tapa_product_register_slice, shard_product1, product1)
		.invoke(tapa_product_register_slice, shard_product2, product2)
		.invoke(tapa_product_register_slice, shard_product3, product3)
		.invoke(tapa_product_register_slice, shard_product4, product4)
		.invoke(tapa_product_register_slice, shard_product5, product5)
		.invoke(tapa_product_register_slice, shard_product6, product6)
		.invoke(tapa_product_register_slice, shard_product7, product7)
		.invoke(tapa_merge_route_source, product0, reduce_product0, merge_source[0])
		.invoke(tapa_merge_route_source, product1, reduce_product1, merge_source[1])
		.invoke(tapa_merge_route_source, product2, reduce_product2, merge_source[2])
		.invoke(tapa_merge_route_source, product3, reduce_product3, merge_source[3])
		.invoke(tapa_merge_route_source, product4, reduce_product4, merge_source[4])
		.invoke(tapa_merge_route_source, product5, reduce_product5, merge_source[5])
		.invoke(tapa_merge_route_source, product6, reduce_product6, merge_source[6])
		.invoke(tapa_merge_route_source, product7, reduce_product7, merge_source[7])
		TAPA_INVOKE_FLEX_SPLIT0(0) TAPA_INVOKE_FLEX_SPLIT0(1)
		TAPA_INVOKE_FLEX_SPLIT0(2) TAPA_INVOKE_FLEX_SPLIT0(3)
		TAPA_INVOKE_FLEX_SPLIT0(4) TAPA_INVOKE_FLEX_SPLIT0(5)
		TAPA_INVOKE_FLEX_SPLIT0(6) TAPA_INVOKE_FLEX_SPLIT0(7)
		TAPA_INVOKE_FLEX_MERGE0(0, 2, 0) TAPA_INVOKE_FLEX_MERGE0(1, 3, 1)
		TAPA_INVOKE_FLEX_MERGE0(4, 6, 2) TAPA_INVOKE_FLEX_MERGE0(5, 7, 3)
		TAPA_INVOKE_FLEX_MERGE0(8, 10, 4) TAPA_INVOKE_FLEX_MERGE0(9, 11, 5)
		TAPA_INVOKE_FLEX_MERGE0(12, 14, 6) TAPA_INVOKE_FLEX_MERGE0(13, 15, 7)
		TAPA_INVOKE_FLEX_SPLIT1(0) TAPA_INVOKE_FLEX_SPLIT1(1)
		TAPA_INVOKE_FLEX_SPLIT1(2) TAPA_INVOKE_FLEX_SPLIT1(3)
		TAPA_INVOKE_FLEX_SPLIT1(4) TAPA_INVOKE_FLEX_SPLIT1(5)
		TAPA_INVOKE_FLEX_SPLIT1(6) TAPA_INVOKE_FLEX_SPLIT1(7)
		TAPA_INVOKE_FLEX_MERGE1(0, 4, 0) TAPA_INVOKE_FLEX_MERGE1(2, 6, 1)
		TAPA_INVOKE_FLEX_MERGE1(1, 5, 2) TAPA_INVOKE_FLEX_MERGE1(3, 7, 3)
		TAPA_INVOKE_FLEX_MERGE1(8, 12, 4) TAPA_INVOKE_FLEX_MERGE1(10, 14, 5)
		TAPA_INVOKE_FLEX_MERGE1(9, 13, 6) TAPA_INVOKE_FLEX_MERGE1(11, 15, 7)
		TAPA_INVOKE_FLEX_SPLIT2(0) TAPA_INVOKE_FLEX_SPLIT2(1)
		TAPA_INVOKE_FLEX_SPLIT2(2) TAPA_INVOKE_FLEX_SPLIT2(3)
		TAPA_INVOKE_FLEX_SPLIT2(4) TAPA_INVOKE_FLEX_SPLIT2(5)
		TAPA_INVOKE_FLEX_SPLIT2(6) TAPA_INVOKE_FLEX_SPLIT2(7)
		TAPA_INVOKE_FLEX_MERGE2(0, 8, 0) TAPA_INVOKE_FLEX_MERGE2(2, 10, 1)
		TAPA_INVOKE_FLEX_MERGE2(4, 12, 2) TAPA_INVOKE_FLEX_MERGE2(6, 14, 3)
		TAPA_INVOKE_FLEX_MERGE2(1, 9, 4) TAPA_INVOKE_FLEX_MERGE2(3, 11, 5)
		TAPA_INVOKE_FLEX_MERGE2(5, 13, 6) TAPA_INVOKE_FLEX_MERGE2(7, 15, 7)
		TAPA_INVOKE_FLEX_FILTER(0) TAPA_INVOKE_FLEX_FILTER(1)
		TAPA_INVOKE_FLEX_FILTER(2) TAPA_INVOKE_FLEX_FILTER(3)
		TAPA_INVOKE_FLEX_FILTER(4) TAPA_INVOKE_FLEX_FILTER(5)
		TAPA_INVOKE_FLEX_FILTER(6) TAPA_INVOKE_FLEX_FILTER(7)
		.invoke(tapa_merge_ordered_select,
			merge_reduced[0], merge_reduced[1], merge_reduced[2], merge_reduced[3],
			merge_reduced[4], merge_reduced[5], merge_reduced[6], merge_reduced[7],
			merge_sparse)
		.invoke(tapa_merge_ordered_compact, merge_sparse, merge_compact)
		.invoke(tapa_merge_ordered_emit, merge_compact, merge_results)
		.invoke(tapa_adaptive_reduce, row_headers, merge_results,
			reduce_product0, reduce_product1, reduce_product2, reduce_product3,
			reduce_product4, reduce_product5, reduce_product6, reduce_product7,
			M, N, row_lengths, output_packets, stat_values)
		.invoke(tapa_write_csr, row_lengths, output_packets, stat_values,
			C_capacity, C_rcsr, C_item0, C_item1, C_item2, C_item3, stats);
}

void adaptive_hbm_spgemm_commanded(
		tapa::mmap<const ap_uint<512> > tile_commands,
		tapa::mmap<const ap_uint<512> > task_words,
		tapa::mmap<const id_t> row_task_ptr,
		tapa::mmap<const ap_uint<32> > route,
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
		id_t command_count, id_t force_mode, id_t C_capacity,
		tapa::mmap<id_t> C_rcsr,
		tapa::mmap<ap_uint<512> > C_item0,
		tapa::mmap<ap_uint<512> > C_item1,
		tapa::mmap<ap_uint<512> > C_item2,
		tapa::mmap<ap_uint<512> > C_item3,
		tapa::mmap<id_t> stats) {
	// The graph is intentionally identical to the proven single-tile top.  Only
	// its dispatcher consumes a list and emits one uninterrupted row stream.
	// This keeps the large MERGE/DENSE memories single-instanced and avoids the
	// timing/resource failure mode seen when complete reducers were duplicated.
	tapa::stream<TapaRowHeader, 128> row_headers("row_headers");
	tapa::stream<TapaReaderCommand, 128> command0("command0");
	tapa::stream<TapaReaderCommand, 128> command1("command1");
	tapa::stream<TapaReaderCommand, 128> command2("command2");
	tapa::stream<TapaReaderCommand, 128> command3("command3");
	tapa::stream<TapaReaderCommand, 128> command4("command4");
	tapa::stream<TapaReaderCommand, 128> command5("command5");
	tapa::stream<TapaReaderCommand, 128> command6("command6");
	tapa::stream<TapaReaderCommand, 128> command7("command7");
	tapa::stream<TapaBReaderRawPacket, 4> raw_beat0("raw_beat0");
	tapa::stream<TapaBReaderRawPacket, 4> raw_beat1("raw_beat1");
	tapa::stream<TapaBReaderRawPacket, 4> raw_beat2("raw_beat2");
	tapa::stream<TapaBReaderRawPacket, 4> raw_beat3("raw_beat3");
	tapa::stream<TapaBReaderRawPacket, 4> raw_beat4("raw_beat4");
	tapa::stream<TapaBReaderRawPacket, 4> raw_beat5("raw_beat5");
	tapa::stream<TapaBReaderRawPacket, 4> raw_beat6("raw_beat6");
	tapa::stream<TapaBReaderRawPacket, 4> raw_beat7("raw_beat7");
	tapa::stream<TapaProductPacket, 128> raw_product0("raw_product0");
	tapa::stream<TapaProductPacket, 128> raw_product1("raw_product1");
	tapa::stream<TapaProductPacket, 128> raw_product2("raw_product2");
	tapa::stream<TapaProductPacket, 128> raw_product3("raw_product3");
	tapa::stream<TapaProductPacket, 128> raw_product4("raw_product4");
	tapa::stream<TapaProductPacket, 128> raw_product5("raw_product5");
	tapa::stream<TapaProductPacket, 128> raw_product6("raw_product6");
	tapa::stream<TapaProductPacket, 128> raw_product7("raw_product7");
	tapa::stream<TapaProductPacket, 128> shard_product0("shard_product0");
	tapa::stream<TapaProductPacket, 128> shard_product1("shard_product1");
	tapa::stream<TapaProductPacket, 128> shard_product2("shard_product2");
	tapa::stream<TapaProductPacket, 128> shard_product3("shard_product3");
	tapa::stream<TapaProductPacket, 128> shard_product4("shard_product4");
	tapa::stream<TapaProductPacket, 128> shard_product5("shard_product5");
	tapa::stream<TapaProductPacket, 128> shard_product6("shard_product6");
	tapa::stream<TapaProductPacket, 128> shard_product7("shard_product7");
	tapa::stream<TapaProductPacket, 4> product0("product0");
	tapa::stream<TapaProductPacket, 4> product1("product1");
	tapa::stream<TapaProductPacket, 4> product2("product2");
	tapa::stream<TapaProductPacket, 4> product3("product3");
	tapa::stream<TapaProductPacket, 4> product4("product4");
	tapa::stream<TapaProductPacket, 4> product5("product5");
	tapa::stream<TapaProductPacket, 4> product6("product6");
	tapa::stream<TapaProductPacket, 4> product7("product7");
	tapa::stream<TapaProductPacket, 16> reduce_product0("reduce_product0");
	tapa::stream<TapaProductPacket, 16> reduce_product1("reduce_product1");
	tapa::stream<TapaProductPacket, 16> reduce_product2("reduce_product2");
	tapa::stream<TapaProductPacket, 16> reduce_product3("reduce_product3");
	tapa::stream<TapaProductPacket, 16> reduce_product4("reduce_product4");
	tapa::stream<TapaProductPacket, 16> reduce_product5("reduce_product5");
	tapa::stream<TapaProductPacket, 16> reduce_product6("reduce_product6");
	tapa::stream<TapaProductPacket, 16> reduce_product7("reduce_product7");
	tapa::stream<TapaMergeResultPacket, 32> merge_results("merge_results");
	tapa::streams<TapaBankedMergeItem, 8, 16> merge_source("merge_source");
	tapa::streams<TapaBankedMergeItem, 16, 16> merge_split0("merge_split0");
	tapa::streams<TapaBankedMergeItem, 8, 16> merge_stage0("merge_stage0");
	tapa::streams<TapaBankedMergeItem, 16, 16> merge_split1("merge_split1");
	tapa::streams<TapaBankedMergeItem, 8, 16> merge_stage1("merge_stage1");
	tapa::streams<TapaBankedMergeItem, 16, 16> merge_split2("merge_split2");
	tapa::streams<TapaBankedMergeItem, 8, 16> merge_stage2("merge_stage2");
	tapa::streams<TapaBankedMergeItem, 8, 32> merge_reduced("merge_reduced");
	tapa::stream<TapaMergeSparsePacket, 32> merge_sparse("merge_sparse");
	tapa::stream<TapaMergeCompactPacket, 32> merge_compact("merge_compact");
	tapa::stream<TapaRowLength, 128> row_lengths("row_lengths");
	tapa::stream<TapaOutputPacket, 128> output_packets("output_packets");
	tapa::stream<id_t, ADAPT_FP32_STAT_COUNT> stat_values("stat_values");

	tapa::task()
		.invoke(tapa_dispatch_commands, tile_commands,
			task_words, row_task_ptr, route, command_count, force_mode,
			row_headers, command0, command1, command2, command3,
			command4, command5, command6, command7)
		.invoke(tapa_b_reader_fetch, B0, B0_beats, command0, raw_beat0)
		.invoke(tapa_b_reader_fetch, B1, B1_beats, command1, raw_beat1)
		.invoke(tapa_b_reader_fetch, B2, B2_beats, command2, raw_beat2)
		.invoke(tapa_b_reader_fetch, B3, B3_beats, command3, raw_beat3)
		.invoke(tapa_b_reader_fetch, B4, B4_beats, command4, raw_beat4)
		.invoke(tapa_b_reader_fetch, B5, B5_beats, command5, raw_beat5)
		.invoke(tapa_b_reader_fetch, B6, B6_beats, command6, raw_beat6)
		.invoke(tapa_b_reader_fetch, B7, B7_beats, command7, raw_beat7)
		.invoke(tapa_b_reader_scale, raw_beat0, raw_product0)
		.invoke(tapa_b_reader_scale, raw_beat1, raw_product1)
		.invoke(tapa_b_reader_scale, raw_beat2, raw_product2)
		.invoke(tapa_b_reader_scale, raw_beat3, raw_product3)
		.invoke(tapa_b_reader_scale, raw_beat4, raw_product4)
		.invoke(tapa_b_reader_scale, raw_beat5, raw_product5)
		.invoke(tapa_b_reader_scale, raw_beat6, raw_product6)
		.invoke(tapa_b_reader_scale, raw_beat7, raw_product7)
		.invoke(tapa_shard_local_merge, raw_product0, shard_product0)
		.invoke(tapa_shard_local_merge, raw_product1, shard_product1)
		.invoke(tapa_shard_local_merge, raw_product2, shard_product2)
		.invoke(tapa_shard_local_merge, raw_product3, shard_product3)
		.invoke(tapa_shard_local_merge, raw_product4, shard_product4)
		.invoke(tapa_shard_local_merge, raw_product5, shard_product5)
		.invoke(tapa_shard_local_merge, raw_product6, shard_product6)
		.invoke(tapa_shard_local_merge, raw_product7, shard_product7)
		.invoke(tapa_product_register_slice, shard_product0, product0)
		.invoke(tapa_product_register_slice, shard_product1, product1)
		.invoke(tapa_product_register_slice, shard_product2, product2)
		.invoke(tapa_product_register_slice, shard_product3, product3)
		.invoke(tapa_product_register_slice, shard_product4, product4)
		.invoke(tapa_product_register_slice, shard_product5, product5)
		.invoke(tapa_product_register_slice, shard_product6, product6)
		.invoke(tapa_product_register_slice, shard_product7, product7)
		.invoke(tapa_merge_route_source, product0, reduce_product0, merge_source[0])
		.invoke(tapa_merge_route_source, product1, reduce_product1, merge_source[1])
		.invoke(tapa_merge_route_source, product2, reduce_product2, merge_source[2])
		.invoke(tapa_merge_route_source, product3, reduce_product3, merge_source[3])
		.invoke(tapa_merge_route_source, product4, reduce_product4, merge_source[4])
		.invoke(tapa_merge_route_source, product5, reduce_product5, merge_source[5])
		.invoke(tapa_merge_route_source, product6, reduce_product6, merge_source[6])
		.invoke(tapa_merge_route_source, product7, reduce_product7, merge_source[7])
		TAPA_INVOKE_FLEX_SPLIT0(0) TAPA_INVOKE_FLEX_SPLIT0(1)
		TAPA_INVOKE_FLEX_SPLIT0(2) TAPA_INVOKE_FLEX_SPLIT0(3)
		TAPA_INVOKE_FLEX_SPLIT0(4) TAPA_INVOKE_FLEX_SPLIT0(5)
		TAPA_INVOKE_FLEX_SPLIT0(6) TAPA_INVOKE_FLEX_SPLIT0(7)
		TAPA_INVOKE_FLEX_MERGE0(0, 2, 0) TAPA_INVOKE_FLEX_MERGE0(1, 3, 1)
		TAPA_INVOKE_FLEX_MERGE0(4, 6, 2) TAPA_INVOKE_FLEX_MERGE0(5, 7, 3)
		TAPA_INVOKE_FLEX_MERGE0(8, 10, 4) TAPA_INVOKE_FLEX_MERGE0(9, 11, 5)
		TAPA_INVOKE_FLEX_MERGE0(12, 14, 6) TAPA_INVOKE_FLEX_MERGE0(13, 15, 7)
		TAPA_INVOKE_FLEX_SPLIT1(0) TAPA_INVOKE_FLEX_SPLIT1(1)
		TAPA_INVOKE_FLEX_SPLIT1(2) TAPA_INVOKE_FLEX_SPLIT1(3)
		TAPA_INVOKE_FLEX_SPLIT1(4) TAPA_INVOKE_FLEX_SPLIT1(5)
		TAPA_INVOKE_FLEX_SPLIT1(6) TAPA_INVOKE_FLEX_SPLIT1(7)
		TAPA_INVOKE_FLEX_MERGE1(0, 4, 0) TAPA_INVOKE_FLEX_MERGE1(2, 6, 1)
		TAPA_INVOKE_FLEX_MERGE1(1, 5, 2) TAPA_INVOKE_FLEX_MERGE1(3, 7, 3)
		TAPA_INVOKE_FLEX_MERGE1(8, 12, 4) TAPA_INVOKE_FLEX_MERGE1(10, 14, 5)
		TAPA_INVOKE_FLEX_MERGE1(9, 13, 6) TAPA_INVOKE_FLEX_MERGE1(11, 15, 7)
		TAPA_INVOKE_FLEX_SPLIT2(0) TAPA_INVOKE_FLEX_SPLIT2(1)
		TAPA_INVOKE_FLEX_SPLIT2(2) TAPA_INVOKE_FLEX_SPLIT2(3)
		TAPA_INVOKE_FLEX_SPLIT2(4) TAPA_INVOKE_FLEX_SPLIT2(5)
		TAPA_INVOKE_FLEX_SPLIT2(6) TAPA_INVOKE_FLEX_SPLIT2(7)
		TAPA_INVOKE_FLEX_MERGE2(0, 8, 0) TAPA_INVOKE_FLEX_MERGE2(2, 10, 1)
		TAPA_INVOKE_FLEX_MERGE2(4, 12, 2) TAPA_INVOKE_FLEX_MERGE2(6, 14, 3)
		TAPA_INVOKE_FLEX_MERGE2(1, 9, 4) TAPA_INVOKE_FLEX_MERGE2(3, 11, 5)
		TAPA_INVOKE_FLEX_MERGE2(5, 13, 6) TAPA_INVOKE_FLEX_MERGE2(7, 15, 7)
		TAPA_INVOKE_FLEX_FILTER(0) TAPA_INVOKE_FLEX_FILTER(1)
		TAPA_INVOKE_FLEX_FILTER(2) TAPA_INVOKE_FLEX_FILTER(3)
		TAPA_INVOKE_FLEX_FILTER(4) TAPA_INVOKE_FLEX_FILTER(5)
		TAPA_INVOKE_FLEX_FILTER(6) TAPA_INVOKE_FLEX_FILTER(7)
		.invoke(tapa_merge_ordered_select,
			merge_reduced[0], merge_reduced[1], merge_reduced[2], merge_reduced[3],
			merge_reduced[4], merge_reduced[5], merge_reduced[6], merge_reduced[7],
			merge_sparse)
		.invoke(tapa_merge_ordered_compact, merge_sparse, merge_compact)
		.invoke(tapa_merge_ordered_emit, merge_compact, merge_results)
		.invoke(tapa_adaptive_reduce, row_headers, merge_results,
			reduce_product0, reduce_product1, reduce_product2, reduce_product3,
			reduce_product4, reduce_product5, reduce_product6, reduce_product7,
			0, 0, row_lengths, output_packets, stat_values)
		.invoke(tapa_write_csr, row_lengths, output_packets, stat_values,
			C_capacity, C_rcsr, C_item0, C_item1, C_item2, C_item3, stats);
}
