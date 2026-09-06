#ifndef ADAPTIVE_HBM_TAPA_H
#define ADAPTIVE_HBM_TAPA_H

#include <ap_int.h>
#include <tapa.h>

#include <cstdint>

// Keep this header independent of the legacy core.h/hls::stream type graph.
// These values are the deployed wire ABI shared with adaptive_spgemm_fp32.h.
using id_t = std::uint32_t;

#ifndef ADAPTIVE_SPGEMM_FP32_H
// Physical direct-address workspace.  This is a row-local hardware capacity,
// not a supported-matrix-width limit.  The next row-persistent top decouples
// global 32-bit columns from this 16-bit local DENSE address space.
#define ADAPT_FP32_DENSE_CAPACITY 65536
enum AdaptiveFp32ForceMode {
	ADAPT_FP32_USE_ROUTE = 0,
	ADAPT_FP32_FORCE_MERGE = 1,
	ADAPT_FP32_FORCE_DENSE = 2
};
enum AdaptiveFp32RouteCode {
	ADAPT_FP32_EMPTY = 0,
	ADAPT_FP32_DIRECT = 1,
	ADAPT_FP32_MERGE = 2,
	ADAPT_FP32_DENSE = 3
};
enum AdaptiveFp32StatIndex {
	ADAPT_FP32_STAT_ROWS = 0,
	ADAPT_FP32_STAT_EMPTY_ROWS = 1,
	ADAPT_FP32_STAT_DIRECT_ROWS = 2,
	ADAPT_FP32_STAT_MERGE_ROWS = 3,
	ADAPT_FP32_STAT_DENSE_ROWS = 4,
	ADAPT_FP32_STAT_PARTIAL_PRODUCTS = 5,
	ADAPT_FP32_STAT_MERGE_TRANSACTIONS = 6,
	ADAPT_FP32_STAT_BANK_REPLAY_CYCLES = 7,
	ADAPT_FP32_STAT_RAW_DEPENDENCIES = 8,
	ADAPT_FP32_STAT_OUTPUT_NNZ = 9,
	ADAPT_FP32_STAT_INPUT_OVERFLOW_ROWS = 10,
	ADAPT_FP32_STAT_OUTPUT_OVERFLOW_ROWS = 11,
	ADAPT_FP32_STAT_DENSE_ROUTE_BEATS = 12,
	// The first 13 entries above are the deployed legacy ABI.  Keep their
	// indices fixed; the entries below are an append-only TAPA diagnostic ABI.
	ADAPT_FP32_STAT_DIAGNOSTIC_VERSION = 13,
	ADAPT_FP32_STAT_DENSE_COLLECT_CYCLES = 14,
	ADAPT_FP32_STAT_DENSE_EPOCH_CLEAR_CYCLES = 15,
	ADAPT_FP32_STAT_DENSE_BANK0_PRODUCTS = 16,
	ADAPT_FP32_STAT_DENSE_BANK1_PRODUCTS = 17,
	ADAPT_FP32_STAT_DENSE_BANK2_PRODUCTS = 18,
	ADAPT_FP32_STAT_DENSE_BANK3_PRODUCTS = 19,
	ADAPT_FP32_STAT_DENSE_BANK4_PRODUCTS = 20,
	ADAPT_FP32_STAT_DENSE_BANK5_PRODUCTS = 21,
	ADAPT_FP32_STAT_DENSE_BANK6_PRODUCTS = 22,
	ADAPT_FP32_STAT_DENSE_BANK7_PRODUCTS = 23,
	ADAPT_FP32_STAT_DENSE_BANK0_UNIQUE_ADDRESSES = 24,
	ADAPT_FP32_STAT_DENSE_BANK1_UNIQUE_ADDRESSES = 25,
	ADAPT_FP32_STAT_DENSE_BANK2_UNIQUE_ADDRESSES = 26,
	ADAPT_FP32_STAT_DENSE_BANK3_UNIQUE_ADDRESSES = 27,
	ADAPT_FP32_STAT_DENSE_BANK4_UNIQUE_ADDRESSES = 28,
	ADAPT_FP32_STAT_DENSE_BANK5_UNIQUE_ADDRESSES = 29,
	ADAPT_FP32_STAT_DENSE_BANK6_UNIQUE_ADDRESSES = 30,
	ADAPT_FP32_STAT_DENSE_BANK7_UNIQUE_ADDRESSES = 31,
	ADAPT_FP32_STAT_DENSE_TOUCHED_ADDRESSES = 32,
	ADAPT_FP32_STAT_DENSE_BITMAP_CLEAR_CYCLES = 33,
	ADAPT_FP32_STAT_DENSE_BITMAP_BUILD_CYCLES = 34,
	ADAPT_FP32_STAT_DENSE_BITMAP_WORD_SCANS = 35,
	ADAPT_FP32_STAT_DENSE_NONEMPTY_WORDS = 36,
	ADAPT_FP32_STAT_DENSE_ACTIVE_ADDRESS_GROUPS = 37,
	ADAPT_FP32_STAT_DENSE_CANDIDATE_COLUMNS = 38,
	ADAPT_FP32_STAT_DENSE_EXTRACT_ITERATIONS = 39,
	ADAPT_FP32_STAT_DENSE_OUTPUT_PACKETS = 40,
	ADAPT_FP32_STAT_MERGE_STREAM_CYCLES = 41,
	ADAPT_FP32_STAT_DENSE_COALESCED_PRODUCTS = 42,
	ADAPT_FP32_STAT_COUNT = 43
};

constexpr id_t ADAPT_FP32_DIAGNOSTIC_VERSION = 0x444e5331U; // "DNS1"
#endif

// TAPA replacement for the deployed Vitis-HLS unified kernel.  The argument
// order intentionally stays identical so the existing XRT Host-side packing
// and selector remain usable after the new xclbin is linked.
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
		tapa::mmap<id_t> stats);

// Persistent command-list top.  DMC1 commands retain the verified legacy
// dimension-window compatibility ABI.  RPC1 commands describe one execution
// fragment of a complete logical output row; the Host chooses MERGE or DENSE
// once for the complete row and every capacity fragment inherits that mode.
//
// Tk is deliberately NOT a command dimension.  Every command contains the
// complete K contribution for its output tile.  A Gustavson MERGE row must see
// all of its sorted input runs in one reduction; splitting K would materialize
// partial sparse rows and require a second merge pass.  If the B working set
// becomes limiting, narrow the internal address page instead of introducing a
// Host-visible K tile.
//
// The Host concatenates the tile-local task,
// row-pointer, and route arrays into capacity-safe HBM super-batches; the
// kernel consumes every command in the super-batch without returning to XRT.
//
// Legacy DMC1 command word ABI (32-bit lanes):
//   0 task descriptor base, 1 row-pointer base, 2 route base,
//   3 tile rows (Tm), 4 tile columns (Tn), 5 virtual row base,
//   6 source N-window id, 7 source global row begin,
//   15 magic 0x444d4331 ("DMC1").
// Row-persistent RPC1 uses lanes 0..5 identically, lane 6 as dense_base and
// lane 7 bit 0 as logical_row_last; lane 15 contains magic 0x52504331.  RPC1
// commands currently contain one physical fragment.  Multiple fragments of
// the same logical row are ordered by global column range, use the same route,
// and only the final fragment advances C_rcsr.  Thus fragmentation is a
// storage/capacity mechanism and never a selector decision unit.
// B offsets embedded in task descriptors are already absolute within the
// concatenated B0..B7 buffers.  Output rows are emitted in command order into
// one concatenated CSR payload; different M/N tiles therefore never require a
// cross-command numerical reduction.
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
		tapa::mmap<id_t> stats);

#endif
