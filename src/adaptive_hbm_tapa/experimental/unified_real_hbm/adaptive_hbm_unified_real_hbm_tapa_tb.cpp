#include "adaptive_hbm_unified_real_hbm_tapa.h"
#include "../../adaptive_hbm_tapa_scalable.h"
#include "../unified_adaptive_full_output/adaptive_hbm_unified_adaptive_full_output_tapa.h"

#include <cmath>
#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <utility>
#include <vector>

void merge15_csim_set_writer_delay_us(unsigned delay_us);

namespace {

using Key = std::pair<uint32_t, uint32_t>;

static ap_uint<32> raw(float value) {
	union { float f; uint32_t u; } bits;
	bits.f = value;
	return bits.u;
}

static float fp(ap_uint<32> value) {
	union { float f; uint32_t u; } bits;
	bits.u = (uint32_t)value;
	return bits.f;
}

static ap_uint<64> item(uint32_t column, float value) {
	ap_uint<64> packed = 0;
	packed.range(31, 0) = column;
	packed.range(63, 32) = raw(value);
	return packed;
}

static ap_uint<512> beat(
		const std::vector<std::pair<uint32_t, float> >& values) {
	ap_uint<512> packed = 0;
	for (unsigned lane = 0; lane < values.size(); ++lane)
		packed.range(lane * 64 + 63, lane * 64)
			= item(values[lane].first, values[lane].second);
	return packed;
}

static ap_uint<128> descriptor(uint32_t offset, uint32_t length,
		unsigned shard, float scale) {
	ap_uint<128> value = 0;
	value.range(31, 0) = offset;
	value.range(55, 32) = length;
	value.range(58, 56) = shard;
	value.range(127, 96) = raw(scale);
	return value;
}

}  // namespace

int main(int argc, char** argv) {
	const char* bitstream = argc > 1 ? argv[1] : "";
	std::vector<ap_uint<512> > B[8];
	std::vector<ap_uint<128> > tasks;
	std::vector<id_t> row_ptr = {0};
	std::vector<ap_uint<32> > routes;
	std::vector<ap_uint<32> > masks;
	std::vector<ap_uint<64> > geometry;
	std::vector<id_t> logical_rows;
	std::map<Key, float> expected;
	std::map<uint32_t, unsigned> expected_nnz;
	std::map<uint32_t, unsigned> expected_route;
	std::set<uint32_t> expected_wide_rows;

	auto add_b_task = [&](unsigned shard,
			const std::vector<std::pair<uint32_t, float> >& values,
			float scale) {
		const uint32_t offset = B[shard].size();
		B[shard].push_back(beat(values));
		tasks.push_back(descriptor(offset, values.size(), shard, scale));
	};
	auto add_profile = [&](unsigned route, unsigned products,
			unsigned source_mask, uint32_t base, uint32_t span) {
		logical_rows.push_back(logical_rows.size());
		routes.push_back((ap_uint<32>(products) << 2) | route);
		masks.push_back(source_mask);
		geometry.push_back((ap_uint<64>(span) << 32) | base);
		row_ptr.push_back(tasks.size());
	};

	// Ordinary MERGE: two real HBM shards, duplicate column 1001.
	add_b_task(0, {{1001, 1.0f}, {1003, 2.0f}}, 2.0f);
	add_b_task(1, {{1001, 3.0f}, {1002, 4.0f}}, 1.0f);
	add_profile(ADAPT_FP32_MERGE, 4, 0x03, 1000, 32);
	expected[{0, 1001}] = 5.0f;
	expected[{0, 1002}] = 4.0f;
	expected[{0, 1003}] = 4.0f;
	expected_nnz[0] = 3;
	expected_route[0] = UNIFIED_FULL_OUTPUT_MERGE_ROW;

	// Heavy MERGE: the Host route is still MERGE; only the row-product count
	// selects the internal vector microengine.
	add_b_task(0, {{2001, 1.0f}, {2002, 1.0f}, {2003, 1.0f},
		{2004, 1.0f}, {2005, 1.0f}, {2006, 1.0f}, {2007, 1.0f},
		{2008, 1.0f}}, 1.0f);
	add_b_task(1, {{2004, 2.0f}, {2005, 2.0f}, {2006, 2.0f},
		{2007, 2.0f}, {2008, 2.0f}, {2009, 2.0f}, {2010, 2.0f},
		{2011, 2.0f}}, 1.0f);
	add_profile(ADAPT_FP32_MERGE, 16, 0x03, 2000, 32);
	for (uint32_t col = 2001; col <= 2011; ++col)
		expected[{1, col}] = col < 2004 ? 1.0f
			: col <= 2008 ? 3.0f : 2.0f;
	expected_nnz[1] = 11;
	expected_route[1] = UNIFIED_FULL_OUTPUT_MERGE_ROW;

	// DENSE: same-bank traffic plus an exact cancellation before extraction.
	add_b_task(2, {{3005, 1.0f}, {3009, 2.0f}, {3013, 3.0f}}, 1.0f);
	add_b_task(3, {{3005, 4.0f}, {3009, -2.0f}, {3014, 6.0f}}, 1.0f);
	add_profile(ADAPT_FP32_DENSE, 6, 0x0c, 3000, 64);
	expected[{2, 3005}] = 5.0f;
	expected[{2, 3013}] = 3.0f;
	expected[{2, 3014}] = 6.0f;
	expected_nnz[2] = 3;
	expected_route[2] = UNIFIED_FULL_OUTPUT_DENSE;

	// Wide DENSE: one logical row spans both 32K physical contexts.  The
	// duplicated high-half key cancels before extraction.
	add_b_task(4, {{4005, 1.0f}, {36773, 2.0f}, {36774, 3.0f}}, 1.0f);
	add_b_task(5, {{4005, 4.0f}, {36773, -2.0f}, {36775, 6.0f}}, 1.0f);
	add_profile(ADAPT_FP32_DENSE, 6, 0x30, 4000, 32776);
	expected[{3, 4005}] = 5.0f;
	expected[{3, 36774}] = 3.0f;
	expected[{3, 36775}] = 6.0f;
	expected_nnz[3] = 3;
	expected_route[3] = UNIFIED_FULL_OUTPUT_DENSE;
	expected_wide_rows.insert(3);

	// Zero tasks/source mask: command reader creates only a phantom EOR.
	add_profile(ADAPT_FP32_EMPTY, 0, 0x00, 0, 1);
	expected_nnz[4] = 0;
	expected_route[4] = UNIFIED_FULL_OUTPUT_MERGE_ROW;

	// A second disjoint <=64K command belongs to logical row zero.  This is
	// the extreme-span fallback: the Host copies the complete row's MERGE route
	// to both fragments, and one FPGA invocation emits two tagged EORs for the
	// same logical row without truncating the 32-bit global column.
	add_b_task(6, {{70001, 7.0f}}, 1.0f);
	add_profile(ADAPT_FP32_MERGE, 1, 0x40, 65536, 10000);
	logical_rows.back() = 0;
	expected[{0, 70001}] = 7.0f;
	expected_nnz[0] = 4;

	// Exact regression for the routed-board deadlock: one shard owns far more
	// than the old four-entry central command FIFO could absorb before the row
	// boundaries were broadcast.  Independent embedded command readers must
	// complete this row without relying on an artificially deep on-chip FIFO.
	{
		const uint32_t row = logical_rows.size();
		const uint32_t base = 80000;
		for (unsigned command = 0; command < 64; ++command) {
			add_b_task(1, {{base + command, 1.0f}}, 1.0f);
			expected[{row, base + command}] = 1.0f;
		}
		for (unsigned shard = 0; shard < 8; ++shard) {
			if (shard == 1) continue;
			add_b_task(shard, {{base + 64 + shard, 1.0f}}, 1.0f);
			expected[{row, base + 64 + shard}] = 1.0f;
		}
		add_profile(ADAPT_FP32_DENSE, 71, 0xff, base, 128);
		expected_nnz[row] = 71;
		expected_route[row] = UNIFIED_FULL_OUTPUT_DENSE;
	}

	// Sustained real-matrix scheduler regression: the original integration
	// proof had only six physical fragments, while ex9 presents thousands.
	// Feed 4096 consecutive, all-eight-shard DENSE rows so both accumulator
	// contexts are repeatedly allocated and retired behind realistic reader
	// boundaries.  Each row uses distinct keys, which makes output accounting
	// exact without hiding a missing completion behind a duplicate reduction.
	for (uint32_t stress = 0; stress < 4096; ++stress) {
		const uint32_t row = logical_rows.size();
		const uint32_t base = 100000 + stress * 64;
		for (unsigned shard = 0; shard < 8; ++shard) {
			add_b_task(shard, {{base + shard, 1.0f}}, 1.0f);
			expected[{row, base + shard}] = 1.0f;
		}
		add_profile(ADAPT_FP32_DENSE, 8, 0xff, base, 64);
		expected_nnz[row] = 8;
		expected_route[row] = UNIFIED_FULL_OUTPUT_DENSE;
	}

	for (unsigned shard = 0; shard < 8; ++shard)
		if (B[shard].empty()) B[shard].resize(1);

	// Append the same per-shard command-tail ABI used by the production Host.
	const std::array<id_t, 8> B_data_beats = {
		(id_t)B[0].size(), (id_t)B[1].size(), (id_t)B[2].size(),
		(id_t)B[3].size(), (id_t)B[4].size(), (id_t)B[5].size(),
		(id_t)B[6].size(), (id_t)B[7].size()};
	const size_t header_words = (routes.size() + 15) / 16;
	std::array<std::vector<ap_uint<512> >, 8> command_tail;
	std::array<uint64_t, 8> command_counts{};
	for (unsigned shard = 0; shard < 8; ++shard)
		command_tail[shard].resize(header_words);
	for (size_t fragment = 0; fragment < routes.size(); ++fragment) {
		std::array<uint32_t, 8> counts{};
		for (id_t task = row_ptr[fragment]; task < row_ptr[fragment + 1]; ++task)
			++counts[(unsigned)tasks[task].range(58, 56)];
		for (unsigned shard = 0; shard < 8; ++shard) {
			const unsigned slot = fragment & 15U;
			command_tail[shard][fragment >> 4].range(
				slot * 32 + 31, slot * 32) = counts[shard];
			command_counts[shard] += counts[shard];
		}
	}
	for (unsigned shard = 0; shard < 8; ++shard)
		command_tail[shard].resize(header_words
			+ (command_counts[shard] + 3) / 4);
	std::array<uint64_t, 8> command_cursor{};
	for (size_t fragment = 0; fragment < routes.size(); ++fragment) {
		for (id_t task = row_ptr[fragment]; task < row_ptr[fragment + 1]; ++task) {
			const unsigned shard = (unsigned)tasks[task].range(58, 56);
			const uint64_t cursor = command_cursor[shard]++;
			command_tail[shard][header_words + (cursor >> 2)].range(
				(cursor & 3U) * 128 + 127, (cursor & 3U) * 128) = tasks[task];
		}
	}
	for (unsigned shard = 0; shard < 8; ++shard)
		B[shard].insert(B[shard].end(), command_tail[shard].begin(),
			command_tail[shard].end());

	// 4096 stress rows can concentrate their packet output in one physical
	// port, so size all diagnostic memories above that worst case.  This is a
	// CSim-only buffer bound, not an accelerator capacity change.
	constexpr id_t kCapacity = 8192;
	std::vector<ap_uint<512> > output[4];
	std::vector<id_t> port_stats[4];
	for (unsigned port = 0; port < 4; ++port) {
		output[port].resize(kCapacity);
		port_stats[port].resize(4);
	}
	std::vector<ap_uint<512> > metadata(kCapacity);
	std::vector<ap_uint<64> > completions(kCapacity);
	std::vector<id_t> meta_stats(4), completion_stats(2), scheduler_stats(16);
	std::vector<id_t> dense_stats(12), heavy_stats(8);

	// Force bounded packet/meta FIFOs to absorb realistic downstream write
	// pauses.  This hook exists only in the CSim build of the writer.
	merge15_csim_set_writer_delay_us(5);
	tapa::invoke(adaptive_hbm_unified_real_hbm_tapa, bitstream,
		tapa::read_only_mmap<const ap_uint<32> >(routes),
		tapa::read_only_mmap<const ap_uint<32> >(masks),
		tapa::read_only_mmap<const ap_uint<64> >(geometry),
		tapa::read_only_mmap<const id_t>(logical_rows),
		tapa::read_only_mmap<const ap_uint<512> >(B[0]),
		tapa::read_only_mmap<const ap_uint<512> >(B[1]),
		tapa::read_only_mmap<const ap_uint<512> >(B[2]),
		tapa::read_only_mmap<const ap_uint<512> >(B[3]),
		tapa::read_only_mmap<const ap_uint<512> >(B[4]),
		tapa::read_only_mmap<const ap_uint<512> >(B[5]),
		tapa::read_only_mmap<const ap_uint<512> >(B[6]),
		tapa::read_only_mmap<const ap_uint<512> >(B[7]),
		B_data_beats[0], B_data_beats[1], B_data_beats[2],
		B_data_beats[3], B_data_beats[4], B_data_beats[5],
		B_data_beats[6], B_data_beats[7],
		(id_t)routes.size(), (id_t)12, kCapacity, kCapacity, kCapacity,
		tapa::write_only_mmap<ap_uint<512> >(output[0]),
		tapa::write_only_mmap<ap_uint<512> >(output[1]),
		tapa::write_only_mmap<ap_uint<512> >(output[2]),
		tapa::write_only_mmap<ap_uint<512> >(output[3]),
		tapa::write_only_mmap<ap_uint<512> >(metadata),
		tapa::write_only_mmap<ap_uint<64> >(completions),
		tapa::write_only_mmap<id_t>(port_stats[0]),
		tapa::write_only_mmap<id_t>(port_stats[1]),
		tapa::write_only_mmap<id_t>(port_stats[2]),
		tapa::write_only_mmap<id_t>(port_stats[3]),
		tapa::write_only_mmap<id_t>(meta_stats),
		tapa::write_only_mmap<id_t>(completion_stats),
		tapa::write_only_mmap<id_t>(scheduler_stats),
		tapa::write_only_mmap<id_t>(dense_stats),
		tapa::write_only_mmap<id_t>(heavy_stats));
	merge15_csim_set_writer_delay_us(0);

	unsigned total_items = 0, total_eors = 0;
	for (unsigned port = 0; port < 4; ++port) {
		if (port_stats[port][3] != 0) return 1;
		total_items += port_stats[port][1];
		total_eors += port_stats[port][2];
	}
	const unsigned expected_eors = routes.size() + expected_wide_rows.size();
	if (total_items != expected.size() || total_eors != expected_eors
			|| completion_stats[0] != 0 || completion_stats[1] != 2) {
		std::printf("UNIFIED_REAL_HBM_CSIM: FAIL totals items=%u/%zu eors=%u/%u completions=%u\n",
			total_items, expected.size(), total_eors, expected_eors,
			(unsigned)completion_stats[0]);
		return 2;
	}

	std::map<Key, float> observed;
	std::map<uint32_t, unsigned> eors;
	std::map<uint32_t, unsigned> eor_nnz;
	unsigned consumed[4] = {};
	for (unsigned word = 0; word < meta_stats[0]; ++word) {
		for (unsigned slot = 0; slot < 4; ++slot) {
			const ap_uint<128> record
				= metadata[word].range(slot * 128 + 127, slot * 128);
			if (!record[127]) continue;
			const uint32_t row = record.range(31, 0);
			const unsigned valid = record.range(67, 64);
			const bool eor = record[68];
			const unsigned route_code = record.range(73, 72);
			const unsigned error = record.range(79, 78);
			const unsigned port = record.range(81, 80);
			if (!expected_nnz.count(row) || expected_route[row] != route_code
					|| error != 0 || port > 3) return 3;
			if (valid != 0) {
				const ap_uint<512> data = output[port][consumed[port]++];
				for (unsigned lane = 0; lane < valid; ++lane) {
					const ap_uint<64> value
						= data.range(lane * 64 + 63, lane * 64);
					observed[{row, (uint32_t)value.range(31, 0)}]
						= fp(value.range(63, 32));
				}
			}
			if (eor) {
				eor_nnz[row] += (unsigned)record.range(63, 32);
				++eors[row];
			}
		}
	}
	if (observed.size() != expected.size()) return 5;
	for (const auto& want : expected) {
		const auto got = observed.find(want.first);
		if (got == observed.end()
				|| std::fabs(got->second - want.second) > 1e-5f) return 6;
	}
	for (const auto& row : expected_nnz) {
		const unsigned fragment_eors = row.first == 0 ? 2u : 1u;
		if (eors[row.first] != fragment_eors
				+ (expected_wide_rows.count(row.first) ? 1u : 0u)
				|| eor_nnz[row.first] != row.second) return 7;
	}
	const unsigned expected_dense_fragments = 3 + 4096;
	const unsigned expected_dense_physical_units = 4 + 4096;
	if (scheduler_stats[0] != routes.size()
			|| scheduler_stats[1] != routes.size()
			|| scheduler_stats[3] != expected_dense_fragments
			|| heavy_stats[2] != 1
			|| dense_stats[0] + dense_stats[6] != expected_dense_physical_units
			|| dense_stats[5] != 0 || dense_stats[11] != 0) return 8;

	std::printf("UNIFIED_REAL_HBM_CSIM: PASS fragments=%zu logical_rows=%zu ordinary_merge=2 heavy_merge=1 dense=3 wide_dense=1 empty=1 cross_64k_row=1 imbalanced_shard_commands=64 items=%zu real_fetch=8 real_scale=8 real_shard_local_merge=8 postlocal_adapter=8 four_hbm=1\n",
		routes.size(), expected_nnz.size(), expected.size());
	return 0;
}
