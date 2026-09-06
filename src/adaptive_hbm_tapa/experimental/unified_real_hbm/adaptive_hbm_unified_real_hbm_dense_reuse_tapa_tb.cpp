#include "adaptive_hbm_unified_real_hbm_tapa.h"
#include "../../adaptive_hbm_tapa_scalable.h"
#include "../unified_adaptive_full_output/adaptive_hbm_unified_adaptive_full_output_tapa.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

static ap_uint<32> raw(float value) {
	union { float f; uint32_t u; } bits;
	bits.f = value;
	return bits.u;
}

static ap_uint<64> item(uint32_t column, float value) {
	ap_uint<64> packed = 0;
	packed.range(31, 0) = column;
	packed.range(63, 32) = raw(value);
	return packed;
}

static ap_uint<512> beat(uint32_t column, float value) {
	ap_uint<512> packed = 0;
	packed.range(63, 0) = item(column, value);
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

// Production-top regression for the first physical DENSE-slot reuse.  Two
// rows occupy slots 0 and 1; the third row must wait for a post-output
// completion and then reuse one slot without deadlock or loss.
int main(int argc, char** argv) {
	const char* bitstream = argc > 1 ? argv[1] : "";
	std::vector<ap_uint<512> > B[8];
	std::vector<ap_uint<128> > tasks;
	std::vector<id_t> row_ptr = {0};
	std::vector<ap_uint<32> > routes;
	std::vector<ap_uint<32> > masks;
	std::vector<ap_uint<64> > geometry;
	std::vector<id_t> logical_rows;

	for (unsigned row = 0; row < 3; ++row) {
		const unsigned shard = row;
		const uint32_t offset = B[shard].size();
		B[shard].push_back(beat(row, 1.0f));
		tasks.push_back(descriptor(offset, 1, shard, 1.0f));
		row_ptr.push_back(tasks.size());
		routes.push_back((ap_uint<32>(1) << 2) | ADAPT_FP32_DENSE);
		masks.push_back(ap_uint<32>(1) << shard);
		geometry.push_back((ap_uint<64>(8) << 32) | 0);
		logical_rows.push_back(row);
	}
	for (unsigned shard = 0; shard < 8; ++shard)
		if (B[shard].empty()) B[shard].resize(1);

	std::vector<ap_uint<512> > task_words(1);
	for (unsigned index = 0; index < tasks.size(); ++index)
		task_words[0].range(index * 128 + 127, index * 128) = tasks[index];

	constexpr id_t kCapacity = 64;
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

	tapa::invoke(adaptive_hbm_unified_real_hbm_tapa, bitstream,
		tapa::read_only_mmap<const ap_uint<512> >(task_words),
		tapa::read_only_mmap<const id_t>(row_ptr),
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
		(id_t)B[0].size(), (id_t)B[1].size(), (id_t)B[2].size(),
		(id_t)B[3].size(), (id_t)B[4].size(), (id_t)B[5].size(),
		(id_t)B[6].size(), (id_t)B[7].size(),
		(id_t)3, (id_t)4096, kCapacity, kCapacity, kCapacity,
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

	unsigned items = 0;
	unsigned eors = 0;
	for (unsigned port = 0; port < 4; ++port) {
		if (port_stats[port][3] != 0) return 1;
		items += port_stats[port][1];
		eors += port_stats[port][2];
	}
	if (items != 3 || eors != 3 || completion_stats[0] != 3) return 2;
	if (scheduler_stats[0] != 3 || scheduler_stats[1] != 3
			|| scheduler_stats[3] != 3 || scheduler_stats[6] != 2) return 3;
	if (dense_stats[0] + dense_stats[6] != 3
			|| dense_stats[5] != 0 || dense_stats[11] != 0) return 4;

	std::printf("UNIFIED_REAL_HBM_DENSE_REUSE: PASS rows=3 slots=2 "
		"slot_reuse=1 items=%u completions=%u\n",
		items, (unsigned)completion_stats[0]);
	return 0;
}
