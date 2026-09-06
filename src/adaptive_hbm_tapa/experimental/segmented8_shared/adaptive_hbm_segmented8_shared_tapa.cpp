#include "adaptive_hbm_segmented8_shared_tapa.h"

namespace {

struct Candidate {
	ap_uint<16> col;
	ap_uint<32> value;
	bool valid;
};

static uint32_t raw_float(float value) {
#pragma HLS INLINE
	union { float f; uint32_t u; } bits;
	bits.f = value;
	return bits.u;
}

static float from_raw(ap_uint<32> value) {
#pragma HLS INLINE
	union { float f; uint32_t u; } bits;
	bits.u = (uint32_t)value;
	return bits.f;
}

static bool nonzero(float value) {
#pragma HLS INLINE
	return (raw_float(value) & 0x7fffffffU) != 0;
}

static bool less_than(const Candidate& lhs, const Candidate& rhs) {
#pragma HLS INLINE
	if (lhs.valid != rhs.valid) return lhs.valid;
	return lhs.valid && lhs.col < rhs.col;
}

static void compare_swap(Candidate& lhs, Candidate& rhs, bool ascending) {
#pragma HLS INLINE
	const bool exchange = ascending ? less_than(rhs, lhs) : less_than(lhs, rhs);
	if (exchange) {
		const Candidate temporary = lhs;
		lhs = rhs;
		rhs = temporary;
	}
}

static void sort8(Candidate value[8]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=value complete
	compare_swap(value[0], value[1], true);
	compare_swap(value[2], value[3], false);
	compare_swap(value[4], value[5], true);
	compare_swap(value[6], value[7], false);
	compare_swap(value[0], value[2], true);
	compare_swap(value[1], value[3], true);
	compare_swap(value[4], value[6], false);
	compare_swap(value[5], value[7], false);
	compare_swap(value[0], value[1], true);
	compare_swap(value[2], value[3], true);
	compare_swap(value[4], value[5], false);
	compare_swap(value[6], value[7], false);
	compare_swap(value[0], value[4], true);
	compare_swap(value[1], value[5], true);
	compare_swap(value[2], value[6], true);
	compare_swap(value[3], value[7], true);
	compare_swap(value[0], value[2], true);
	compare_swap(value[1], value[3], true);
	compare_swap(value[4], value[6], true);
	compare_swap(value[5], value[7], true);
	compare_swap(value[0], value[1], true);
	compare_swap(value[2], value[3], true);
	compare_swap(value[4], value[5], true);
	compare_swap(value[6], value[7], true);
}

static ap_uint<4> popcount8(ap_uint<8> value) {
#pragma HLS INLINE
	const ap_uint<3> low = value[0] + value[1] + value[2] + value[3];
	const ap_uint<3> high = value[4] + value[5] + value[6] + value[7];
	return low + high;
}

static Segmented8SharedToken end_token() {
#pragma HLS INLINE
	Segmented8SharedToken token = 0;
	token[556] = 1;
	return token;
}

}  // namespace

void segmented8_shared_read(
		tapa::mmap<const ap_uint<512> > data_in,
		tapa::mmap<const ap_uint<64> > meta_in,
		id_t packet_count,
		tapa::ostream<Segmented8SharedToken>& output) {
	for (id_t packet = 0; packet < packet_count; ++packet) {
#pragma HLS PIPELINE II=1
		const ap_uint<64> meta = meta_in[packet];
		Segmented8SharedToken token = 0;
		token.range(511, 0) = data_in[packet];
		token.range(543, 512) = meta.range(31, 0);
		token.range(551, 544) = meta.range(39, 32);
		token.range(553, 552) = meta.range(41, 40);
		token[554] = meta[42];
		output.write(token);
	}
	output.write(end_token());
}

void segmented8_shared_reduce(
		tapa::istream<Segmented8SharedToken>& input,
		tapa::ostream<Segmented8SharedToken>& output) {
	bool previous_valid[2] = {false, false};
	id_t previous_row[2] = {0, 0};
	ap_uint<16> previous_last_col[2] = {0, 0};
#pragma HLS ARRAY_PARTITION variable=previous_valid complete
#pragma HLS ARRAY_PARTITION variable=previous_row complete
#pragma HLS ARRAY_PARTITION variable=previous_last_col complete
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1048576 avg=4096
		Segmented8SharedToken source;
		// A blocking read becomes a global enable for this 16-stage pipeline.
		// An allocator may wait for the current row's EOR before it sends the
		// next row, so an empty input must insert a bubble instead of freezing
		// EOR and the preceding packet-local reductions in flight.
		if (!input.try_read(source)) continue;
		if (source[556]) {
			output.write(source);
			done = true;
			continue;
		}
		const id_t row = source.range(543, 512);
		const ap_uint<2> route = source.range(553, 552);
		const unsigned route_slot = route == SEGMENTED8_SHARED_DENSE ? 1 : 0;
		const bool eor = source[554];
		Segmented8SharedToken result = source;
		result.range(511, 0) = 0;
		result.range(551, 544) = 0;
		result[555] = 0;
		if (eor) {
			previous_valid[route_slot] = false;
			output.write(result);
			continue;
		}

		Candidate sorted[8];
		float sum0[8], sum1[8], sum2[8], sum3[8];
		bool head0[8], head1[8], head2[8];
#pragma HLS ARRAY_PARTITION variable=sorted complete
#pragma HLS ARRAY_PARTITION variable=sum0 complete
#pragma HLS ARRAY_PARTITION variable=sum1 complete
#pragma HLS ARRAY_PARTITION variable=sum2 complete
#pragma HLS ARRAY_PARTITION variable=sum3 complete
#pragma HLS ARRAY_PARTITION variable=head0 complete
#pragma HLS ARRAY_PARTITION variable=head1 complete
#pragma HLS ARRAY_PARTITION variable=head2 complete
		const ap_uint<8> source_mask = source.range(551, 544);
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			const ap_uint<64> item
				= source.range(lane * 64 + 63, lane * 64);
			sorted[lane].col = item.range(15, 0);
			sorted[lane].value = item.range(63, 32);
			sorted[lane].valid = source_mask[lane];
		}
		sort8(sorted);
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			sum0[lane] = from_raw(sorted[lane].value);
			head0[lane] = lane == 0 || !sorted[lane].valid
				|| !sorted[lane - 1].valid
				|| sorted[lane].col != sorted[lane - 1].col;
		}
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			if (lane >= 1 && sorted[lane].valid && !head0[lane]) {
				sum1[lane] = sum0[lane] + sum0[lane - 1];
				head1[lane] = head0[lane] || head0[lane - 1];
			} else { sum1[lane] = sum0[lane]; head1[lane] = head0[lane]; }
		}
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			if (lane >= 2 && sorted[lane].valid && !head1[lane]) {
				sum2[lane] = sum1[lane] + sum1[lane - 2];
				head2[lane] = head1[lane] || head1[lane - 2];
			} else { sum2[lane] = sum1[lane]; head2[lane] = head1[lane]; }
		}
		ap_uint<8> emitted_mask = 0;
		ap_uint<512> emitted_items = 0;
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			sum3[lane] = lane >= 4 && sorted[lane].valid && !head2[lane]
				? sum2[lane] + sum2[lane - 4] : sum2[lane];
			const bool group_end = sorted[lane].valid
				&& (lane == 7 || !sorted[lane + 1].valid
					|| sorted[lane].col != sorted[lane + 1].col);
			if (group_end && nonzero(sum3[lane])) {
				ap_uint<64> item = 0;
				item.range(31, 0) = sorted[lane].col;
				item.range(63, 32) = raw_float(sum3[lane]);
				emitted_items.range(lane * 64 + 63, lane * 64) = item;
				emitted_mask[lane] = 1;
			}
		}

		bool have_first = false;
		ap_uint<16> first_col = 0;
		ap_uint<16> last_col = 0;
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			if (emitted_mask[lane]) {
				const ap_uint<16> col
					= emitted_items.range(lane * 64 + 15, lane * 64);
				if (!have_first) { first_col = col; have_first = true; }
				last_col = col;
			}
		}
		result.range(511, 0) = emitted_items;
		result.range(551, 544) = emitted_mask;
		result[555] = have_first && previous_valid[route_slot]
			&& previous_row[route_slot] == row
			&& previous_last_col[route_slot] == first_col;
		if (have_first) {
			previous_valid[route_slot] = true;
			previous_row[route_slot] = row;
			previous_last_col[route_slot] = last_col;
		}
		output.write(result);
	}
}

void segmented8_shared_route(
		tapa::istream<Segmented8SharedToken>& input,
		tapa::ostream<Segmented8SharedToken>& merge_output,
		tapa::ostream<Segmented8SharedToken>& dense_output) {
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		const Segmented8SharedToken token = input.read();
		if (token[556]) {
			merge_output.write(token);
			dense_output.write(token);
			done = true;
		} else if (token.range(553, 552) == SEGMENTED8_SHARED_DENSE) {
			dense_output.write(token);
		} else {
			merge_output.write(token);
		}
	}
}

void segmented8_shared_write(
		tapa::istream<Segmented8SharedToken>& input,
		tapa::mmap<ap_uint<512> > data_out,
		tapa::mmap<ap_uint<64> > meta_out,
		tapa::mmap<id_t> stats_out) {
	id_t packets = 0;
	id_t items = 0;
	id_t eors = 0;
	id_t boundaries = 0;
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		const Segmented8SharedToken token = input.read();
		if (token[556]) {
			done = true;
			continue;
		}
		data_out[packets] = token.range(511, 0);
		ap_uint<64> meta = 0;
		meta.range(31, 0) = token.range(543, 512);
		meta.range(39, 32) = token.range(551, 544);
		meta.range(41, 40) = token.range(553, 552);
		meta[42] = token[554];
		meta[43] = token[555];
		meta_out[packets] = meta;
		items += popcount8(token.range(551, 544));
		eors += token[554];
		boundaries += token[555];
		++packets;
	}
	stats_out[0] = packets;
	stats_out[1] = items;
	stats_out[2] = eors;
	stats_out[3] = boundaries;
}

void adaptive_hbm_segmented8_shared_tapa(
		tapa::mmap<const ap_uint<512> > data_in,
		tapa::mmap<const ap_uint<64> > meta_in,
		id_t packet_count,
		tapa::mmap<ap_uint<512> > merge_data_out,
		tapa::mmap<ap_uint<64> > merge_meta_out,
		tapa::mmap<id_t> merge_stats_out,
		tapa::mmap<ap_uint<512> > dense_data_out,
		tapa::mmap<ap_uint<64> > dense_meta_out,
		tapa::mmap<id_t> dense_stats_out) {
	tapa::stream<Segmented8SharedToken, 8> raw("raw");
	tapa::stream<Segmented8SharedToken, 8> reduced("reduced");
	tapa::stream<Segmented8SharedToken, 8> merge("merge");
	tapa::stream<Segmented8SharedToken, 8> dense("dense");
	tapa::task()
		.invoke(segmented8_shared_read, data_in, meta_in, packet_count, raw)
		.invoke(segmented8_shared_reduce, raw, reduced)
		.invoke(segmented8_shared_route, reduced, merge, dense)
		.invoke(segmented8_shared_write, merge, merge_data_out,
			merge_meta_out, merge_stats_out)
		.invoke(segmented8_shared_write, dense, dense_data_out,
			dense_meta_out, dense_stats_out);
}
