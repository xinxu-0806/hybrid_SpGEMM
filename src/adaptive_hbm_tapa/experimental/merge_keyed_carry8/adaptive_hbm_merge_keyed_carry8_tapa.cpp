#include "adaptive_hbm_merge_keyed_carry8_tapa.h"

namespace {

// Work packet layout:
//   direct items[511:0], direct mask[519:512], phase sums[775:520],
//   phase mask[783:776], base FP32 value[815:784],
//   boundary FP32 contribution[847:816], boundary valid[848],
//   carry column[880:849], carry valid[881], row[913:882],
//   context[916:914], generation[924:917], EOR[925],
//   boundary merge[926], protocol error[927], terminal[928].
using CarryWorkToken = ap_uint<929>;

// Result packet layout:
//   items[511:0], mask[519:512], row[551:520], context[554:552],
//   generation[562:555], EOR[563], boundary merge[564],
//   protocol error[565], zero carry suppressed[566], carry flush[567],
//   terminal[568].
using CarryResultToken = ap_uint<569>;

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

static ap_uint<4> popcount8(ap_uint<8> value) {
#pragma HLS INLINE
	const ap_uint<3> low = value[0] + value[1] + value[2] + value[3];
	const ap_uint<3> high = value[4] + value[5] + value[6] + value[7];
	return low + high;
}

static MergeKeyedCarry8InputToken input_terminal() {
#pragma HLS INLINE
	MergeKeyedCarry8InputToken token = 0;
	token[576] = 1;
	return token;
}

static CarryWorkToken work_terminal() {
#pragma HLS INLINE
	CarryWorkToken token = 0;
	token[928] = 1;
	return token;
}

static CarryResultToken result_terminal() {
#pragma HLS INLINE
	CarryResultToken token = 0;
	token[568] = 1;
	return token;
}

}  // namespace

void merge_keyed_carry8_read(
		tapa::mmap<const ap_uint<512> > data_in,
		tapa::mmap<const ap_uint<64> > meta_in,
		id_t packet_count,
		tapa::ostream<MergeKeyedCarry8InputToken>& output) {
	for (id_t packet = 0; packet < packet_count; ++packet) {
#pragma HLS PIPELINE II=1
		MergeKeyedCarry8InputToken token = 0;
		token.range(511, 0) = data_in[packet];
		token.range(575, 512) = meta_in[packet];
		output.write(token);
	}
	output.write(input_terminal());
}

// One carry state belongs to each allocator context.  Consecutive updates of
// the same key rotate over eight independent phase registers.  A phase is not
// revisited for eight accepted packets, longer than the five-cycle FP32 add,
// so the input loop can remain II=1 without an unsafe scalar feedback path.
void merge_keyed_carry8_state(
		tapa::istream<MergeKeyedCarry8InputToken>& input,
		tapa::ostream<CarryWorkToken>& output) {
	bool active[8] = {false, false, false, false, false, false, false, false};
	id_t active_row[8];
	ap_uint<8> active_generation[8];
	ap_uint<32> active_col[8];
	ap_uint<32> base_value[8];
	ap_uint<3> next_phase[8];
	float phase_sum[8][8];
	float phase_snapshot[8][8];
	ap_uint<8> phase_valid[8];
#pragma HLS ARRAY_PARTITION variable=active complete
#pragma HLS ARRAY_PARTITION variable=active_row complete
#pragma HLS ARRAY_PARTITION variable=active_generation complete
#pragma HLS ARRAY_PARTITION variable=active_col complete
#pragma HLS ARRAY_PARTITION variable=base_value complete
#pragma HLS ARRAY_PARTITION variable=next_phase complete
// Eight physical phase banks; context is the small address within each bank.
// The snapshot mirror prevents an update read and a flush read from competing
// for the same 1R1W memory port.  Both copies are written together.
#pragma HLS ARRAY_PARTITION variable=phase_sum complete dim=2
#pragma HLS ARRAY_PARTITION variable=phase_snapshot complete dim=2
#pragma HLS ARRAY_PARTITION variable=phase_valid complete
#pragma HLS DEPENDENCE variable=phase_sum inter false
#pragma HLS DEPENDENCE variable=phase_snapshot inter false
	for (int context = 0; context < 8; ++context) {
#pragma HLS UNROLL
		active_row[context] = 0;
		active_generation[context] = 0;
		active_col[context] = 0;
		base_value[context] = 0;
		next_phase[context] = 0;
		phase_valid[context] = 0;
		for (int phase = 0; phase < 8; ++phase) {
#pragma HLS UNROLL
			phase_sum[context][phase] = 0.0f;
			phase_snapshot[context][phase] = 0.0f;
		}
	}

	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1048576 avg=4096
		MergeKeyedCarry8InputToken source;
		if (!input.try_read(source)) continue;
		if (source[576]) {
			output.write(work_terminal());
			done = true;
			continue;
		}

		const ap_uint<64> meta = source.range(575, 512);
		const id_t row = meta.range(31, 0);
		const ap_uint<8> source_mask = meta.range(39, 32);
		const ap_uint<3> context = meta.range(42, 40);
		const ap_uint<8> generation = meta.range(50, 43);
		const bool eor = meta[51];
		ap_uint<64> compact[8];
		ap_uint<4> group_count = 0;
#pragma HLS ARRAY_PARTITION variable=compact complete
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			compact[lane] = 0;
		}
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			if (source_mask[lane]) {
				compact[group_count] = source.range(lane * 64 + 63, lane * 64);
				++group_count;
			}
		}

		bool protocol_error = eor && group_count != 0;
		for (int lane = 1; lane < 8; ++lane) {
#pragma HLS UNROLL
			if (lane < group_count
					&& compact[lane].range(31, 0)
						<= compact[lane - 1].range(31, 0)) {
				protocol_error = true;
			}
		}
		const bool identity_matches = active[context]
			&& active_row[context] == row
			&& active_generation[context] == generation;
		if (active[context] && !identity_matches) protocol_error = true;

		CarryWorkToken work = 0;
		work.range(913, 882) = row;
		work.range(916, 914) = context;
		work.range(924, 917) = generation;
		work[925] = eor;
		work[927] = protocol_error;
		bool flush = false;
		bool boundary_merge = false;
		bool start_new = false;
		ap_uint<64> new_carry = 0;
		ap_uint<4> direct_begin = 0;
		ap_uint<4> direct_end = 0;

		if (eor) {
			if (identity_matches) flush = true;
			else if (active[context] || group_count != 0) work[927] = 1;
		} else if (group_count != 0) {
			const ap_uint<32> first_col = compact[0].range(31, 0);
			const bool same_key = identity_matches
				&& active_col[context] == first_col;
			if (identity_matches && active_col[context] > first_col) {
				work[927] = 1;
			}
			if (same_key && group_count == 1) {
				// The only group remains open.  Rotate the recurrence over eight
				// phase registers instead of feeding one FP adder every cycle.
				const ap_uint<3> phase = next_phase[context];
				const float contribution = from_raw(compact[0].range(63, 32));
				const float updated = phase_valid[context][phase]
					? phase_sum[context][phase] + contribution : contribution;
#pragma HLS BIND_OP variable=updated op=fadd impl=fulldsp latency=5
				phase_sum[context][phase] = updated;
				phase_snapshot[context][phase] = updated;
				phase_valid[context][phase] = 1;
				next_phase[context] = phase + 1;
				boundary_merge = true;
			} else {
				flush = identity_matches;
				boundary_merge = same_key;
				if (same_key) {
					// The current first group closes the previous key.  It is an
					// independent ninth input to the flush tree, not an FP feedback.
					work.range(847, 816) = compact[0].range(63, 32);
					work[848] = 1;
					direct_begin = 1;
				} else {
					direct_begin = 0;
				}
				direct_end = group_count - 1;
				start_new = true;
				new_carry = compact[group_count - 1];
			}
		}

		if (flush) {
			work[881] = 1;
			work.range(880, 849) = active_col[context];
			work.range(815, 784) = base_value[context];
			work.range(783, 776) = phase_valid[context];
			for (int phase = 0; phase < 8; ++phase) {
#pragma HLS UNROLL
				work.range(520 + phase * 32 + 31, 520 + phase * 32)
					= raw_float(phase_snapshot[context][phase]);
			}
		}

		ap_uint<64> direct_item[8];
		ap_uint<8> direct_mask = 0;
#pragma HLS ARRAY_PARTITION variable=direct_item complete
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			direct_item[lane] = 0;
		}
		ap_uint<4> direct_slot = flush ? (ap_uint<4>)1 : (ap_uint<4>)0;
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			if (lane >= direct_begin && lane < direct_end) {
				direct_item[direct_slot] = compact[lane];
				direct_mask[direct_slot] = 1;
				++direct_slot;
			}
		}
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			work.range(lane * 64 + 63, lane * 64) = direct_item[lane];
		}
		work.range(519, 512) = direct_mask;
		work[926] = boundary_merge;

		if (flush || start_new || eor) {
			phase_valid[context] = 0;
			next_phase[context] = 0;
		}
		if (start_new) {
			active[context] = true;
			active_row[context] = row;
			active_generation[context] = generation;
			active_col[context] = new_carry.range(31, 0);
			base_value[context] = new_carry.range(63, 32);
		} else if (eor) {
			active[context] = false;
		}
		output.write(work);
	}
}

// A completed key is reduced outside the state loop.  There is no recurrence
// here, so the fully pipelined FP tree can accept one carry snapshot per cycle.
void merge_keyed_carry8_flush_reduce(
		tapa::istream<CarryWorkToken>& input,
		tapa::ostream<CarryResultToken>& output) {
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1048576 avg=4096
		CarryWorkToken work;
		if (!input.try_read(work)) continue;
		if (work[928]) {
			output.write(result_terminal());
			done = true;
			continue;
		}
		float value[16];
		float level1[8];
		float level2[4];
		float level3[2];
#pragma HLS ARRAY_PARTITION variable=value complete
#pragma HLS ARRAY_PARTITION variable=level1 complete
#pragma HLS ARRAY_PARTITION variable=level2 complete
#pragma HLS ARRAY_PARTITION variable=level3 complete
		const ap_uint<8> phase_mask = work.range(783, 776);
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			value[lane] = phase_mask[lane]
				? from_raw(work.range(520 + lane * 32 + 31, 520 + lane * 32))
				: 0.0f;
		}
		value[8] = work[881] ? from_raw(work.range(815, 784)) : 0.0f;
		value[9] = work[848] ? from_raw(work.range(847, 816)) : 0.0f;
		for (int lane = 10; lane < 16; ++lane) {
#pragma HLS UNROLL
			value[lane] = 0.0f;
		}
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			level1[lane] = value[2 * lane] + value[2 * lane + 1];
		}
		for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
			level2[lane] = level1[2 * lane] + level1[2 * lane + 1];
		}
		for (int lane = 0; lane < 2; ++lane) {
#pragma HLS UNROLL
			level3[lane] = level2[2 * lane] + level2[2 * lane + 1];
		}
		const float sum = level3[0] + level3[1];
#pragma HLS BIND_OP variable=level1 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=level2 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=level3 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum op=fadd impl=fulldsp latency=5

		CarryResultToken result = 0;
		result.range(511, 0) = work.range(511, 0);
		result.range(519, 512) = work.range(519, 512);
		result.range(551, 520) = work.range(913, 882);
		result.range(554, 552) = work.range(916, 914);
		result.range(562, 555) = work.range(924, 917);
		result[563] = work[925];
		result[564] = work[926];
		result[565] = work[927];
		result[567] = work[881];
		if (work[881]) {
			if (nonzero(sum)) {
				ap_uint<64> item = 0;
				item.range(31, 0) = work.range(880, 849);
				item.range(63, 32) = raw_float(sum);
				result.range(63, 0) = item;
				result[512] = 1;
			} else {
				result[566] = 1;
			}
		}
		output.write(result);
	}
}

void merge_keyed_carry8_write(
		tapa::istream<CarryResultToken>& input,
		tapa::mmap<ap_uint<512> > data_out,
		tapa::mmap<ap_uint<64> > meta_out,
		tapa::mmap<id_t> statistics_out) {
	id_t work_packets = 0;
	id_t output_packets = 0;
	id_t output_items = 0;
	id_t eors = 0;
	id_t boundary_merges = 0;
	id_t carry_flushes = 0;
	id_t zero_carries = 0;
	id_t protocol_errors = 0;
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		const CarryResultToken result = input.read();
		if (result[568]) {
			done = true;
			continue;
		}
		++work_packets;
		const ap_uint<8> mask = result.range(519, 512);
		const bool eor = result[563];
		output_items += popcount8(mask);
		eors += eor;
		boundary_merges += result[564];
		protocol_errors += result[565];
		zero_carries += result[566];
		carry_flushes += result[567];
		if (mask != 0 || eor) {
			data_out[output_packets] = result.range(511, 0);
			ap_uint<64> meta = 0;
			meta.range(31, 0) = result.range(551, 520);
			meta.range(39, 32) = mask;
			meta.range(42, 40) = result.range(554, 552);
			meta.range(50, 43) = result.range(562, 555);
			meta[51] = eor;
			meta_out[output_packets] = meta;
			++output_packets;
		}
	}
	statistics_out[0] = work_packets;
	statistics_out[1] = output_packets;
	statistics_out[2] = output_items;
	statistics_out[3] = eors;
	statistics_out[4] = boundary_merges;
	statistics_out[5] = carry_flushes;
	statistics_out[6] = zero_carries;
	statistics_out[7] = protocol_errors;
}

void adaptive_hbm_merge_keyed_carry8_tapa(
		tapa::mmap<const ap_uint<512> > data_in,
		tapa::mmap<const ap_uint<64> > meta_in,
		id_t packet_count,
		tapa::mmap<ap_uint<512> > data_out,
		tapa::mmap<ap_uint<64> > meta_out,
		tapa::mmap<id_t> statistics_out) {
	tapa::stream<MergeKeyedCarry8InputToken, 8> source("source");
	tapa::stream<CarryWorkToken, 8> work("work");
	tapa::stream<CarryResultToken, 8> result("result");
	tapa::task()
		.invoke(merge_keyed_carry8_read, data_in, meta_in, packet_count, source)
		.invoke(merge_keyed_carry8_state, source, work)
		.invoke(merge_keyed_carry8_flush_reduce, work, result)
		.invoke(merge_keyed_carry8_write, result, data_out, meta_out,
			statistics_out);
}
