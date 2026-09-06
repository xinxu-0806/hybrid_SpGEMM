#include "adaptive_hbm_unified_adaptive_datapath_tapa.h"

// Compose the already accepted numerical blocks without modifying their
// source checkpoints.  File-local helper prefixes avoid collisions when the
// implementations share this translation unit.
#include "../segmented8_shared/adaptive_hbm_segmented8_shared_tapa.h"
#include "../merge_keyed_carry8/adaptive_hbm_merge_keyed_carry8_tapa.h"
#include "../unified_dense_dual_context/adaptive_hbm_unified_dense_dual_context_tapa.h"

#define raw_float joint_segmented_raw_float
#define from_raw joint_segmented_from_raw
#define nonzero joint_segmented_nonzero
#define popcount8 joint_segmented_popcount8
#define end_token joint_segmented_end_token
#include "../segmented8_shared/adaptive_hbm_segmented8_shared_tapa.cpp"
#undef end_token
#undef popcount8
#undef nonzero
#undef from_raw
#undef raw_float

#define raw_float joint_carry_raw_float
#define from_raw joint_carry_from_raw
#define nonzero joint_carry_nonzero
#define popcount8 joint_carry_popcount8
#include "../merge_keyed_carry8/adaptive_hbm_merge_keyed_carry8_tapa.cpp"
#undef popcount8
#undef nonzero
#undef from_raw
#undef raw_float

#define raw_float joint_dense_raw_float
#define from_raw joint_dense_from_raw
#define nonzero joint_dense_nonzero
#define popcount8 joint_dense_popcount8
#define end_token joint_dense_end_token
#define make_completion joint_dense_make_completion
#include "../unified_dense_dual_context/adaptive_hbm_unified_dense_dual_context_tapa.cpp"
#undef make_completion
#undef end_token
#undef popcount8
#undef nonzero
#undef from_raw
#undef raw_float

namespace {

// row[31:0], context[34:32], generation[42:35], route[44:43], EOR[45],
// dense slot[46], terminal[47].
using JointAdaptiveTag = ap_uint<48>;

// One item per original packet lane.  Eight independent lane FIFOs decouple
// packet ingestion from the eight DENSE banks, so distinct columns hashing to
// the same bank are replayed locally without blocking unrelated banks.
// item[63:0], row[95:64], context[98:96], generation[106:99], error[107],
// EOR[108], terminal[109], logical 64K row uses both slots[110].
using JointDenseLaneToken = ap_uint<111>;

static JointAdaptiveTag joint_end_tag() {
#pragma HLS INLINE
	JointAdaptiveTag tag = 0;
	tag[47] = 1;
	return tag;
}

static JointDenseLaneToken joint_dense_lane_terminal() {
#pragma HLS INLINE
	JointDenseLaneToken token = 0;
	token[109] = 1;
	return token;
}

static ap_uint<4> joint_popcount8(ap_uint<8> value) {
#pragma HLS INLINE
	const ap_uint<3> low = value[0] + value[1] + value[2] + value[3];
	const ap_uint<3> high = value[4] + value[5] + value[6] + value[7];
	return low + high;
}

static ap_uint<64> joint_make_completion(id_t row, ap_uint<3> context,
		ap_uint<8> generation) {
#pragma HLS INLINE
	ap_uint<64> event = 0;
	event.range(2, 0) = context;
	event.range(10, 3) = generation;
	event.range(42, 11) = row;
	return event;
}

}  // namespace

void unified_adaptive_read(
		tapa::mmap<const ap_uint<512> > data_in,
		tapa::mmap<const ap_uint<64> > meta_in,
		id_t packet_count,
		tapa::ostream<Segmented8SharedToken>& numerical,
		tapa::ostream<JointAdaptiveTag>& tags) {
	for (id_t packet = 0; packet < packet_count; ++packet) {
#pragma HLS PIPELINE II=1
		const ap_uint<64> meta = meta_in[packet];
		Segmented8SharedToken token = 0;
		token.range(511, 0) = data_in[packet];
		token.range(543, 512) = meta.range(31, 0);
		token.range(551, 544) = meta.range(39, 32);
		token.range(553, 552) = meta.range(41, 40);
		token[554] = meta[42];
		numerical.write(token);

		JointAdaptiveTag tag = 0;
		tag.range(31, 0) = meta.range(31, 0);
		tag.range(34, 32) = meta.range(45, 43);
		tag.range(42, 35) = meta.range(53, 46);
		tag.range(44, 43) = meta.range(41, 40);
		tag[45] = meta[42];
		tag[46] = meta[54];
		tags.write(tag);
	}
	Segmented8SharedToken terminal = 0;
	terminal[556] = 1;
	numerical.write(terminal);
	tags.write(joint_end_tag());
}

// This is the only algorithmic fork.  It restores allocator identity after
// packet-local reduction and sends DENSE rows straight into their physical
// accumulator slot; no intermediate Host round-trip or third algorithm mode.
void unified_adaptive_route(
		tapa::istream<Segmented8SharedToken>& numerical,
		tapa::istream<JointAdaptiveTag>& tags,
		tapa::ostream<MergeKeyedCarry8InputToken>& merge_output,
		tapa::ostream<DenseDualContextToken>& dense_output0,
		tapa::ostream<DenseDualContextToken>& dense_output1) {
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		const Segmented8SharedToken reduced = numerical.read();
		const JointAdaptiveTag tag = tags.read();
		if (reduced[556] || tag[47]) {
			MergeKeyedCarry8InputToken merge_end = 0;
			merge_end[576] = 1;
			merge_output.write(merge_end);
			DenseDualContextToken dense_end = 0;
			dense_end[566] = 1;
			dense_output0.write(dense_end);
			dense_output1.write(dense_end);
			done = true;
			continue;
		}
		const bool identity_error
			= reduced.range(543, 512) != tag.range(31, 0)
			|| reduced.range(553, 552) != tag.range(44, 43)
			|| reduced[554] != tag[45];
		if (tag.range(44, 43) == UNIFIED_ADAPTIVE_ROUTE_DENSE) {
			DenseDualContextToken token = 0;
			token.range(511, 0) = reduced.range(511, 0);
			token.range(543, 512) = tag.range(31, 0);
			token.range(551, 544) = reduced.range(551, 544);
			token[552] = tag[46];
			token[553] = tag[45];
			token.range(556, 554) = tag.range(34, 32);
			token.range(564, 557) = tag.range(42, 35);
			token[565] = identity_error;
			if (tag[46]) dense_output1.write(token);
			else dense_output0.write(token);
		} else {
			MergeKeyedCarry8InputToken token = 0;
			token.range(511, 0) = reduced.range(511, 0);
			ap_uint<64> meta = 0;
			meta.range(31, 0) = tag.range(31, 0);
			meta.range(39, 32) = reduced.range(551, 544);
			meta.range(42, 40) = tag.range(34, 32);
			meta.range(50, 43) = tag.range(42, 35);
			meta[51] = tag[45];
			meta[52] = identity_error;
			token.range(575, 512) = meta;
			merge_output.write(token);
		}
	}
}

void unified_dense_lane_demux(
		tapa::istream<DenseDualContextToken>& input,
		tapa::ostream<JointDenseLaneToken>& lane0,
		tapa::ostream<JointDenseLaneToken>& lane1,
		tapa::ostream<JointDenseLaneToken>& lane2,
		tapa::ostream<JointDenseLaneToken>& lane3,
		tapa::ostream<JointDenseLaneToken>& lane4,
		tapa::ostream<JointDenseLaneToken>& lane5,
		tapa::ostream<JointDenseLaneToken>& lane6,
		tapa::ostream<JointDenseLaneToken>& lane7) {
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		DenseDualContextToken input_token;
		// Preserve pipeline draining across the idle interval between rows.
		if (!input.try_read(input_token)) continue;
		if (input_token[566]) {
			const JointDenseLaneToken terminal = joint_dense_lane_terminal();
			lane0.write(terminal); lane1.write(terminal);
			lane2.write(terminal); lane3.write(terminal);
			lane4.write(terminal); lane5.write(terminal);
			lane6.write(terminal); lane7.write(terminal);
			done = true;
			continue;
		}
		JointDenseLaneToken common = 0;
		common.range(95, 64) = input_token.range(543, 512);
		common.range(98, 96) = input_token.range(556, 554);
		common.range(106, 99) = input_token.range(564, 557);
		common[107] = input_token[565];
		common[108] = input_token[553];
		common[110] = input_token[567];
		if (input_token[553]) {
			lane0.write(common); lane1.write(common);
			lane2.write(common); lane3.write(common);
			lane4.write(common); lane5.write(common);
			lane6.write(common); lane7.write(common);
			continue;
		}
		const ap_uint<8> mask = input_token.range(551, 544);
#define JOINT_DENSE_LANE_WRITE(L, STREAM) \
		if (mask[L]) { \
			JointDenseLaneToken token = common; \
			token.range(63, 0) = input_token.range(L * 64 + 63, L * 64); \
			STREAM.write(token); \
		}
		JOINT_DENSE_LANE_WRITE(0, lane0)
		JOINT_DENSE_LANE_WRITE(1, lane1)
		JOINT_DENSE_LANE_WRITE(2, lane2)
		JOINT_DENSE_LANE_WRITE(3, lane3)
		JOINT_DENSE_LANE_WRITE(4, lane4)
		JOINT_DENSE_LANE_WRITE(5, lane5)
		JOINT_DENSE_LANE_WRITE(6, lane6)
		JOINT_DENSE_LANE_WRITE(7, lane7)
#undef JOINT_DENSE_LANE_WRITE
	}
}

template <int SLOT>
static void unified_dense_bank_arbiter_impl(
		tapa::istream<JointDenseLaneToken>& lane0,
		tapa::istream<JointDenseLaneToken>& lane1,
		tapa::istream<JointDenseLaneToken>& lane2,
		tapa::istream<JointDenseLaneToken>& lane3,
		tapa::istream<JointDenseLaneToken>& lane4,
		tapa::istream<JointDenseLaneToken>& lane5,
		tapa::istream<JointDenseLaneToken>& lane6,
		tapa::istream<JointDenseLaneToken>& lane7,
		tapa::ostream<DenseDualContextToken>& output) {
#pragma HLS INLINE off
	JointDenseLaneToken head[8];
	bool valid[8];
#pragma HLS ARRAY_PARTITION variable=head complete
#pragma HLS ARRAY_PARTITION variable=valid complete
	for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
		head[lane] = 0;
		valid[lane] = false;
	}
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1048576 avg=4096
#define JOINT_DENSE_REFILL(L, STREAM) \
		if (!valid[L]) { \
			JointDenseLaneToken token; \
			if (STREAM.try_read(token)) { head[L] = token; valid[L] = true; } \
		}
		JOINT_DENSE_REFILL(0, lane0)
		JOINT_DENSE_REFILL(1, lane1)
		JOINT_DENSE_REFILL(2, lane2)
		JOINT_DENSE_REFILL(3, lane3)
		JOINT_DENSE_REFILL(4, lane4)
		JOINT_DENSE_REFILL(5, lane5)
		JOINT_DENSE_REFILL(6, lane6)
		JOINT_DENSE_REFILL(7, lane7)
#undef JOINT_DENSE_REFILL

		bool all_valid = true;
		bool all_terminal = true;
		bool all_eor = true;
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			all_valid &= valid[lane];
			all_terminal &= valid[lane] && head[lane][109];
			all_eor &= valid[lane] && head[lane][108];
		}
		if (all_valid && all_terminal) {
			DenseDualContextToken terminal = 0;
			terminal[566] = 1;
			output.write(terminal);
			done = true;
			continue;
		}
		if (all_valid && all_eor) {
			DenseDualContextToken boundary = 0;
			boundary.range(543, 512) = head[0].range(95, 64);
			boundary[552] = SLOT;
			boundary[553] = 1;
			boundary.range(556, 554) = head[0].range(98, 96);
			boundary.range(564, 557) = head[0].range(106, 99);
			boundary[567] = head[0][110];
			bool error = false;
			for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
				error |= head[lane][107]
					|| head[lane].range(95, 64) != head[0].range(95, 64)
					|| head[lane].range(98, 96) != head[0].range(98, 96)
					|| head[lane].range(106, 99) != head[0].range(106, 99);
				error |= head[lane][110] != head[0][110];
				valid[lane] = false;
			}
			boundary[565] = error;
			output.write(boundary);
			continue;
		}

		ap_uint<8> chosen_lane[8];
#pragma HLS ARRAY_PARTITION variable=chosen_lane complete
		ap_uint<8> consumed = 0;
		ap_uint<8> output_mask = 0;
		for (int bank = 0; bank < 8; ++bank) {
#pragma HLS UNROLL
			ap_uint<8> request = 0;
			for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
				request[lane] = valid[lane] && !head[lane][108]
					&& !head[lane][109]
					&& head[lane].range(2, 0) == (ap_uint<3>)bank;
			}
			chosen_lane[bank] = request & (ap_uint<8>)(~request + 1);
			output_mask[bank] = request != 0;
			consumed |= chosen_lane[bank];
		}
		if (output_mask == 0) continue;
		unsigned first_lane = 0;
		for (int lane = 7; lane >= 0; --lane) {
#pragma HLS UNROLL
			if (consumed[lane]) first_lane = lane;
		}
		DenseDualContextToken result = 0;
		result.range(543, 512) = head[first_lane].range(95, 64);
		result.range(551, 544) = output_mask;
		result[552] = SLOT;
		result.range(556, 554) = head[first_lane].range(98, 96);
		result.range(564, 557) = head[first_lane].range(106, 99);
		result[567] = head[first_lane][110];
		bool error = false;
		for (int bank = 0; bank < 8; ++bank) {
#pragma HLS UNROLL
			ap_uint<64> item = 0;
			for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
				if (chosen_lane[bank][lane]) item = head[lane].range(63, 0);
			}
			result.range(bank * 64 + 63, bank * 64) = item;
		}
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			if (consumed[lane]) {
				error |= head[lane][107]
					|| head[lane].range(95, 64)
						!= head[first_lane].range(95, 64)
					|| head[lane].range(98, 96)
						!= head[first_lane].range(98, 96)
					|| head[lane].range(106, 99)
						!= head[first_lane].range(106, 99)
					|| head[lane][110] != head[first_lane][110];
				valid[lane] = false;
			}
		}
		result[565] = error;
		output.write(result);
	}
}

void unified_dense_bank_arbiter0(
		tapa::istream<JointDenseLaneToken>& lane0,
		tapa::istream<JointDenseLaneToken>& lane1,
		tapa::istream<JointDenseLaneToken>& lane2,
		tapa::istream<JointDenseLaneToken>& lane3,
		tapa::istream<JointDenseLaneToken>& lane4,
		tapa::istream<JointDenseLaneToken>& lane5,
		tapa::istream<JointDenseLaneToken>& lane6,
		tapa::istream<JointDenseLaneToken>& lane7,
		tapa::ostream<DenseDualContextToken>& output) {
	unified_dense_bank_arbiter_impl<0>(lane0, lane1, lane2, lane3,
		lane4, lane5, lane6, lane7, output);
}

void unified_dense_bank_arbiter1(
		tapa::istream<JointDenseLaneToken>& lane0,
		tapa::istream<JointDenseLaneToken>& lane1,
		tapa::istream<JointDenseLaneToken>& lane2,
		tapa::istream<JointDenseLaneToken>& lane3,
		tapa::istream<JointDenseLaneToken>& lane4,
		tapa::istream<JointDenseLaneToken>& lane5,
		tapa::istream<JointDenseLaneToken>& lane6,
		tapa::istream<JointDenseLaneToken>& lane7,
		tapa::ostream<DenseDualContextToken>& output) {
	unified_dense_bank_arbiter_impl<1>(lane0, lane1, lane2, lane3,
		lane4, lane5, lane6, lane7, output);
}

void unified_adaptive_merge_write(
		tapa::istream<CarryResultToken>& input,
		tapa::mmap<ap_uint<512> > data_out,
		tapa::mmap<ap_uint<64> > meta_out,
		tapa::mmap<ap_uint<64> > completion_out,
		tapa::mmap<id_t> stats_out) {
	id_t packets = 0;
	id_t items = 0;
	id_t eors = 0;
	id_t completions = 0;
	id_t boundaries = 0;
	id_t flushes = 0;
	id_t zero_carries = 0;
	id_t errors = 0;
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		const CarryResultToken result = input.read();
		if (result[568]) { done = true; continue; }
		const ap_uint<8> mask = result.range(519, 512);
		const bool eor = result[563];
		const id_t row = result.range(551, 520);
		const ap_uint<3> context = result.range(554, 552);
		const ap_uint<8> generation = result.range(562, 555);
		items += joint_popcount8(mask);
		boundaries += result[564];
		errors += result[565];
		zero_carries += result[566];
		flushes += result[567];
		if (mask != 0 || eor) {
			data_out[packets] = result.range(511, 0);
			ap_uint<64> meta = 0;
			meta.range(31, 0) = row;
			meta.range(39, 32) = mask;
			meta.range(42, 40) = context;
			meta.range(50, 43) = generation;
			meta[51] = eor;
			meta[52] = result[565];
			meta_out[packets++] = meta;
		}
		if (eor) {
			completion_out[completions++] = joint_make_completion(
				row, context, generation);
			++eors;
		}
	}
	stats_out[0] = packets;
	stats_out[1] = items;
	stats_out[2] = eors;
	stats_out[3] = completions;
	stats_out[4] = boundaries;
	stats_out[5] = flushes;
	stats_out[6] = zero_carries;
	stats_out[7] = errors;
}

void adaptive_hbm_unified_adaptive_datapath_tapa(
		tapa::mmap<const ap_uint<512> > data_in,
		tapa::mmap<const ap_uint<64> > meta_in,
		id_t packet_count,
		id_t column_count,
		tapa::mmap<ap_uint<512> > merge_data_out,
		tapa::mmap<ap_uint<64> > merge_meta_out,
		tapa::mmap<ap_uint<64> > merge_completion_out,
		tapa::mmap<id_t> merge_stats_out,
		tapa::mmap<ap_uint<512> > dense_data_out0,
		tapa::mmap<ap_uint<64> > dense_meta_out0,
		tapa::mmap<ap_uint<64> > dense_completion_out0,
		tapa::mmap<id_t> dense_stats_out0,
		tapa::mmap<ap_uint<512> > dense_data_out1,
		tapa::mmap<ap_uint<64> > dense_meta_out1,
		tapa::mmap<ap_uint<64> > dense_completion_out1,
		tapa::mmap<id_t> dense_stats_out1) {
	tapa::stream<Segmented8SharedToken, 8> raw("raw");
	tapa::stream<JointAdaptiveTag, 8> tags("tags");
	tapa::stream<Segmented8SharedToken, 8> reduced("reduced");
	tapa::stream<MergeKeyedCarry8InputToken, 8> merge("merge");
	tapa::streams<DenseDualContextToken, 2, 16> dense_prebank("dense_prebank");
	tapa::streams<JointDenseLaneToken, 8, 16> dense_lane0("dense_lane0");
	tapa::streams<JointDenseLaneToken, 8, 16> dense_lane1("dense_lane1");
	tapa::streams<DenseDualContextToken, 2, 16> dense("dense");
	tapa::stream<CarryWorkToken, 8> carry_work("carry_work");
	tapa::stream<CarryResultToken, 8> carry_result("carry_result");
	tapa::streams<DenseDualContextToken, 2, 16> dense_result("dense_result");
	tapa::streams<ap_uint<192>, 2, 2> dense_statistics("dense_statistics");
	tapa::task()
		.invoke(unified_adaptive_read, data_in, meta_in, packet_count, raw, tags)
		.invoke(segmented8_shared_reduce, raw, reduced)
		.invoke(unified_adaptive_route, reduced, tags, merge,
			dense_prebank[0], dense_prebank[1])
		.invoke(unified_dense_lane_demux, dense_prebank[0],
			dense_lane0[0], dense_lane0[1], dense_lane0[2], dense_lane0[3],
			dense_lane0[4], dense_lane0[5], dense_lane0[6], dense_lane0[7])
		.invoke(unified_dense_lane_demux, dense_prebank[1],
			dense_lane1[0], dense_lane1[1], dense_lane1[2], dense_lane1[3],
			dense_lane1[4], dense_lane1[5], dense_lane1[6], dense_lane1[7])
		.invoke(unified_dense_bank_arbiter0,
			dense_lane0[0], dense_lane0[1], dense_lane0[2], dense_lane0[3],
			dense_lane0[4], dense_lane0[5], dense_lane0[6], dense_lane0[7],
			dense[0])
		.invoke(unified_dense_bank_arbiter1,
			dense_lane1[0], dense_lane1[1], dense_lane1[2], dense_lane1[3],
			dense_lane1[4], dense_lane1[5], dense_lane1[6], dense_lane1[7],
			dense[1])
		.invoke(merge_keyed_carry8_state, merge, carry_work)
		.invoke(merge_keyed_carry8_flush_reduce, carry_work, carry_result)
		.invoke(unified_adaptive_merge_write, carry_result, merge_data_out,
			merge_meta_out, merge_completion_out, merge_stats_out)
		.invoke(dense_dual_context_engine0, dense[0], column_count,
			dense_result[0], dense_statistics[0])
		.invoke(dense_dual_context_engine1, dense[1], column_count,
			dense_result[1], dense_statistics[1])
		.invoke(dense_dual_context_write, dense_result[0], dense_statistics[0],
			dense_data_out0, dense_meta_out0, dense_completion_out0,
			dense_stats_out0)
		.invoke(dense_dual_context_write, dense_result[1], dense_statistics[1],
			dense_data_out1, dense_meta_out1, dense_completion_out1,
			dense_stats_out1);
}
