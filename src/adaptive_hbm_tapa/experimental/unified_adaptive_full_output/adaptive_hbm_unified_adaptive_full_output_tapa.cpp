#include "adaptive_hbm_unified_adaptive_full_output_tapa.h"

// Reuse the accepted task bodies.  The two included files are composition
// checkpoints, not separate kernels in this graph; only tasks explicitly
// invoked by the top below are synthesized.
#include "../unified_allocator_merge_output/adaptive_hbm_unified_allocator_merge_output_tapa.cpp"
#include "../unified_adaptive_datapath/adaptive_hbm_unified_adaptive_datapath_tapa.cpp"

namespace {

// numerical[63:0], row[95:64], context[98:96], generation[106:99],
// physical DENSE slot[107], EOR[108], terminal[109], sequence[125:110],
// protocol error[126], internal heavy-MERGE microengine[127],
// logical 64K DENSE row uses both physical 32K slots[128].
using UnifiedFullDenseSourceToken = ap_uint<129>;

// row[31:0], context[34:32], generation[42:35], physical slot[43],
// EOR[44], protocol error[45], terminal[46], internal heavy MERGE[47],
// logical 64K DENSE row[48].
using UnifiedFullDenseTag = ap_uint<49>;

static UnifiedFullDenseSourceToken unified_full_dense_source_end() {
#pragma HLS INLINE
	UnifiedFullDenseSourceToken token = 0;
	token[109] = 1;
	return token;
}

static UnifiedFullDenseTag unified_full_dense_tag_end() {
#pragma HLS INLINE
	UnifiedFullDenseTag tag = 0;
	tag[46] = 1;
	return tag;
}

static ap_uint<128> unified_full_make_record(id_t row, id_t row_nnz,
		ap_uint<4> valid, bool eor, ap_uint<3> context,
		ap_uint<2> route, ap_uint<4> logical_unit, ap_uint<2> error,
		ap_uint<2> port, ap_uint<8> generation) {
#pragma HLS INLINE
	ap_uint<128> record = 0;
	record.range(31, 0) = row;
	record.range(63, 32) = row_nnz;
	record.range(67, 64) = valid;
	record[68] = eor;
	record.range(71, 69) = context;
	record.range(73, 72) = route;
	record.range(77, 74) = logical_unit;
	record.range(79, 78) = error;
	record.range(81, 80) = port;
	record.range(89, 82) = generation;
	record[127] = 1;
	return record;
}

static PacketBundle unified_full_make_bundle(ap_uint<512> word,
		ap_uint<128> record) {
#pragma HLS INLINE
	PacketBundle bundle = 0;
	bundle.range(511, 0) = word;
	bundle.range(639, 512) = record;
	return bundle;
}

static ap_uint<64> unified_full_make_completion(id_t row,
		ap_uint<3> context, ap_uint<8> generation, bool dense,
		ap_uint<4> logical_unit) {
#pragma HLS INLINE
	ap_uint<64> event = 0;
	event.range(2, 0) = context;
	event.range(10, 3) = generation;
	event.range(42, 11) = row;
	event.range(46, 43) = logical_unit;
	event[47] = dense;
	return event;
}

static ap_uint<2> unified_full_first_dense_slot(ap_uint<2> free_slots) {
#pragma HLS INLINE
	return free_slots[0] ? (ap_uint<2>)0 : (ap_uint<2>)1;
}

static ap_uint<4> unified_full_low_mask4(ap_uint<3> count) {
#pragma HLS INLINE
	switch ((unsigned)count) {
	case 0: return 0x0;
	case 1: return 0x1;
	case 2: return 0x3;
	case 3: return 0x7;
	default: return 0xf;
	}
}

static ap_uint<8> unified_full_low_mask8(ap_uint<4> count) {
#pragma HLS INLINE
	switch ((unsigned)count) {
	case 0: return 0x00;
	case 1: return 0x01;
	case 2: return 0x03;
	case 3: return 0x07;
	case 4: return 0x0f;
	case 5: return 0x1f;
	case 6: return 0x3f;
	case 7: return 0x7f;
	default: return 0xff;
	}
}

static ap_uint<4> unified_full_popcount8(ap_uint<8> value) {
#pragma HLS INLINE
	const ap_uint<2> pair0 = value[0] + value[1];
	const ap_uint<2> pair1 = value[2] + value[3];
	const ap_uint<2> pair2 = value[4] + value[5];
	const ap_uint<2> pair3 = value[6] + value[7];
	const ap_uint<3> half0 = pair0 + pair1;
	const ap_uint<3> half1 = pair2 + pair3;
	return half0 + half1;
}

}  // namespace

// One logical context pool is shared by both algorithms.  A MERGE row owns a
// buddy-aligned tree-leaf block; a DENSE row owns one of two physical
// accumulators.  Both resources are released only by a post-output completion.
void unified_full_scheduler(
		tapa::istream<MergeAllocator8CommandToken>& command_input,
		tapa::istream<MergeAllocator8EorToken>& eor_input,
		id_t heavy_merge_product_threshold,
		tapa::ostream<MergeAllocator8DispatchToken>& dispatch_output,
		tapa::ostream<MergeAllocator8AckToken>& ack_output,
		tapa::ostream<ap_uint<512> >& statistics_output) {
	ap_uint<8> busy = 0;
	ap_uint<8> free_leaf = 0xff;
	ap_uint<2> free_dense = 3;
	id_t active_row[8] = {};
	ap_uint<8> active_generation[8] = {};
	ap_uint<8> active_leaf_mask[8] = {};
	ap_uint<1> active_is_dense[8] = {};
	ap_uint<1> active_is_heavy[8] = {};
	ap_uint<1> active_is_wide[8] = {};
	ap_uint<1> active_dense_slot[8] = {};
#pragma HLS ARRAY_PARTITION variable=active_row complete
#pragma HLS ARRAY_PARTITION variable=active_generation complete
#pragma HLS ARRAY_PARTITION variable=active_leaf_mask complete
#pragma HLS ARRAY_PARTITION variable=active_is_dense complete
#pragma HLS ARRAY_PARTITION variable=active_is_heavy complete
#pragma HLS ARRAY_PARTITION variable=active_is_wide complete
#pragma HLS ARRAY_PARTITION variable=active_dense_slot complete
	ap_uint<8> generation_counter = 0;
	MergeAllocator8CommandToken held_command = 0;
	MergeAllocator8EorToken held_eor = 0;
	bool command_valid = false;
	bool eor_valid = false;
	bool command_done = false;
	bool eor_done = false;
	bool dispatch_done = false;
	id_t accepted_count = 0;
	id_t completed_count = 0;
	id_t merge_count = 0;
	id_t dense_count = 0;
	id_t peak_contexts = 0;
	id_t peak_leaves = 0;
	id_t peak_dense = 0;
	id_t level_count[4] = {};
#pragma HLS ARRAY_PARTITION variable=level_count complete
	id_t stale_count = 0;
	id_t blocked_cycles = 0;
	id_t route_errors = 0;

	while (!(dispatch_done && eor_done && !eor_valid && busy == 0)) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=1048576 avg=4096
		const ap_uint<8> allocation_busy = busy;
		const ap_uint<8> allocation_free_leaf = free_leaf;
		const ap_uint<2> allocation_free_dense = free_dense;

		if (eor_valid) {
			const ap_uint<3> context = held_eor.range(2, 0);
			const ap_uint<8> generation = held_eor.range(10, 3);
			const id_t row = held_eor.range(42, 11);
			const bool accepted = busy[context]
				&& active_generation[context] == generation
				&& active_row[context] == row;
			ap_uint<8> released = 0;
			ap_uint<3> status = 0;
			if (accepted) {
				if (active_is_dense[context]) {
					if (active_is_wide[context]) free_dense = 3;
					else free_dense[active_dense_slot[context]] = 1;
				} else if (!active_is_heavy[context]) {
					released = active_leaf_mask[context];
					free_leaf |= released;
				}
				busy[context] = 0;
				active_leaf_mask[context] = 0;
				active_is_wide[context] = 0;
				++completed_count;
			} else {
				status = 1;
				++stale_count;
			}
			MergeAllocator8AckToken ack = 0;
			ack.range(63, 0) = make_ack(held_eor.range(63, 0),
				accepted, status, released);
			ack_output.write(ack);
			eor_valid = false;
		}

		if (command_valid) {
			const ap_uint<128> command = held_command.range(127, 0);
			const ap_uint<8> source_mask = command.range(71, 64);
			const ap_uint<2> route = command.range(73, 72);
			const bool dense = route == UNIFIED_FULL_ROUTE_DENSE;
			const bool wide = dense && command[106];
			const bool heavy = !dense && heavy_merge_product_threshold != 0
				&& (id_t)command.range(105, 74)
					>= heavy_merge_product_threshold;
			const bool route_valid = dense
				|| route == UNIFIED_FULL_ROUTE_MERGE;
			const ap_uint<4> width = allocation_width(source_mask);
			ap_uint<3> context = 0;
			ap_uint<3> leaf_base = 0;
			ap_uint<8> leaf_mask = 0;
			const bool have_context = find_context(allocation_busy, context);
			const bool have_leaves = dense || heavy || find_leaf_block(
				allocation_free_leaf, width, leaf_base, leaf_mask);
			const bool have_dense = !dense || (wide
				? allocation_free_dense == 3 : allocation_free_dense != 0);
			if (source_mask != 0 && route_valid && have_context
					&& have_leaves && have_dense) {
				const ap_uint<8> generation = generation_counter + 1;
				generation_counter = generation;
				const ap_uint<1> dense_slot = wide ? (ap_uint<1>)0
					: (ap_uint<1>)unified_full_first_dense_slot(
						allocation_free_dense);
				busy[context] = 1;
				active_row[context] = command.range(31, 0);
				active_generation[context] = generation;
				active_is_dense[context] = dense;
				active_is_heavy[context] = heavy;
				active_is_wide[context] = wide;
				active_dense_slot[context] = dense_slot;
				active_leaf_mask[context]
					= (dense || heavy) ? (ap_uint<8>)0 : leaf_mask;
				if (dense) {
					if (wide) free_dense = 0;
					else free_dense[dense_slot] = 0;
				}
				else if (!heavy) free_leaf &= ~leaf_mask;
				MergeAllocator8DispatchToken dispatch = 0;
				ap_uint<128> record = make_dispatch(command,
					(dense || heavy) ? (ap_uint<8>)0 : leaf_mask, leaf_base,
					context, generation,
					(dense || heavy) ? (ap_uint<4>)1 : width);
				record[121] = dense;
				record[122] = dense_slot;
				record[123] = heavy;
				record[124] = wide;
				dispatch.range(127, 0) = record;
				dispatch_output.write(dispatch);
				++accepted_count;
				if (dense) ++dense_count;
				else if (!heavy) {
					++merge_count;
					++level_count[(unsigned)allocation_level(width)];
				} else ++merge_count;
				command_valid = false;
			} else if (!route_valid) {
				++route_errors;
				command_valid = false;
			} else {
				++blocked_cycles;
			}
		} else if (command_done && !dispatch_done) {
			MergeAllocator8DispatchToken terminal = 0;
			terminal[128] = 1;
			dispatch_output.write(terminal);
			dispatch_done = true;
		}

		const id_t live_contexts
			= allocator_merge_output_allocator_popcount8(busy);
		const id_t live_leaves = 8
			- allocator_merge_output_allocator_popcount8(free_leaf);
		const id_t live_dense = 2 - free_dense[0] - free_dense[1];
		if (live_contexts > peak_contexts) peak_contexts = live_contexts;
		if (live_leaves > peak_leaves) peak_leaves = live_leaves;
		if (live_dense > peak_dense) peak_dense = live_dense;

		// Refill after consuming the held tokens.  This keeps one command and
		// one completion buffered without reducing II, and prevents a FIFO
		// read from feeding the allocator state transition combinationally in
		// the same cycle.
		if (!command_valid && !command_done) {
			MergeAllocator8CommandToken token;
			if (command_input.try_read(token)) {
				if (token[128]) command_done = true;
				else { held_command = token; command_valid = true; }
			}
		}
		if (!eor_valid && !eor_done) {
			MergeAllocator8EorToken token;
			if (eor_input.try_read(token)) {
				if (token[64]) eor_done = true;
				else { held_eor = token; eor_valid = true; }
			}
		}
	}

	MergeAllocator8AckToken ack_end = 0;
	ack_end[64] = 1;
	ack_output.write(ack_end);
	ap_uint<512> statistics = 0;
	statistics.range(31, 0) = accepted_count;
	statistics.range(63, 32) = completed_count;
	statistics.range(95, 64) = merge_count;
	statistics.range(127, 96) = dense_count;
	statistics.range(159, 128) = peak_contexts;
	statistics.range(191, 160) = peak_leaves;
	statistics.range(223, 192) = peak_dense;
	statistics.range(255, 224) = level_count[0];
	statistics.range(287, 256) = level_count[1];
	statistics.range(319, 288) = level_count[2];
	statistics.range(351, 320) = level_count[3];
	statistics.range(383, 352) = stale_count;
	statistics.range(415, 384) = blocked_cycles;
	statistics.range(447, 416) = route_errors;
	statistics.range(479, 448) = free_leaf;
	statistics.range(481, 480) = free_dense;
	statistics.range(511, 482) = busy;
	statistics_output.write(statistics);
}

void unified_full_dispatch_broadcast(
		tapa::istream<MergeAllocator8DispatchToken>& dispatch_input,
		tapa::mmap<const id_t> global_bases,
		tapa::ostream<UnifiedSourceDispatchToken>& source_dispatch0,
		tapa::ostream<UnifiedSourceDispatchToken>& source_dispatch1,
		tapa::ostream<UnifiedSourceDispatchToken>& source_dispatch2,
		tapa::ostream<UnifiedSourceDispatchToken>& source_dispatch3,
		tapa::ostream<UnifiedSourceDispatchToken>& source_dispatch4,
		tapa::ostream<UnifiedSourceDispatchToken>& source_dispatch5,
		tapa::ostream<UnifiedSourceDispatchToken>& source_dispatch6,
		tapa::ostream<UnifiedSourceDispatchToken>& source_dispatch7,
		tapa::ostream<BaseToken>& base0, tapa::ostream<BaseToken>& base1,
		tapa::ostream<BaseToken>& base2, tapa::ostream<BaseToken>& base3,
		tapa::ostream<BaseToken>& base4, tapa::ostream<BaseToken>& base5,
		tapa::ostream<BaseToken>& base6, tapa::ostream<BaseToken>& base7,
		tapa::ostream<BaseToken>& base8, tapa::ostream<BaseToken>& base9,
		tapa::ostream<BaseToken>& base10, tapa::ostream<BaseToken>& base11,
		tapa::ostream<BaseToken>& base12, tapa::ostream<BaseToken>& base13,
		tapa::ostream<BaseToken>& base14,
		tapa::ostream<BaseToken>& dense_base0,
		tapa::ostream<BaseToken>& dense_base1,
		tapa::ostream<BaseToken>& heavy_base) {
	ap_uint<16> sequence = 0;
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		const MergeAllocator8DispatchToken dispatch_token
			= dispatch_input.read();
		UnifiedSourceDispatchToken source_token = 0;
		if (dispatch_token[128]) {
			source_token[144] = 1;
			source_dispatch0.write(source_token);
			source_dispatch1.write(source_token);
			source_dispatch2.write(source_token);
			source_dispatch3.write(source_token);
			source_dispatch4.write(source_token);
			source_dispatch5.write(source_token);
			source_dispatch6.write(source_token);
			source_dispatch7.write(source_token);
			const BaseToken base_end = end_base();
			base0.write(base_end); base1.write(base_end); base2.write(base_end);
			base3.write(base_end); base4.write(base_end); base5.write(base_end);
			base6.write(base_end); base7.write(base_end); base8.write(base_end);
			base9.write(base_end); base10.write(base_end); base11.write(base_end);
			base12.write(base_end); base13.write(base_end); base14.write(base_end);
			dense_base0.write(base_end); dense_base1.write(base_end);
			heavy_base.write(base_end);
			done = true;
			continue;
		}
		const ap_uint<128> dispatch = dispatch_token.range(127, 0);
		const id_t row = dispatch.range(31, 0);
		const id_t descriptor = dispatch.range(63, 32);
		const bool dense = dispatch[121];
		const bool heavy = dispatch[123];
		const bool wide = dispatch[124];
		BaseToken base = 0;
		base.range(31, 0) = row;
		base.range(63, 32) = global_bases[descriptor];
		if (dense) {
			if (wide) {
				dense_base0.write(base);
				BaseToken upper = base;
				upper.range(63, 32) = (id_t)base.range(63, 32) + 32768;
				dense_base1.write(upper);
			} else if (dispatch[122]) dense_base1.write(base);
			else dense_base0.write(base);
		} else if (heavy) {
			heavy_base.write(base);
		} else {
			const ap_uint<2> level = dispatch.range(92, 91);
			const ap_uint<3> leaf_base = dispatch.range(95, 93) << level;
			const ap_uint<4> exit = level == 0 ? (ap_uint<4>)leaf_base
				: level == 1 ? (ap_uint<4>)(8 + (leaf_base >> 1))
				: level == 2 ? (ap_uint<4>)(12 + (leaf_base >> 2))
				: (ap_uint<4>)14;
			write_base(base, exit, base0, base1, base2, base3, base4, base5,
				base6, base7, base8, base9, base10, base11, base12, base13,
				base14);
		}
		source_token.range(127, 0) = dispatch;
		source_token.range(143, 128) = sequence;
		source_dispatch0.write(source_token);
		source_dispatch1.write(source_token);
		source_dispatch2.write(source_token);
		source_dispatch3.write(source_token);
		source_dispatch4.write(source_token);
		source_dispatch5.write(source_token);
		source_dispatch6.write(source_token);
		source_dispatch7.write(source_token);
		++sequence;
	}
}

void unified_full_source_read(
		tapa::mmap<const ap_uint<64> > source_memory,
		ap_uint<3> source_index,
		tapa::istream<UnifiedSourceDispatchToken>& dispatch_input,
		tapa::ostream<UnifiedRoutedLeafToken>& merge_output,
		tapa::ostream<UnifiedFullDenseSourceToken>& dense_output,
		tapa::ostream<UnifiedFullDenseSourceToken>& heavy_output) {
	id_t cursor = 0;
	bool done = false;
	while (!done) {
		const UnifiedSourceDispatchToken source_dispatch
			= dispatch_input.read();
		if (source_dispatch[144]) {
			UnifiedRoutedLeafToken merge_end = 0;
			merge_end[132] = 1;
			merge_output.write(merge_end);
			dense_output.write(unified_full_dense_source_end());
			heavy_output.write(unified_full_dense_source_end());
			done = true;
			continue;
		}
		const ap_uint<128> dispatch = source_dispatch.range(127, 0);
		const ap_uint<16> sequence = source_dispatch.range(143, 128);
		const id_t row = dispatch.range(31, 0);
		const ap_uint<8> source_mask = dispatch.range(71, 64);
		const ap_uint<3> context = dispatch.range(82, 80);
		const ap_uint<8> generation = dispatch.range(90, 83);
		const bool dense = dispatch[121];
		const bool heavy = dispatch[123];
		const bool wide = dispatch[124];
		const bool active = source_mask[source_index];
		id_t count = 0;
		if (active) count = source_memory[cursor++];

		if (dense || heavy) {
			for (id_t position = 0; position < count; ++position) {
#pragma HLS PIPELINE II=1
				UnifiedFullDenseSourceToken token = 0;
				token.range(63, 0) = source_memory[cursor++];
				token.range(95, 64) = row;
				token.range(98, 96) = context;
				token.range(106, 99) = generation;
				token[107] = dispatch[122];
				token.range(125, 110) = sequence;
				token[127] = heavy;
				token[128] = wide;
				if (heavy) heavy_output.write(token);
				else dense_output.write(token);
			}
			UnifiedFullDenseSourceToken eor = 0;
			eor.range(95, 64) = row;
			eor.range(98, 96) = context;
			eor.range(106, 99) = generation;
			eor[107] = dispatch[122];
			eor[108] = 1;
			eor.range(125, 110) = sequence;
			eor[127] = heavy;
			eor[128] = wide;
			if (heavy) heavy_output.write(eor);
			else dense_output.write(eor);
			continue;
		}

		const ap_uint<2> level = dispatch.range(92, 91);
		const ap_uint<3> leaf_base = dispatch.range(95, 93) << level;
		const ap_uint<4> active_count
			= allocator_merge_output_allocator_popcount8(source_mask);
		const ap_uint<4> width = allocation_width(source_mask);
		ap_uint<4> inactive_rank = 0;
		for (int source = 0; source < 8; ++source) {
#pragma HLS UNROLL
			if (source < source_index && !source_mask[source]) ++inactive_rank;
		}
		const bool padding = !active && inactive_rank < width - active_count;
		const ap_uint<3> target_leaf = active
			? unified_mapped_leaf(dispatch, source_index)
			: (ap_uint<3>)(leaf_base + active_count + inactive_rank);
		for (id_t position = 0; position < count; ++position) {
#pragma HLS PIPELINE II=1
			const ap_uint<64> numerical = source_memory[cursor++];
			MergeForwardTree8Token token = 0;
			token.range(127, 0) = make_tree_item(numerical, row, context,
				generation, level, false, sequence);
			UnifiedRoutedLeafToken routed = 0;
			routed.range(128, 0) = token;
			routed.range(131, 129) = target_leaf;
			merge_output.write(routed);
		}
		if (active || padding) {
			MergeForwardTree8Token eor = 0;
			eor.range(127, 0) = make_tree_item(0, row, context, generation,
				level, true, sequence);
			UnifiedRoutedLeafToken routed = 0;
			routed.range(128, 0) = eor;
			routed.range(131, 129) = target_leaf;
			merge_output.write(routed);
		}
	}
}

// DENSE and heavy-MERGE packets share one physical
// segmented reducer.  The reducer is stateless across packets; keyed carry and
// the two DENSE accumulators retain their independent cross-packet state.
void unified_full_segmented_pair_mux(
		tapa::istream<Segmented8SharedToken>& numerical0,
		tapa::istream<UnifiedFullDenseTag>& tags0,
		tapa::istream<Segmented8SharedToken>& numerical1,
		tapa::istream<UnifiedFullDenseTag>& tags1,
		tapa::ostream<Segmented8SharedToken>& numerical,
		tapa::ostream<UnifiedFullDenseTag>& tags) {
	bool ended[2] = {false, false};
	ap_uint<1> turn = 0;
	while (!(ended[0] && ended[1])) {
#pragma HLS PIPELINE II=1
		bool moved = false;
		Segmented8SharedToken packet;
		if (!ended[turn] && (turn == 0
				? numerical0.try_read(packet) : numerical1.try_read(packet))) {
			const UnifiedFullDenseTag tag
				= turn == 0 ? tags0.read() : tags1.read();
			if (packet[556] || tag[46]) ended[turn] = true;
			else { numerical.write(packet); tags.write(tag); }
			turn = !turn;
			moved = true;
		}
		const ap_uint<1> other = !turn;
		if (!moved && !ended[other] && (other == 0
				? numerical0.try_read(packet) : numerical1.try_read(packet))) {
			const UnifiedFullDenseTag tag
				= other == 0 ? tags0.read() : tags1.read();
			if (packet[556] || tag[46]) ended[other] = true;
			else { numerical.write(packet); tags.write(tag); }
			turn = !other;
		}
	}
	Segmented8SharedToken numerical_end = 0;
	numerical_end[556] = 1;
	numerical.write(numerical_end);
	tags.write(unified_full_dense_tag_end());
}

// Four sorted [fp32|local-column] items, valid count, allocator identity,
// EOR/terminal and a protocol-error bit.  This is the HLS form of AiSpGEMM's
// 4x4 P-merger array: each node merges two sorted streams four items at a time
// and feeds its residual heads into the following cycle.
using UnifiedHeavyVector4Token = ap_uint<323>;

static UnifiedHeavyVector4Token unified_full_heavy_vector_end() {
#pragma HLS INLINE
	UnifiedHeavyVector4Token token = 0;
	token[321] = 1;
	return token;
}

static ap_uint<64> unified_full_heavy_vector_item(ap_uint<256> word,
		ap_uint<2> index) {
#pragma HLS INLINE
	switch ((unsigned)index) {
	case 0: return word.range(63, 0);
	case 1: return word.range(127, 64);
	case 2: return word.range(191, 128);
	default: return word.range(255, 192);
	}
}

static UnifiedHeavyVector4Token unified_full_make_heavy_vector(
		ap_uint<256> word, ap_uint<3> count,
		const UnifiedFullDenseSourceToken& identity,
		bool eor, bool error) {
#pragma HLS INLINE
	UnifiedHeavyVector4Token token = 0;
	token.range(255, 0) = word;
	token.range(258, 256) = count;
	token.range(290, 259) = identity.range(95, 64);
	token.range(293, 291) = identity.range(98, 96);
	token.range(301, 294) = identity.range(106, 99);
	token.range(317, 302) = identity.range(125, 110);
	token[318] = eor;
	token[319] = error || identity[126];
	token[320] = 1;
	return token;
}

// Convert one sorted shard stream into compact sorted four-item vectors.
void unified_full_heavy_source_pack4(
		tapa::istream<UnifiedFullDenseSourceToken>& input,
		tapa::ostream<UnifiedHeavyVector4Token>& output) {
	ap_uint<256> word = 0;
	ap_uint<3> count = 0;
	UnifiedFullDenseSourceToken identity = 0;
	bool active = false;
	bool row_error = false;
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		const UnifiedFullDenseSourceToken token = input.read();
		if (token[109]) {
			if (count != 0)
				output.write(unified_full_make_heavy_vector(
					word, count, identity, false, true));
			output.write(unified_full_heavy_vector_end());
			done = true;
			continue;
		}
		if (token[108]) {
			if (active && (token.range(95, 64) != identity.range(95, 64)
					|| token.range(98, 96) != identity.range(98, 96)
					|| token.range(106, 99) != identity.range(106, 99)
					|| token.range(125, 110) != identity.range(125, 110)))
				row_error = true;
			if (count != 0)
				output.write(unified_full_make_heavy_vector(
					word, count, identity, false, row_error));
			output.write(unified_full_make_heavy_vector(
				0, 0, token, true, row_error));
			word = 0;
			count = 0;
			active = false;
			row_error = false;
			continue;
		}
		if (!active) {
			identity = token;
			active = true;
		} else if (token.range(95, 64) != identity.range(95, 64)
				|| token.range(98, 96) != identity.range(98, 96)
				|| token.range(106, 99) != identity.range(106, 99)
				|| token.range(125, 110) != identity.range(125, 110)) {
			row_error = true;
		}
		word.range((unsigned)count * 64 + 63, (unsigned)count * 64)
			= token.range(63, 0);
		row_error |= token[126];
		++count;
		if (count == 4) {
			output.write(unified_full_make_heavy_vector(
				word, count, identity, false, row_error));
			word = 0;
			count = 0;
		}
	}
}

static bool unified_full_heavy_vector_identity_equal(
		const UnifiedHeavyVector4Token& lhs,
		const UnifiedHeavyVector4Token& rhs) {
#pragma HLS INLINE
	return lhs.range(317, 259) == rhs.range(317, 259);
}

static UnifiedFullDenseSourceToken unified_full_heavy_vector_identity(
		const UnifiedHeavyVector4Token& token) {
#pragma HLS INLINE
	UnifiedFullDenseSourceToken identity = 0;
	identity.range(95, 64) = token.range(290, 259);
	identity.range(98, 96) = token.range(293, 291);
	identity.range(106, 99) = token.range(301, 294);
	identity.range(125, 110) = token.range(317, 302);
	identity[126] = token[319];
	identity[127] = 1;
	return identity;
}

// A valid bit above the numerical payload makes an invalid padding item sort
// after every legal 16-bit column, including column 65535.
using UnifiedHeavySortItem = ap_uint<65>;

static void unified_full_heavy_compare_swap(
		UnifiedHeavySortItem& lhs, UnifiedHeavySortItem& rhs) {
#pragma HLS INLINE
	const bool swap = (!lhs[64] && rhs[64])
		|| (lhs[64] && rhs[64]
			&& lhs.range(15, 0) > rhs.range(15, 0));
	if (swap) {
		const UnifiedHeavySortItem temporary = lhs;
		lhs = rhs;
		rhs = temporary;
	}
}

// After the cross comparisons of an eight-item bitonic merge, each four-item
// half is itself bitonic rather than arbitrary.  Two compare-exchange levels
// are therefore sufficient; using a general three-level sort4 here adds one
// unnecessary 16-bit compare/mux to the feedback recurrence.
static void unified_full_heavy_bitonic_merge4(
		UnifiedHeavySortItem item[4]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=item complete
	unified_full_heavy_compare_swap(item[0], item[2]);
	unified_full_heavy_compare_swap(item[1], item[3]);
	unified_full_heavy_compare_swap(item[0], item[1]);
	unified_full_heavy_compare_swap(item[2], item[3]);
}

// Merge two sorted four-item vectors.  Pairing lhs[i] with rhs[3-i] is the
// first level of an eight-item bitonic merge; the low and high groups are then
// sorted by independent fixed sort4 networks.  The high group is the only
// feedback state of the packet-frontier P-merger below.
static void unified_full_heavy_merge_vectors4(
		ap_uint<256> lhs_word, ap_uint<3> lhs_count,
		ap_uint<256> rhs_word, ap_uint<3> rhs_count,
		ap_uint<256>& low_word, ap_uint<3>& low_count,
		ap_uint<256>& high_word, ap_uint<3>& high_count) {
#pragma HLS INLINE
	UnifiedHeavySortItem low[4];
	UnifiedHeavySortItem high[4];
#pragma HLS ARRAY_PARTITION variable=low complete
#pragma HLS ARRAY_PARTITION variable=high complete
	for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
		UnifiedHeavySortItem lhs = 0;
		UnifiedHeavySortItem rhs = 0;
		lhs.range(63, 0) = lhs_word.range(lane * 64 + 63, lane * 64);
		lhs[64] = lane < lhs_count;
		const int reverse_lane = 3 - lane;
		rhs.range(63, 0) = rhs_word.range(
			reverse_lane * 64 + 63, reverse_lane * 64);
		rhs[64] = reverse_lane < rhs_count;
		unified_full_heavy_compare_swap(lhs, rhs);
		low[lane] = lhs;
		high[lane] = rhs;
	}
	unified_full_heavy_bitonic_merge4(low);
	unified_full_heavy_bitonic_merge4(high);
	low_word = 0;
	high_word = 0;
	for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
		low_word.range(lane * 64 + 63, lane * 64) = low[lane].range(63, 0);
		high_word.range(lane * 64 + 63, lane * 64) = high[lane].range(63, 0);
	}
	const ap_uint<4> total = lhs_count + rhs_count;
	low_count = total < 4 ? (ap_uint<3>)total : (ap_uint<3>)4;
	high_count = total > 4 ? (ap_uint<3>)(total - 4) : (ap_uint<3>)0;
}

static ap_uint<17> unified_full_heavy_packet_frontier(
		ap_uint<256> word, ap_uint<3> count) {
#pragma HLS INLINE
	// source_pack4 emits a short packet only immediately before its EOR.
	// Treating that packet's boundary as +infinity prevents another output from
	// passing its unseen EOR.  A full packet uses its last (largest) column.
	if (count != 4) return (ap_uint<17>)65536;
	ap_uint<17> frontier = 0;
	frontier.range(15, 0) = word.range(207, 192);
	return frontier;
}

static UnifiedHeavyVector4Token unified_full_heavy_row_boundary(
		const UnifiedHeavyVector4Token& identity, bool error) {
#pragma HLS INLINE
	UnifiedHeavyVector4Token boundary = identity;
	boundary.range(255, 0) = 0;
	boundary.range(258, 256) = 0;
	boundary[318] = 1;
	boundary[319] = error;
	boundary[320] = 1;
	boundary[321] = 0;
	return boundary;
}

// Packet-frontier P-merger for two globally sorted vector streams.  After the
// first two packets are merged, only the sorted high four items feed back.
// The input whose previously consumed packet has the smaller maximum column is
// advanced next.  Therefore the state-selection recurrence is one 17-bit
// comparison; it no longer contains index -> rank -> safe-count -> popcount ->
// index.  This is the packet-level P-merger organization used to make the
// heavy-row path both four-wide and clockable.
void unified_full_heavy_merge4(
		tapa::istream<UnifiedHeavyVector4Token>& input0,
		tapa::istream<UnifiedHeavyVector4Token>& input1,
		tapa::ostream<UnifiedHeavyVector4Token>& output) {
	bool finished = false;
	while (!finished) {
		UnifiedHeavyVector4Token first0 = input0.read();
		UnifiedHeavyVector4Token first1 = input1.read();
		if (first0[321] || first1[321]) {
			finished = true;
			continue;
		}

		UnifiedHeavyVector4Token identity = first0;
		bool row_error = first0[319] || first1[319]
			|| !unified_full_heavy_vector_identity_equal(first0, first1);
		const bool first_eor0 = first0[318];
		const bool first_eor1 = first1[318];
		if (first_eor0 && first_eor1) {
			output.write(unified_full_heavy_row_boundary(identity, row_error));
			continue;
		}

		// If one child is empty, the other child is already globally sorted and
		// can be forwarded without using the feedback network.
		if (first_eor0 || first_eor1) {
			const bool forward0 = !first_eor0;
			UnifiedHeavyVector4Token token = forward0 ? first0 : first1;
			bool row_done = false;
			while (!row_done) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1048576 avg=1024
				if (token[318]) {
					row_error |= token[319]
						|| !unified_full_heavy_vector_identity_equal(identity, token);
					row_done = true;
				} else {
					row_error |= token[319]
						|| !unified_full_heavy_vector_identity_equal(identity, token);
					token[319] = row_error;
					output.write(token);
					token = forward0 ? input0.read() : input1.read();
				}
			}
			output.write(unified_full_heavy_row_boundary(identity, row_error));
			continue;
		}

		ap_uint<256> low_word = 0;
		ap_uint<256> carry_word = 0;
		ap_uint<3> low_count = 0;
		ap_uint<3> carry_count = 0;
		unified_full_heavy_merge_vectors4(
			first0.range(255, 0), first0.range(258, 256),
			first1.range(255, 0), first1.range(258, 256),
			low_word, low_count, carry_word, carry_count);
		if (low_count != 0) {
			UnifiedHeavyVector4Token low = identity;
			low.range(255, 0) = low_word;
			low.range(258, 256) = low_count;
			low[318] = 0;
			low[319] = row_error;
			output.write(low);
		}

		ap_uint<17> frontier0 = unified_full_heavy_packet_frontier(
			first0.range(255, 0), first0.range(258, 256));
		ap_uint<17> frontier1 = unified_full_heavy_packet_frontier(
			first1.range(255, 0), first1.range(258, 256));
		bool done0 = false;
		bool done1 = false;
		while (!(done0 && done1)) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=2 max=1048576 avg=4096
			const bool take0 = !done0 && (done1 || frontier0 <= frontier1);
			UnifiedHeavyVector4Token token = take0
				? input0.read() : input1.read();
			row_error |= token[319]
				|| !unified_full_heavy_vector_identity_equal(identity, token);
			if (token[318] || token[321]) {
				if (take0) {
					done0 = true;
					frontier0 = 65536;
				} else {
					done1 = true;
					frontier1 = 65536;
				}
				row_error |= token[321];
				continue;
			}

			ap_uint<256> next_carry_word = 0;
			ap_uint<3> next_carry_count = 0;
			unified_full_heavy_merge_vectors4(
				carry_word, carry_count,
				token.range(255, 0), token.range(258, 256),
				low_word, low_count, next_carry_word, next_carry_count);
			carry_word = next_carry_word;
			carry_count = next_carry_count;
			if (take0)
				frontier0 = unified_full_heavy_packet_frontier(
					token.range(255, 0), token.range(258, 256));
			else
				frontier1 = unified_full_heavy_packet_frontier(
					token.range(255, 0), token.range(258, 256));
			if (low_count != 0) {
				UnifiedHeavyVector4Token low = identity;
				low.range(255, 0) = low_word;
				low.range(258, 256) = low_count;
				low[318] = 0;
				low[319] = row_error;
				output.write(low);
			}
		}
		if (carry_count != 0) {
			UnifiedHeavyVector4Token carry = identity;
			carry.range(255, 0) = carry_word;
			carry.range(258, 256) = carry_count;
			carry[318] = 0;
			carry[319] = row_error;
			output.write(carry);
		}
		output.write(unified_full_heavy_row_boundary(identity, row_error));
	}
	output.write(unified_full_heavy_vector_end());
}

void unified_full_heavy_vector_to_segmented(
		tapa::istream<UnifiedHeavyVector4Token>& input,
		tapa::ostream<Segmented8SharedToken>& numerical,
		tapa::ostream<UnifiedFullDenseTag>& tags) {
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		UnifiedHeavyVector4Token vector;
		if (!input.try_read(vector)) continue;
		if (vector[321]) {
			Segmented8SharedToken numerical_end = 0;
			numerical_end[556] = 1;
			numerical.write(numerical_end);
			tags.write(unified_full_dense_tag_end());
			done = true;
			continue;
		}
		Segmented8SharedToken packet = 0;
		packet.range(255, 0) = vector.range(255, 0);
		const ap_uint<3> count = vector.range(258, 256);
		packet.range(551, 544) = unified_full_low_mask8(count);
		packet.range(543, 512) = vector.range(290, 259);
		packet.range(553, 552) = SEGMENTED8_SHARED_MERGE;
		packet[554] = vector[318];
		UnifiedFullDenseTag tag = 0;
		tag.range(31, 0) = vector.range(290, 259);
		tag.range(34, 32) = vector.range(293, 291);
		tag.range(42, 35) = vector.range(301, 294);
		tag[44] = vector[318];
		tag[45] = vector[319];
		tag[47] = 1;
		numerical.write(packet);
		tags.write(tag);
	}
}

void unified_full_dense_pack8(
		tapa::istream<UnifiedFullDenseSourceToken>& input0,
		tapa::istream<UnifiedFullDenseSourceToken>& input1,
		tapa::istream<UnifiedFullDenseSourceToken>& input2,
		tapa::istream<UnifiedFullDenseSourceToken>& input3,
		tapa::istream<UnifiedFullDenseSourceToken>& input4,
		tapa::istream<UnifiedFullDenseSourceToken>& input5,
		tapa::istream<UnifiedFullDenseSourceToken>& input6,
		tapa::istream<UnifiedFullDenseSourceToken>& input7,
		tapa::ostream<Segmented8SharedToken>& numerical,
		tapa::ostream<UnifiedFullDenseTag>& tags) {
	UnifiedFullDenseSourceToken held[8] = {};
	bool valid[8] = {};
	bool ended[8] = {};
#pragma HLS ARRAY_PARTITION variable=held complete
#pragma HLS ARRAY_PARTITION variable=valid complete
#pragma HLS ARRAY_PARTITION variable=ended complete
	while (!(ended[0] && ended[1] && ended[2] && ended[3]
			&& ended[4] && ended[5] && ended[6] && ended[7]
			&& !valid[0] && !valid[1] && !valid[2] && !valid[3]
			&& !valid[4] && !valid[5] && !valid[6] && !valid[7])) {
#pragma HLS PIPELINE II=1
#define UNIFIED_FULL_DENSE_REFILL(L, INPUT) \
		if (!ended[L] && !valid[L]) { \
			UnifiedFullDenseSourceToken token; \
			if (INPUT.try_read(token)) { \
				if (token[109]) ended[L] = true; \
				else { held[L] = token; valid[L] = true; } \
			} \
		}
		UNIFIED_FULL_DENSE_REFILL(0, input0)
		UNIFIED_FULL_DENSE_REFILL(1, input1)
		UNIFIED_FULL_DENSE_REFILL(2, input2)
		UNIFIED_FULL_DENSE_REFILL(3, input3)
		UNIFIED_FULL_DENSE_REFILL(4, input4)
		UNIFIED_FULL_DENSE_REFILL(5, input5)
		UNIFIED_FULL_DENSE_REFILL(6, input6)
		UNIFIED_FULL_DENSE_REFILL(7, input7)
#undef UNIFIED_FULL_DENSE_REFILL
		bool all_valid = true;
		bool all_eor = true;
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			all_valid &= valid[lane];
			all_eor &= valid[lane] && held[lane][108];
		}
		if (!all_valid) continue;

		const id_t row = held[0].range(95, 64);
		const ap_uint<3> context = held[0].range(98, 96);
		const ap_uint<8> generation = held[0].range(106, 99);
		const ap_uint<1> slot = held[0][107];
		const ap_uint<16> sequence = held[0].range(125, 110);
		const bool heavy = held[0][127];
		const bool wide = held[0][128];
		bool protocol_error = false;
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			protocol_error |= held[lane].range(95, 64) != row
				|| held[lane].range(98, 96) != context
				|| held[lane].range(106, 99) != generation
				|| held[lane][107] != slot
				|| held[lane].range(125, 110) != sequence
				|| held[lane][127] != heavy
				|| held[lane][128] != wide
				|| held[lane][126];
		}
		Segmented8SharedToken packet = 0;
		packet.range(543, 512) = row;
		packet.range(553, 552) = heavy
			? (ap_uint<2>)SEGMENTED8_SHARED_MERGE
			: (ap_uint<2>)SEGMENTED8_SHARED_DENSE;
		UnifiedFullDenseTag tag = 0;
		tag.range(31, 0) = row;
		tag.range(34, 32) = context;
		tag.range(42, 35) = generation;
		tag[43] = slot;
		tag[45] = protocol_error;
		tag[47] = heavy;
		tag[48] = wide;
		if (all_eor) {
			packet[554] = 1;
			tag[44] = 1;
			for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
				valid[lane] = false;
			}
		} else {
			ap_uint<8> mask = 0;
			for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
				if (!held[lane][108]) {
					packet.range(lane * 64 + 63, lane * 64)
						= held[lane].range(63, 0);
					mask[lane] = 1;
					valid[lane] = false;
				}
			}
			packet.range(551, 544) = mask;
		}
		numerical.write(packet);
		tags.write(tag);
	}
	Segmented8SharedToken numerical_end = 0;
	numerical_end[556] = 1;
	numerical.write(numerical_end);
	tags.write(unified_full_dense_tag_end());
}

void unified_full_dense_restore(
		tapa::istream<Segmented8SharedToken>& numerical,
		tapa::istream<UnifiedFullDenseTag>& tags,
		tapa::ostream<DenseDualContextToken>& dense_output0,
		tapa::ostream<DenseDualContextToken>& dense_output1,
		tapa::ostream<MergeKeyedCarry8InputToken>& heavy_output) {
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		// numerical and tags are a one-to-one pair.  Test both before either
		// blocking read so an inter-row gap is a bubble and cannot gate the
		// two-stage restore pipeline while an EOR is still in flight.
		if (numerical.empty() || tags.empty()) continue;
		const Segmented8SharedToken reduced = numerical.read();
		const UnifiedFullDenseTag tag = tags.read();
		if (reduced[556] || tag[46]) {
			DenseDualContextToken terminal = 0;
			terminal[566] = 1;
			dense_output0.write(terminal);
			dense_output1.write(terminal);
			MergeKeyedCarry8InputToken heavy_terminal = 0;
			heavy_terminal[576] = 1;
			heavy_output.write(heavy_terminal);
			done = true;
			continue;
		}
		const bool heavy = tag[47];
		const bool identity_error
			= reduced.range(543, 512) != tag.range(31, 0)
			|| reduced.range(553, 552) != (heavy
				? (ap_uint<2>)SEGMENTED8_SHARED_MERGE
				: (ap_uint<2>)SEGMENTED8_SHARED_DENSE)
			|| reduced[554] != tag[44];
		if (heavy) {
			MergeKeyedCarry8InputToken token = 0;
			token.range(511, 0) = reduced.range(511, 0);
			ap_uint<64> meta = 0;
			meta.range(31, 0) = tag.range(31, 0);
			meta.range(39, 32) = reduced.range(551, 544);
			meta.range(42, 40) = tag.range(34, 32);
			meta.range(50, 43) = tag.range(42, 35);
			meta[51] = tag[44];
			meta[52] = tag[45] || identity_error;
			token.range(575, 512) = meta;
			heavy_output.write(token);
			continue;
		}
		const bool wide = tag[48];
		DenseDualContextToken lower = 0;
		DenseDualContextToken upper = 0;
		lower.range(543, 512) = upper.range(543, 512) = tag.range(31, 0);
		lower[552] = 0;
		upper[552] = 1;
		lower[553] = upper[553] = tag[44];
		lower.range(556, 554) = upper.range(556, 554) = tag.range(34, 32);
		lower.range(564, 557) = upper.range(564, 557) = tag.range(42, 35);
		lower[567] = upper[567] = wide;
		if (!wide) {
			DenseDualContextToken token = lower;
			token.range(511, 0) = reduced.range(511, 0);
			token.range(551, 544) = reduced.range(551, 544);
			token[552] = tag[43];
			token[565] = tag[45] || identity_error;
			if (tag[43]) dense_output1.write(token);
			else dense_output0.write(token);
			continue;
		}

		ap_uint<8> lower_mask = 0;
		ap_uint<8> upper_mask = 0;
		bool range_error = false;
		const ap_uint<8> mask = reduced.range(551, 544);
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			if (!mask[lane]) continue;
			ap_uint<64> item = reduced.range(lane * 64 + 63, lane * 64);
			const ap_uint<32> column = item.range(31, 0);
			range_error |= column.range(31, 16) != 0;
			item.range(31, 0) = column.range(14, 0);
			if (column[15]) {
				upper.range(lane * 64 + 63, lane * 64) = item;
				upper_mask[lane] = 1;
			} else {
				lower.range(lane * 64 + 63, lane * 64) = item;
				lower_mask[lane] = 1;
			}
		}
		lower.range(551, 544) = lower_mask;
		upper.range(551, 544) = upper_mask;
		lower[565] = upper[565] = tag[45] || identity_error || range_error;
		// Every wide row sends one EOR to each physical engine.  Numerical
		// packets are sent only to the half that owns at least one lane.
		if (lower_mask != 0 || tag[44]) dense_output0.write(lower);
		if (upper_mask != 0 || tag[44]) dense_output1.write(upper);
	}
}

// The heavy MERGE microengine shares the packet-local segmented reducer with
// DENSE, then keeps one open key per logical row context across packet
// boundaries.  Convert its post-flush result to the existing compact-token
// shape only after the carry has been resolved; completion is still delayed
// until the compacted packet reaches the common output fabric.
void unified_full_heavy_restore(
		tapa::istream<CarryResultToken>& input,
		tapa::ostream<DenseDualContextToken>& output,
		tapa::ostream<ap_uint<256> >& statistics_output) {
	id_t packets = 0;
	id_t items = 0;
	id_t eors = 0;
	id_t boundary_merges = 0;
	id_t carry_flushes = 0;
	id_t zero_carries = 0;
	id_t protocol_errors = 0;
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		CarryResultToken result;
		if (!input.try_read(result)) continue;
		if (result[568]) {
			DenseDualContextToken terminal = 0;
			terminal[566] = 1;
			output.write(terminal);
			done = true;
			continue;
		}
		DenseDualContextToken token = 0;
		token.range(511, 0) = result.range(511, 0);
		token.range(543, 512) = result.range(551, 520);
		token.range(551, 544) = result.range(519, 512);
		token[553] = result[563];
		token.range(556, 554) = result.range(554, 552);
		token.range(564, 557) = result.range(562, 555);
		token[565] = result[565];
		output.write(token);
		++packets;
		items += unified_full_popcount8(result.range(519, 512));
		eors += result[563];
		boundary_merges += result[564];
		protocol_errors += result[565];
		zero_carries += result[566];
		carry_flushes += result[567];
	}
	ap_uint<256> statistics = 0;
	statistics.range(31, 0) = packets;
	statistics.range(63, 32) = items;
	statistics.range(95, 64) = eors;
	statistics.range(127, 96) = boundary_merges;
	statistics.range(159, 128) = carry_flushes;
	statistics.range(191, 160) = zero_carries;
	statistics.range(223, 192) = protocol_errors;
	statistics_output.write(statistics);
}

// A single loop that repeatedly inserts valid lanes into a 512-bit word
// synthesizes as a deep priority-select chain.  These three streaming stages
// form the same stable compaction as a balanced 2 -> 4 -> 8 network.  Each
// stage accepts one token per cycle and provides an explicit register boundary.
void unified_full_dense_compact2(
		tapa::istream<DenseDualContextToken>& input,
		tapa::ostream<DenseDualContextToken>& output) {
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		DenseDualContextToken token;
		if (!input.try_read(token)) continue;
		if (token[566]) {
			output.write(token);
			done = true;
			continue;
		}
		if (token[553]) {
			output.write(token);
			continue;
		}
		const ap_uint<8> mask = token.range(551, 544);
		ap_uint<512> packed = 0;
		ap_uint<8> compact_mask = 0;
		for (int pair = 0; pair < 4; ++pair) {
#pragma HLS UNROLL
			const int lo = pair * 2;
			const ap_uint<64> item0
				= token.range(lo * 64 + 63, lo * 64);
			const ap_uint<64> item1
				= token.range((lo + 1) * 64 + 63, (lo + 1) * 64);
			packed.range(lo * 64 + 63, lo * 64)
				= mask[lo] ? item0 : item1;
			packed.range((lo + 1) * 64 + 63, (lo + 1) * 64) = item1;
			compact_mask[lo] = mask[lo] || mask[lo + 1];
			compact_mask[lo + 1] = mask[lo] && mask[lo + 1];
		}
		token.range(511, 0) = packed;
		token.range(551, 544) = compact_mask;
		output.write(token);
	}
}

void unified_full_dense_compact4(
		tapa::istream<DenseDualContextToken>& input,
		tapa::ostream<DenseDualContextToken>& output) {
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		DenseDualContextToken token;
		if (!input.try_read(token)) continue;
		if (token[566]) {
			output.write(token);
			done = true;
			continue;
		}
		if (token[553]) {
			output.write(token);
			continue;
		}
		const ap_uint<8> mask = token.range(551, 544);
		ap_uint<512> packed = 0;
		ap_uint<8> compact_mask = 0;
		for (int half = 0; half < 2; ++half) {
#pragma HLS UNROLL
			const int base = half * 4;
			ap_uint<64> item[4];
			ap_uint<64> compact_item[4] = {};
#pragma HLS ARRAY_PARTITION variable=item complete
#pragma HLS ARRAY_PARTITION variable=compact_item complete
			for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
				item[lane] = token.range(
					(base + lane) * 64 + 63, (base + lane) * 64);
			}
			const ap_uint<2> low_count = mask[base] + mask[base + 1];
			const ap_uint<2> high_count = mask[base + 2] + mask[base + 3];
			switch ((unsigned)low_count) {
			case 0:
				compact_item[0] = item[2]; compact_item[1] = item[3];
				break;
			case 1:
				compact_item[0] = item[0]; compact_item[1] = item[2];
				compact_item[2] = item[3];
				break;
			default:
				compact_item[0] = item[0]; compact_item[1] = item[1];
				compact_item[2] = item[2]; compact_item[3] = item[3];
				break;
			}
			for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
				packed.range((base + lane) * 64 + 63,
					(base + lane) * 64) = compact_item[lane];
			}
			compact_mask.range(base + 3, base)
				= unified_full_low_mask4(low_count + high_count);
		}
		token.range(511, 0) = packed;
		token.range(551, 544) = compact_mask;
		output.write(token);
	}
}

void unified_full_dense_compact8(
		tapa::istream<DenseDualContextToken>& input,
		tapa::ostream<DenseDualContextToken>& output) {
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		DenseDualContextToken token;
		if (!input.try_read(token)) continue;
		if (token[566]) {
			output.write(token);
			done = true;
			continue;
		}
		if (token[553]) {
			output.write(token);
			continue;
		}
		const ap_uint<8> mask = token.range(551, 544);
		ap_uint<64> item[8];
		ap_uint<64> packed_item[8] = {};
#pragma HLS ARRAY_PARTITION variable=item complete
#pragma HLS ARRAY_PARTITION variable=packed_item complete
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			item[lane] = token.range(lane * 64 + 63, lane * 64);
		}
		const ap_uint<3> low_count = mask[0] + mask[1] + mask[2] + mask[3];
		const ap_uint<3> high_count = mask[4] + mask[5] + mask[6] + mask[7];
		for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
			packed_item[lane] = item[lane];
		}
		switch ((unsigned)low_count) {
		case 0:
			for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
				packed_item[lane] = item[lane + 4];
			}
			break;
		case 1:
			for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
				packed_item[lane + 1] = item[lane + 4];
			}
			break;
		case 2:
			for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
				packed_item[lane + 2] = item[lane + 4];
			}
			break;
		case 3:
			for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
				packed_item[lane + 3] = item[lane + 4];
			}
			break;
		default:
			for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
				packed_item[lane + 4] = item[lane + 4];
			}
			break;
		}
		ap_uint<512> packed = 0;
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			packed.range(lane * 64 + 63, lane * 64) = packed_item[lane];
		}
		token.range(511, 0) = packed;
		token.range(551, 544)
			= unified_full_low_mask8(low_count + high_count);
		output.write(token);
	}
}

void unified_full_dense_packetize(
		tapa::istream<DenseDualContextToken>& input,
		tapa::istream<BaseToken>& bases,
		ap_uint<1> slot, ap_uint<2> port_id,
		tapa::ostream<PacketBundle>& output,
		tapa::ostream<CompletionToken>& completions) {
	bool active = false;
	id_t active_row = 0;
	id_t active_base = 0;
	ap_uint<3> active_context = 0;
	ap_uint<8> active_generation = 0;
	bool active_wide = false;
	id_t active_nnz = 0;
	ap_uint<2> row_error = 0;
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		DenseDualContextToken token;
		if (!input.try_read(token)) continue;
		if (token[566]) {
			done = true;
			continue;
		}
		const id_t row = token.range(543, 512);
		const ap_uint<3> context = token.range(556, 554);
		const ap_uint<8> generation = token.range(564, 557);
		const bool eor = token[553];
		const bool wide = token[567];
		ap_uint<2> error = token[565] ? (ap_uint<2>)1 : (ap_uint<2>)0;
		if (token[552] != slot && error == 0) error = 1;
		if (!active) {
			const BaseToken base_token = bases.read();
			if (base_token[64]) {
				error = 2;
				active_base = 0;
			} else {
				active_base = base_token.range(63, 32);
				if (base_token.range(31, 0) != row) error = 2;
			}
			active = true;
			active_row = row;
			active_context = context;
			active_generation = generation;
			active_wide = wide;
			active_nnz = 0;
			row_error = 0;
		} else if (active_row != row || active_context != context
				|| active_generation != generation || active_wide != wide) {
			error = 3;
		}
		if (row_error == 0 && error != 0) row_error = error;

		if (eor) {
			output.write(unified_full_make_bundle(0,
				unified_full_make_record(row, active_nnz, 0, true, context,
					UNIFIED_FULL_OUTPUT_DENSE, slot, row_error, port_id,
					generation)));
			CompletionToken completion = 0;
			completion.range(63, 0) = unified_full_make_completion(row,
				context, generation, true, slot);
			completion[48] = wide;
			completions.write(completion);
			active = false;
			continue;
		}

		const ap_uint<8> mask = token.range(551, 544);
		ap_uint<512> packed = 0;
		ap_uint<8> overflow_mask = 0;
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			if (mask[lane]) {
				const ap_uint<64> local_item
					= token.range(lane * 64 + 63, lane * 64);
				const ap_uint<33> global_column
					= (ap_uint<33>)active_base + local_item.range(31, 0);
				overflow_mask[lane] = global_column[32];
				ap_uint<64> global_item = local_item;
				global_item.range(31, 0) = global_column.range(31, 0);
				// The preceding balanced compaction network guarantees that
				// valid lanes are contiguous, so this is a fixed slice write.
				packed.range(lane * 64 + 63, lane * 64) = global_item;
			}
		}
		const ap_uint<4> count = unified_full_popcount8(mask);
		// Overflow is impossible for a valid matrix/base pair.  Preserve the
		// packet shape and flag the whole row so malformed work cannot turn a
		// compact packet back into one with holes.
		if (overflow_mask != 0 && row_error == 0) row_error = 3;
		if (count != 0) {
			output.write(unified_full_make_bundle(packed,
				unified_full_make_record(row, 0, count, false, context,
					UNIFIED_FULL_OUTPUT_DENSE, slot, row_error, port_id,
					generation)));
			active_nnz += count;
		}
	}
	if (active) {
		output.write(unified_full_make_bundle(0,
			unified_full_make_record(active_row, active_nnz, 0, true,
				active_context, UNIFIED_FULL_OUTPUT_DENSE, slot, 3, port_id,
				active_generation)));
	}
	BaseToken base_end = bases.read();
	while (!base_end[64]) base_end = bases.read();
	output.write(end_bundle());
	completions.write(end_completion());
}

void unified_full_heavy_packetize(
		tapa::istream<DenseDualContextToken>& input,
		tapa::istream<BaseToken>& bases,
		tapa::ostream<PacketBundle>& output,
		tapa::ostream<CompletionToken>& completions) {
	bool active = false;
	id_t active_row = 0;
	id_t active_base = 0;
	ap_uint<3> active_context = 0;
	ap_uint<8> active_generation = 0;
	id_t active_nnz = 0;
	ap_uint<2> row_error = 0;
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		DenseDualContextToken token;
		if (!input.try_read(token)) continue;
		if (token[566]) {
			done = true;
			continue;
		}
		const id_t row = token.range(543, 512);
		const ap_uint<3> context = token.range(556, 554);
		const ap_uint<8> generation = token.range(564, 557);
		const bool eor = token[553];
		ap_uint<2> error = token[565] ? (ap_uint<2>)1 : (ap_uint<2>)0;
		if (!active) {
			const BaseToken base_token = bases.read();
			if (base_token[64]) {
				error = 2;
				active_base = 0;
			} else {
				active_base = base_token.range(63, 32);
				if (base_token.range(31, 0) != row) error = 2;
			}
			active = true;
			active_row = row;
			active_context = context;
			active_generation = generation;
			active_nnz = 0;
			row_error = 0;
		} else if (active_row != row || active_context != context
				|| active_generation != generation) {
			error = 3;
		}
		if (row_error == 0 && error != 0) row_error = error;

		const ap_uint<8> mask = token.range(551, 544);
		const ap_uint<4> count = unified_full_popcount8(mask);
		ap_uint<512> packed = 0;
		ap_uint<8> overflow_mask = 0;
		for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
			if (mask[lane]) {
				const ap_uint<64> local_item
					= token.range(lane * 64 + 63, lane * 64);
				const ap_uint<33> global_column
					= (ap_uint<33>)active_base + local_item.range(31, 0);
				overflow_mask[lane] = global_column[32];
				ap_uint<64> global_item = local_item;
				global_item.range(31, 0) = global_column.range(31, 0);
				packed.range(lane * 64 + 63, lane * 64) = global_item;
			}
		}
		if (overflow_mask != 0 && row_error == 0) row_error = 3;
		if (count != 0 || eor) {
			const id_t completed_nnz = active_nnz + count;
			output.write(unified_full_make_bundle(packed,
				unified_full_make_record(row, eor ? completed_nnz : (id_t)0,
					count, eor, context,
					UNIFIED_FULL_OUTPUT_MERGE_ROW, 15, row_error, 2,
					generation)));
			active_nnz = completed_nnz;
		}
		// Unlike a DENSE extractor boundary, keyed carry may flush the final
		// numerical key in the same token that carries EOR.  The record above
		// therefore carries valid data and EOR together, preserving one output
		// write per iteration and allowing the loop to remain II=1.
		if (eor) {
			CompletionToken completion = 0;
			completion.range(63, 0) = unified_full_make_completion(row,
				context, generation, false, 15);
			completions.write(completion);
			active = false;
		}
	}
	if (active) {
		output.write(unified_full_make_bundle(0,
			unified_full_make_record(active_row, active_nnz, 0, true,
				active_context, UNIFIED_FULL_OUTPUT_MERGE_ROW, 15, 3, 2,
				active_generation)));
	}
	BaseToken base_end = bases.read();
	while (!base_end[64]) base_end = bases.read();
	output.write(end_bundle());
	completions.write(end_completion());
}

void unified_full_bundle_mux2(
		tapa::istream<PacketBundle>& input0,
		tapa::istream<PacketBundle>& input1,
		tapa::ostream<PacketBundle>& output) {
	PacketBundle held[2] = {};
	bool valid[2] = {};
	bool ended[2] = {};
#pragma HLS ARRAY_PARTITION variable=held complete
#pragma HLS ARRAY_PARTITION variable=valid complete
#pragma HLS ARRAY_PARTITION variable=ended complete
	ap_uint<1> turn = 0;
	while (!(ended[0] && ended[1] && !valid[0] && !valid[1])) {
#pragma HLS PIPELINE II=1
		PacketBundle next;
		if (!ended[0] && !valid[0] && input0.try_read(next)) {
			if (next[640]) ended[0] = true;
			else { held[0] = next; valid[0] = true; }
		}
		if (!ended[1] && !valid[1] && input1.try_read(next)) {
			if (next[640]) ended[1] = true;
			else { held[1] = next; valid[1] = true; }
		}
		const ap_uint<1> selected = valid[turn] ? turn : (ap_uint<1>)(!turn);
		if (valid[selected] && output.try_write(held[selected])) {
			valid[selected] = false;
			turn = !selected;
		}
	}
	output.write(end_bundle());
}

void unified_full_completion_mux2(
		tapa::istream<CompletionToken>& input0,
		tapa::istream<CompletionToken>& input1,
		tapa::ostream<CompletionToken>& output) {
	CompletionToken held[2] = {};
	bool valid[2] = {};
	bool ended[2] = {};
#pragma HLS ARRAY_PARTITION variable=held complete
#pragma HLS ARRAY_PARTITION variable=valid complete
#pragma HLS ARRAY_PARTITION variable=ended complete
	ap_uint<1> turn = 0;
	while (!(ended[0] && ended[1] && !valid[0] && !valid[1])) {
#pragma HLS PIPELINE II=1
		CompletionToken next;
		if (!ended[0] && !valid[0] && input0.try_read(next)) {
			if (next[64]) ended[0] = true;
			else { held[0] = next; valid[0] = true; }
		}
		if (!ended[1] && !valid[1] && input1.try_read(next)) {
			if (next[64]) ended[1] = true;
			else { held[1] = next; valid[1] = true; }
		}
		const ap_uint<1> selected = valid[turn] ? turn : (ap_uint<1>)(!turn);
		if (valid[selected] && output.try_write(held[selected])) {
			valid[selected] = false;
			turn = !selected;
		}
	}
	output.write(end_completion());
}

// A 64K logical DENSE row is emitted by both 32K physical packetizers.  Hold
// the first half's post-output completion and release the scheduler only after
// the matching second half has also reached HBM.  Ordinary MERGE/DENSE rows
// remain one-completion pass-throughs.
void unified_full_wide_completion_coalesce(
		tapa::istream<CompletionToken>& input,
		tapa::ostream<CompletionToken>& output) {
	ap_uint<64> pending[8] = {};
	bool valid[8] = {};
#pragma HLS ARRAY_PARTITION variable=pending complete
#pragma HLS ARRAY_PARTITION variable=valid complete
	bool done = false;
	bool terminal_seen = false;
	ap_uint<4> drain_context = 0;
	while (!done) {
#pragma HLS PIPELINE II=1
		if (terminal_seen) {
			if (drain_context < 8) {
				if (valid[drain_context]) {
					CompletionToken orphan = 0;
					orphan.range(63, 0) = pending[drain_context];
					output.write(orphan);
					valid[drain_context] = false;
				}
				++drain_context;
			} else {
				output.write(end_completion());
				done = true;
			}
			continue;
		}
		CompletionToken token;
		if (!input.try_read(token)) continue;
		if (token[64]) {
			// Drain malformed one-sided rows one per cycle before terminal so
			// the allocator cannot deadlock during teardown.
			terminal_seen = true;
			continue;
		}
		ap_uint<64> event = token.range(63, 0);
		if (!event[48]) {
			output.write(token);
			continue;
		}
		const ap_uint<3> context = event.range(2, 0);
		if (!valid[context]) {
			pending[context] = event;
			valid[context] = true;
			continue;
		}
		const bool identity_error
			= pending[context].range(42, 3) != event.range(42, 3)
			|| !pending[context][47] || !event[47];
		ap_uint<64> joined = pending[context];
		joined.range(46, 43) = 0;
		// Bit 49 is internal protocol status and is ignored by the allocator;
		// it remains visible in the diagnostic completion array.
		joined[49] = identity_error;
		CompletionToken completion = 0;
		completion.range(63, 0) = joined;
		output.write(completion);
		valid[context] = false;
	}
}

void unified_full_scheduler_statistics_write(
		tapa::istream<ap_uint<512> >& input,
		tapa::mmap<id_t> statistics) {
	const ap_uint<512> packed = input.read();
	for (int word = 0; word < 16; ++word) {
#pragma HLS PIPELINE II=1
		statistics[word] = packed.range(word * 32 + 31, word * 32);
	}
}

void unified_full_dense_statistics_write(
		tapa::istream<ap_uint<192> >& input0,
		tapa::istream<ap_uint<192> >& input1,
		tapa::mmap<id_t> statistics) {
	const ap_uint<192> packed0 = input0.read();
	const ap_uint<192> packed1 = input1.read();
	for (int word = 0; word < 6; ++word) {
#pragma HLS PIPELINE II=1
		statistics[word] = packed0.range(word * 32 + 31, word * 32);
		statistics[word + 6] = packed1.range(word * 32 + 31, word * 32);
	}
}

void unified_full_heavy_statistics_write(
		tapa::istream<ap_uint<256> >& input,
		tapa::mmap<id_t> statistics) {
	const ap_uint<256> packed = input.read();
	for (int word = 0; word < 8; ++word) {
#pragma HLS PIPELINE II=1
		statistics[word] = packed.range(word * 32 + 31, word * 32);
	}
}

void adaptive_hbm_unified_adaptive_full_output_tapa(
		tapa::mmap<const ap_uint<128> > commands, id_t command_count,
		tapa::mmap<const id_t> global_bases,
		tapa::mmap<const ap_uint<64> > source0,
		tapa::mmap<const ap_uint<64> > source1,
		tapa::mmap<const ap_uint<64> > source2,
		tapa::mmap<const ap_uint<64> > source3,
		tapa::mmap<const ap_uint<64> > source4,
		tapa::mmap<const ap_uint<64> > source5,
		tapa::mmap<const ap_uint<64> > source6,
		tapa::mmap<const ap_uint<64> > source7,
		id_t dense_local_column_count, id_t heavy_merge_product_threshold,
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
		tapa::mmap<id_t> output_completion_stats,
		tapa::mmap<id_t> scheduler_stats,
		tapa::mmap<id_t> dense_stats,
		tapa::mmap<id_t> heavy_merge_stats) {
	tapa::stream<MergeAllocator8CommandToken, 16> command_stream("command_stream");
	tapa::stream<MergeAllocator8DispatchToken, 16> dispatch_stream("dispatch_stream");
	tapa::stream<MergeAllocator8EorToken, 16> scheduler_eors("scheduler_eors");
	tapa::stream<MergeAllocator8AckToken, 16> scheduler_acks("scheduler_acks");
	tapa::stream<CompletionToken, 16> memory_completions("memory_completions");
	tapa::stream<ap_uint<512>, 2> scheduler_statistics("scheduler_statistics");

	tapa::streams<UnifiedSourceDispatchToken, 8, 16> source_dispatches("source_dispatches");
	tapa::streams<UnifiedRoutedLeafToken, 8, 16> routed_sources("routed_sources");
	tapa::streams<UnifiedFullDenseSourceToken, 8, 16> dense_sources("dense_sources");
	tapa::streams<UnifiedFullDenseSourceToken, 8, 16> heavy_sources("heavy_sources");
	tapa::streams<UnifiedHeavyVector4Token, 8, 8> heavy_vector_leaf("heavy_vector_leaf");
	tapa::streams<UnifiedHeavyVector4Token, 4, 8> heavy_vector_stage0("heavy_vector_stage0");
	tapa::streams<UnifiedHeavyVector4Token, 2, 8> heavy_vector_stage1("heavy_vector_stage1");
	tapa::stream<UnifiedHeavyVector4Token, 8> heavy_vector_root("heavy_vector_root");
	tapa::streams<UnifiedRoutedLeafToken, 8, 8> crossbar_stage0("crossbar_stage0");
	tapa::streams<UnifiedRoutedLeafToken, 8, 8> crossbar_stage1("crossbar_stage1");
	tapa::streams<UnifiedRoutedLeafToken, 8, 8> crossbar_stage2("crossbar_stage2");
	tapa::streams<MergeForwardTree8Token, 8, 16> memory_to_leaf("memory_to_leaf");

	tapa::streams<MergeForwardTree8Token, 8, 8> leaf_up("leaf_up");
	tapa::streams<MergeForwardTree8Token, 8, 8> leaf_exit("leaf_exit");
	tapa::streams<MergeForwardTree8Token, 4, 8> level1_up("level1_up");
	tapa::streams<MergeForwardTree8Token, 4, 8> level1_exit("level1_exit");
	tapa::streams<MergeForwardTree8Token, 2, 8> level2_up("level2_up");
	tapa::streams<MergeForwardTree8Token, 2, 8> level2_exit("level2_exit");
	tapa::stream<MergeForwardTree8Token, 8> root_exit("root_exit");
	tapa::stream<MergeForwardTree8Token, 8> root_up("root_up");
	tapa::streams<BaseToken, 15, 4> merge_bases("merge_bases");
	tapa::streams<BaseToken, 2, 4> dense_bases("dense_bases");
	tapa::stream<BaseToken, 4> heavy_base("heavy_base");

	tapa::stream<Segmented8SharedToken, 16> dense_raw("dense_raw");
	tapa::stream<UnifiedFullDenseTag, 16> dense_tags("dense_tags");
	tapa::stream<Segmented8SharedToken, 16> heavy_raw("heavy_raw");
	tapa::stream<UnifiedFullDenseTag, 16> heavy_tags("heavy_tags");
	tapa::stream<Segmented8SharedToken, 16> shared_raw("shared_raw");
	tapa::stream<UnifiedFullDenseTag, 16> shared_tags("shared_tags");
	tapa::stream<Segmented8SharedToken, 16> dense_reduced("dense_reduced");
	tapa::streams<DenseDualContextToken, 2, 16> dense_prebank("dense_prebank");
	tapa::streams<JointDenseLaneToken, 8, 16> dense_lane0("dense_lane0");
	tapa::streams<JointDenseLaneToken, 8, 16> dense_lane1("dense_lane1");
	tapa::streams<DenseDualContextToken, 2, 16> dense_accumulate("dense_accumulate");
	tapa::streams<DenseDualContextToken, 2, 16> dense_result("dense_result");
	tapa::streams<DenseDualContextToken, 2, 8> dense_compact2("dense_compact2");
	tapa::streams<DenseDualContextToken, 2, 8> dense_compact4("dense_compact4");
	tapa::streams<DenseDualContextToken, 2, 8> dense_compact8("dense_compact8");
	tapa::streams<ap_uint<192>, 2, 2> dense_statistics("dense_statistics");
	tapa::stream<MergeKeyedCarry8InputToken, 16> heavy_segmented("heavy_segmented");
	tapa::stream<CarryWorkToken, 8> heavy_carry_work("heavy_carry_work");
	tapa::stream<CarryResultToken, 8> heavy_carry_result("heavy_carry_result");
	tapa::stream<DenseDualContextToken, 8> heavy_result("heavy_result");
	tapa::stream<DenseDualContextToken, 8> heavy_compact2("heavy_compact2");
	tapa::stream<DenseDualContextToken, 8> heavy_compact4("heavy_compact4");
	tapa::stream<DenseDualContextToken, 8> heavy_compact8("heavy_compact8");
	tapa::stream<ap_uint<256>, 2> heavy_statistics("heavy_statistics");

	tapa::streams<PacketBundle, 15, 4> exit_packets("exit_packets");
	tapa::streams<CompletionToken, 15, 4> exit_completions("exit_completions");
	tapa::streams<PacketBundle, 2, 4> dense_packets("dense_packets");
	tapa::streams<CompletionToken, 2, 4> dense_completions("dense_completions");
	tapa::stream<PacketBundle, 4> heavy_packet("heavy_packet");
	tapa::stream<CompletionToken, 4> heavy_completion("heavy_completion");
	tapa::stream<PacketBundle, 2> empty_packet("empty_packet");
	tapa::stream<CompletionToken, 2> empty_completion("empty_completion");
	tapa::streams<PacketBundle, 4, 8> merge_port_packets("merge_port_packets");
	tapa::streams<CompletionToken, 4, 8> merge_port_completions("merge_port_completions");
	tapa::streams<PacketBundle, 3, 8> adaptive_port_packets("adaptive_port_packets");
	tapa::streams<CompletionToken, 3, 8> adaptive_port_completions("adaptive_port_completions");
	tapa::stream<CompletionToken, 8> raw_merged_completions("raw_merged_completions");
	tapa::stream<CompletionToken, 8> merged_completions("merged_completions");
	tapa::streams<RecordToken, 4, 8> records("records");
	tapa::stream<RecordBatch, 4> record_batches("record_batches");
	tapa::stream<WordToken, 4> meta_words("meta_words");
	tapa::stream<StatsToken, 2> meta_summary("meta_summary");

	tapa::task()
		.invoke(merge_allocator8_command_read, commands, command_count,
			command_stream)
		.invoke(unified_full_scheduler, command_stream, scheduler_eors,
			heavy_merge_product_threshold,
			dispatch_stream, scheduler_acks, scheduler_statistics)
		.invoke(unified_allocator_ack_drain, scheduler_acks)
		.invoke(unified_full_dispatch_broadcast, dispatch_stream, global_bases,
			source_dispatches[0], source_dispatches[1], source_dispatches[2],
			source_dispatches[3], source_dispatches[4], source_dispatches[5],
			source_dispatches[6], source_dispatches[7],
			merge_bases[0], merge_bases[1], merge_bases[2], merge_bases[3],
			merge_bases[4], merge_bases[5], merge_bases[6], merge_bases[7],
			merge_bases[8], merge_bases[9], merge_bases[10], merge_bases[11],
			merge_bases[12], merge_bases[13], merge_bases[14],
			dense_bases[0], dense_bases[1], heavy_base)
#define UNIFIED_FULL_SOURCE_READER(S, MEMORY) \
		.invoke(unified_full_source_read, MEMORY, S, source_dispatches[S], \
			routed_sources[S], dense_sources[S], heavy_sources[S])
		UNIFIED_FULL_SOURCE_READER(0, source0)
		UNIFIED_FULL_SOURCE_READER(1, source1)
		UNIFIED_FULL_SOURCE_READER(2, source2)
		UNIFIED_FULL_SOURCE_READER(3, source3)
		UNIFIED_FULL_SOURCE_READER(4, source4)
		UNIFIED_FULL_SOURCE_READER(5, source5)
		UNIFIED_FULL_SOURCE_READER(6, source6)
		UNIFIED_FULL_SOURCE_READER(7, source7)
#undef UNIFIED_FULL_SOURCE_READER
#define UNIFIED_FULL_HEAVY_PACK4(S) \
		.invoke(unified_full_heavy_source_pack4, heavy_sources[S], \
			heavy_vector_leaf[S])
		UNIFIED_FULL_HEAVY_PACK4(0)
		UNIFIED_FULL_HEAVY_PACK4(1)
		UNIFIED_FULL_HEAVY_PACK4(2)
		UNIFIED_FULL_HEAVY_PACK4(3)
		UNIFIED_FULL_HEAVY_PACK4(4)
		UNIFIED_FULL_HEAVY_PACK4(5)
		UNIFIED_FULL_HEAVY_PACK4(6)
		UNIFIED_FULL_HEAVY_PACK4(7)
#undef UNIFIED_FULL_HEAVY_PACK4
		.invoke(unified_full_heavy_merge4, heavy_vector_leaf[0],
			heavy_vector_leaf[1], heavy_vector_stage0[0])
		.invoke(unified_full_heavy_merge4, heavy_vector_leaf[2],
			heavy_vector_leaf[3], heavy_vector_stage0[1])
		.invoke(unified_full_heavy_merge4, heavy_vector_leaf[4],
			heavy_vector_leaf[5], heavy_vector_stage0[2])
		.invoke(unified_full_heavy_merge4, heavy_vector_leaf[6],
			heavy_vector_leaf[7], heavy_vector_stage0[3])
		.invoke(unified_full_heavy_merge4, heavy_vector_stage0[0],
			heavy_vector_stage0[1], heavy_vector_stage1[0])
		.invoke(unified_full_heavy_merge4, heavy_vector_stage0[2],
			heavy_vector_stage0[3], heavy_vector_stage1[1])
		.invoke(unified_full_heavy_merge4, heavy_vector_stage1[0],
			heavy_vector_stage1[1], heavy_vector_root)
		.invoke(unified_allocator_crossbar_stage8,
			routed_sources[0], routed_sources[4], routed_sources[1],
			routed_sources[5], routed_sources[2], routed_sources[6],
			routed_sources[3], routed_sources[7], 2,
			crossbar_stage0[0], crossbar_stage0[4], crossbar_stage0[1],
			crossbar_stage0[5], crossbar_stage0[2], crossbar_stage0[6],
			crossbar_stage0[3], crossbar_stage0[7])
		.invoke(unified_allocator_crossbar_stage8,
			crossbar_stage0[0], crossbar_stage0[2], crossbar_stage0[1],
			crossbar_stage0[3], crossbar_stage0[4], crossbar_stage0[6],
			crossbar_stage0[5], crossbar_stage0[7], 1,
			crossbar_stage1[0], crossbar_stage1[2], crossbar_stage1[1],
			crossbar_stage1[3], crossbar_stage1[4], crossbar_stage1[6],
			crossbar_stage1[5], crossbar_stage1[7])
		.invoke(unified_allocator_crossbar_stage8,
			crossbar_stage1[0], crossbar_stage1[1], crossbar_stage1[2],
			crossbar_stage1[3], crossbar_stage1[4], crossbar_stage1[5],
			crossbar_stage1[6], crossbar_stage1[7], 0,
			crossbar_stage2[0], crossbar_stage2[1], crossbar_stage2[2],
			crossbar_stage2[3], crossbar_stage2[4], crossbar_stage2[5],
			crossbar_stage2[6], crossbar_stage2[7])
		.invoke(unified_allocator_crossbar_finish8,
			crossbar_stage2[0], crossbar_stage2[1], crossbar_stage2[2],
			crossbar_stage2[3], crossbar_stage2[4], crossbar_stage2[5],
			crossbar_stage2[6], crossbar_stage2[7], memory_to_leaf[0],
			memory_to_leaf[1], memory_to_leaf[2], memory_to_leaf[3],
			memory_to_leaf[4], memory_to_leaf[5], memory_to_leaf[6],
			memory_to_leaf[7])
		.invoke<tapa::join, 8>(merge_forward_tree8_leaf,
			memory_to_leaf, leaf_exit, leaf_up)
		.invoke(merge_forward_tree8_node, leaf_up[0], leaf_up[1], 1,
			level1_exit[0], level1_up[0])
		.invoke(merge_forward_tree8_node, leaf_up[2], leaf_up[3], 1,
			level1_exit[1], level1_up[1])
		.invoke(merge_forward_tree8_node, leaf_up[4], leaf_up[5], 1,
			level1_exit[2], level1_up[2])
		.invoke(merge_forward_tree8_node, leaf_up[6], leaf_up[7], 1,
			level1_exit[3], level1_up[3])
		.invoke(merge_forward_tree8_node, level1_up[0], level1_up[1], 2,
			level2_exit[0], level2_up[0])
		.invoke(merge_forward_tree8_node, level1_up[2], level1_up[3], 2,
			level2_exit[1], level2_up[1])
		.invoke(merge_forward_tree8_node, level2_up[0], level2_up[1], 3,
			root_exit, root_up)
		.invoke(merge_forward_tree8_drain, root_up)
		.invoke(unified_full_dense_pack8,
			dense_sources[0], dense_sources[1], dense_sources[2], dense_sources[3],
			dense_sources[4], dense_sources[5], dense_sources[6], dense_sources[7],
			dense_raw, dense_tags)
		.invoke(unified_full_heavy_vector_to_segmented,
			heavy_vector_root, heavy_raw, heavy_tags)
		.invoke(unified_full_segmented_pair_mux,
			dense_raw, dense_tags, heavy_raw, heavy_tags,
			shared_raw, shared_tags)
		.invoke(segmented8_shared_reduce, shared_raw, dense_reduced)
		.invoke(unified_full_dense_restore, dense_reduced, shared_tags,
			dense_prebank[0], dense_prebank[1], heavy_segmented)
		.invoke(merge_keyed_carry8_state, heavy_segmented, heavy_carry_work)
		.invoke(merge_keyed_carry8_flush_reduce,
			heavy_carry_work, heavy_carry_result)
		.invoke(unified_full_heavy_restore, heavy_carry_result,
			heavy_result, heavy_statistics)
		.invoke(unified_dense_lane_demux, dense_prebank[0],
			dense_lane0[0], dense_lane0[1], dense_lane0[2], dense_lane0[3],
			dense_lane0[4], dense_lane0[5], dense_lane0[6], dense_lane0[7])
		.invoke(unified_dense_lane_demux, dense_prebank[1],
			dense_lane1[0], dense_lane1[1], dense_lane1[2], dense_lane1[3],
			dense_lane1[4], dense_lane1[5], dense_lane1[6], dense_lane1[7])
		.invoke(unified_dense_bank_arbiter0,
			dense_lane0[0], dense_lane0[1], dense_lane0[2], dense_lane0[3],
			dense_lane0[4], dense_lane0[5], dense_lane0[6], dense_lane0[7],
			dense_accumulate[0])
		.invoke(unified_dense_bank_arbiter1,
			dense_lane1[0], dense_lane1[1], dense_lane1[2], dense_lane1[3],
			dense_lane1[4], dense_lane1[5], dense_lane1[6], dense_lane1[7],
			dense_accumulate[1])
		.invoke(dense_dual_context_engine0, dense_accumulate[0],
			dense_local_column_count, dense_result[0], dense_statistics[0])
		.invoke(dense_dual_context_engine1, dense_accumulate[1],
			dense_local_column_count, dense_result[1], dense_statistics[1])
		.invoke(unified_full_dense_compact2, dense_result[0], dense_compact2[0])
		.invoke(unified_full_dense_compact2, dense_result[1], dense_compact2[1])
		.invoke(unified_full_dense_compact4, dense_compact2[0], dense_compact4[0])
		.invoke(unified_full_dense_compact4, dense_compact2[1], dense_compact4[1])
		.invoke(unified_full_dense_compact8, dense_compact4[0], dense_compact8[0])
		.invoke(unified_full_dense_compact8, dense_compact4[1], dense_compact8[1])
		.invoke(unified_full_dense_compact2, heavy_result, heavy_compact2)
		.invoke(unified_full_dense_compact4, heavy_compact2, heavy_compact4)
		.invoke(unified_full_dense_compact8, heavy_compact4, heavy_compact8)
#define UNIFIED_FULL_PACKETIZE(STREAM, EXIT, PORT) \
		.invoke(merge15_exit_packetize, STREAM, merge_bases[EXIT], EXIT, PORT, \
			exit_packets[EXIT], exit_completions[EXIT])
		UNIFIED_FULL_PACKETIZE(leaf_exit[0], 0, 0)
		UNIFIED_FULL_PACKETIZE(leaf_exit[1], 1, 1)
		UNIFIED_FULL_PACKETIZE(leaf_exit[2], 2, 2)
		UNIFIED_FULL_PACKETIZE(leaf_exit[3], 3, 3)
		UNIFIED_FULL_PACKETIZE(leaf_exit[4], 4, 0)
		UNIFIED_FULL_PACKETIZE(leaf_exit[5], 5, 1)
		UNIFIED_FULL_PACKETIZE(leaf_exit[6], 6, 2)
		UNIFIED_FULL_PACKETIZE(leaf_exit[7], 7, 3)
		UNIFIED_FULL_PACKETIZE(level1_exit[0], 8, 0)
		UNIFIED_FULL_PACKETIZE(level1_exit[1], 9, 1)
		UNIFIED_FULL_PACKETIZE(level1_exit[2], 10, 2)
		UNIFIED_FULL_PACKETIZE(level1_exit[3], 11, 3)
		UNIFIED_FULL_PACKETIZE(level2_exit[0], 12, 0)
		UNIFIED_FULL_PACKETIZE(level2_exit[1], 13, 1)
		UNIFIED_FULL_PACKETIZE(root_exit, 14, 2)
#undef UNIFIED_FULL_PACKETIZE
		.invoke(unified_full_dense_packetize, dense_compact8[0], dense_bases[0],
			0, 0, dense_packets[0], dense_completions[0])
		.invoke(unified_full_dense_packetize, dense_compact8[1], dense_bases[1],
			1, 1, dense_packets[1], dense_completions[1])
		.invoke(unified_full_heavy_packetize, heavy_compact8, heavy_base,
			heavy_packet, heavy_completion)
		.invoke(unified_allocator_empty_output, empty_packet, empty_completion)
		.invoke(merge15_bundle_mux4, exit_packets[0], exit_packets[4],
			exit_packets[8], exit_packets[12], merge_port_packets[0])
		.invoke(merge15_bundle_mux4, exit_packets[1], exit_packets[5],
			exit_packets[9], exit_packets[13], merge_port_packets[1])
		.invoke(merge15_bundle_mux4, exit_packets[2], exit_packets[6],
			exit_packets[10], exit_packets[14], merge_port_packets[2])
		.invoke(merge15_bundle_mux4, exit_packets[3], exit_packets[7],
			exit_packets[11], empty_packet, merge_port_packets[3])
		.invoke(merge15_completion_mux4, exit_completions[0],
			exit_completions[4], exit_completions[8], exit_completions[12],
			merge_port_completions[0])
		.invoke(merge15_completion_mux4, exit_completions[1],
			exit_completions[5], exit_completions[9], exit_completions[13],
			merge_port_completions[1])
		.invoke(merge15_completion_mux4, exit_completions[2],
			exit_completions[6], exit_completions[10], exit_completions[14],
			merge_port_completions[2])
		.invoke(merge15_completion_mux4, exit_completions[3],
			exit_completions[7], exit_completions[11], empty_completion,
			merge_port_completions[3])
		.invoke(unified_full_bundle_mux2, merge_port_packets[0], dense_packets[0],
			adaptive_port_packets[0])
		.invoke(unified_full_bundle_mux2, merge_port_packets[1], dense_packets[1],
			adaptive_port_packets[1])
		.invoke(unified_full_bundle_mux2, merge_port_packets[2], heavy_packet,
			adaptive_port_packets[2])
		.invoke(unified_full_completion_mux2, merge_port_completions[0],
			dense_completions[0], adaptive_port_completions[0])
		.invoke(unified_full_completion_mux2, merge_port_completions[1],
			dense_completions[1], adaptive_port_completions[1])
		.invoke(unified_full_completion_mux2, merge_port_completions[2],
			heavy_completion, adaptive_port_completions[2])
		.invoke(merge15_completion_mux4, adaptive_port_completions[0],
			adaptive_port_completions[1], adaptive_port_completions[2],
			merge_port_completions[3], raw_merged_completions)
		.invoke(unified_full_wide_completion_coalesce,
			raw_merged_completions, merged_completions)
		.invoke(unified_allocator_completion_broadcast, merged_completions,
			scheduler_eors, memory_completions)
		.invoke(merge15_port_write, adaptive_port_packets[0],
			output_data_word_capacity_per_port, output_item0,
			output_port_stats0, records[0])
		.invoke(merge15_port_write, adaptive_port_packets[1],
			output_data_word_capacity_per_port, output_item1,
			output_port_stats1, records[1])
		.invoke(merge15_port_write, adaptive_port_packets[2],
			output_data_word_capacity_per_port, output_item2,
			output_port_stats2, records[2])
		.invoke(merge15_port_write, merge_port_packets[3],
			output_data_word_capacity_per_port, output_item3,
			output_port_stats3, records[3])
		.invoke(merge15_meta_collect, records[0], records[1], records[2],
			records[3], record_batches, meta_summary)
		.invoke(merge15_meta_pack, record_batches, meta_words)
		.invoke(merge15_meta_write, meta_words, meta_summary,
			output_meta_word_capacity, output_meta, output_meta_stats)
		.invoke(merge15_completion_write, memory_completions,
			output_completion_capacity, output_completion,
			output_completion_stats)
		.invoke(unified_full_scheduler_statistics_write, scheduler_statistics,
			scheduler_stats)
		.invoke(unified_full_dense_statistics_write, dense_statistics[0],
			dense_statistics[1], dense_stats)
		.invoke(unified_full_heavy_statistics_write, heavy_statistics,
			heavy_merge_stats);
}
