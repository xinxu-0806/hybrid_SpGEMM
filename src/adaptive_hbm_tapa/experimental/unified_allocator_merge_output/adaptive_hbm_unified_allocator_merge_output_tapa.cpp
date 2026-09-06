#include "adaptive_hbm_unified_allocator_merge_output_tapa.h"

// Compose the accepted allocator, flexible tree and four-port output blocks.
// Rename only file-local helpers that would otherwise collide in this single
// HLS translation unit; the accepted task bodies remain unchanged.
#include "../merge_allocator8/adaptive_hbm_merge_allocator8_tapa.h"
#include "../merge_forward_tree8/adaptive_hbm_merge_forward_tree8_tapa.h"
#include "../merge15_packetmeta4/adaptive_hbm_merge15_packetmeta4_tapa.h"

#define popcount8 allocator_merge_output_allocator_popcount8
#include "../merge_allocator8/adaptive_hbm_merge_allocator8_tapa.cpp"
#undef popcount8
#include "../merge_forward_tree8/adaptive_hbm_merge_forward_tree8_tapa.cpp"
#include "../merge15_packetmeta4/adaptive_hbm_merge15_packetmeta4_tapa.cpp"

namespace {

// dispatch[127:0], allocation sequence[143:128], terminal[144].
using UnifiedSourceDispatchToken = ap_uint<145>;
// tree token[128:0], destination leaf[131:129], source terminal[132].
using UnifiedRoutedLeafToken = ap_uint<133>;

static MergeAllocator8DispatchToken allocator_dispatch_terminal() {
#pragma HLS INLINE
	MergeAllocator8DispatchToken token = 0;
	token[128] = 1;
	return token;
}

static ap_uint<128> make_tree_item(ap_uint<64> numerical, id_t row,
		ap_uint<3> context, ap_uint<8> generation, ap_uint<2> level,
		bool eor, ap_uint<16> sequence) {
#pragma HLS INLINE
	ap_uint<128> item = 0;
	item.range(15, 0) = numerical.range(15, 0);
	item.range(47, 16) = numerical.range(63, 32);
	item.range(79, 48) = row;
	item.range(82, 80) = context;
	item.range(90, 83) = generation;
	item.range(92, 91) = level;
	item[93] = eor;
	item.range(109, 94) = sequence;
	return item;
}

static void write_leaf(MergeForwardTree8Token token, ap_uint<3> leaf,
		tapa::ostream<MergeForwardTree8Token>& leaf0,
		tapa::ostream<MergeForwardTree8Token>& leaf1,
		tapa::ostream<MergeForwardTree8Token>& leaf2,
		tapa::ostream<MergeForwardTree8Token>& leaf3,
		tapa::ostream<MergeForwardTree8Token>& leaf4,
		tapa::ostream<MergeForwardTree8Token>& leaf5,
		tapa::ostream<MergeForwardTree8Token>& leaf6,
		tapa::ostream<MergeForwardTree8Token>& leaf7) {
#pragma HLS INLINE
	switch ((unsigned)leaf) {
	case 0: leaf0.write(token); break;
	case 1: leaf1.write(token); break;
	case 2: leaf2.write(token); break;
	case 3: leaf3.write(token); break;
	case 4: leaf4.write(token); break;
	case 5: leaf5.write(token); break;
	case 6: leaf6.write(token); break;
	default: leaf7.write(token); break;
	}
}

static void write_base(BaseToken token, ap_uint<4> exit,
		tapa::ostream<BaseToken>& base0, tapa::ostream<BaseToken>& base1,
		tapa::ostream<BaseToken>& base2, tapa::ostream<BaseToken>& base3,
		tapa::ostream<BaseToken>& base4, tapa::ostream<BaseToken>& base5,
		tapa::ostream<BaseToken>& base6, tapa::ostream<BaseToken>& base7,
		tapa::ostream<BaseToken>& base8, tapa::ostream<BaseToken>& base9,
		tapa::ostream<BaseToken>& base10, tapa::ostream<BaseToken>& base11,
		tapa::ostream<BaseToken>& base12, tapa::ostream<BaseToken>& base13,
		tapa::ostream<BaseToken>& base14) {
#pragma HLS INLINE
	switch ((unsigned)exit) {
	case 0: base0.write(token); break;
	case 1: base1.write(token); break;
	case 2: base2.write(token); break;
	case 3: base3.write(token); break;
	case 4: base4.write(token); break;
	case 5: base5.write(token); break;
	case 6: base6.write(token); break;
	case 7: base7.write(token); break;
	case 8: base8.write(token); break;
	case 9: base9.write(token); break;
	case 10: base10.write(token); break;
	case 11: base11.write(token); break;
	case 12: base12.write(token); break;
	case 13: base13.write(token); break;
	default: base14.write(token); break;
	}
}

}  // namespace

// Live variant of the accepted allocator.  Unlike the standalone gate, EORs
// arrive from the actual packetizers.  Dispatch termination is sent as soon as
// every command has launched; the controller then remains alive until the
// packetizers return their terminal and every context has been released.
void unified_allocator_live_control(
		tapa::istream<MergeAllocator8CommandToken>& command_input,
		tapa::istream<MergeAllocator8EorToken>& eor_input,
		tapa::ostream<MergeAllocator8DispatchToken>& dispatch_output,
		tapa::ostream<MergeAllocator8AckToken>& ack_output,
		tapa::ostream<ap_uint<384> >& statistics_output) {
	ap_uint<8> busy = 0;
	ap_uint<8> free_leaf = 0xff;
	id_t active_row[8];
	ap_uint<8> active_generation[8];
	ap_uint<8> active_leaf_mask[8];
#pragma HLS ARRAY_PARTITION variable=active_row complete
#pragma HLS ARRAY_PARTITION variable=active_generation complete
#pragma HLS ARRAY_PARTITION variable=active_leaf_mask complete
	for (int slot = 0; slot < 8; ++slot) {
#pragma HLS UNROLL
		active_row[slot] = 0;
		active_generation[slot] = 0;
		active_leaf_mask[slot] = 0;
	}
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
	id_t peak_contexts = 0;
	id_t peak_leaves = 0;
	id_t exit_count[4] = {};
#pragma HLS ARRAY_PARTITION variable=exit_count complete
	id_t stale_count = 0;
	id_t blocked_cycles = 0;

	while (!(dispatch_done && eor_done && !eor_valid && busy == 0)) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=1048576 avg=4096
		const ap_uint<8> allocation_busy = busy;
		const ap_uint<8> allocation_free_leaf = free_leaf;
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

		if (eor_valid && accepted_count >= (id_t)held_eor.range(58, 43)) {
			const ap_uint<3> context = held_eor.range(2, 0);
			const ap_uint<8> generation = held_eor.range(10, 3);
			const id_t row = held_eor.range(42, 11);
			const bool accepted = busy[context]
				&& active_generation[context] == generation
				&& active_row[context] == row;
			ap_uint<8> released = 0;
			ap_uint<3> status = 0;
			if (accepted) {
				released = active_leaf_mask[context];
				free_leaf |= released;
				busy[context] = 0;
				active_leaf_mask[context] = 0;
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
			const ap_uint<4> width = allocation_width(source_mask);
			ap_uint<3> context = 0;
			ap_uint<3> leaf_base = 0;
			ap_uint<8> leaf_mask = 0;
			const bool have_context = find_context(allocation_busy, context);
			const bool have_leaves = find_leaf_block(allocation_free_leaf,
				width, leaf_base, leaf_mask);
			if (source_mask != 0 && have_context && have_leaves) {
				const ap_uint<8> generation = generation_counter + 1;
				generation_counter = generation;
				busy[context] = 1;
				free_leaf &= ~leaf_mask;
				active_row[context] = command.range(31, 0);
				active_generation[context] = generation;
				active_leaf_mask[context] = leaf_mask;
				MergeAllocator8DispatchToken dispatch = 0;
				dispatch.range(127, 0) = make_dispatch(command, leaf_mask,
					leaf_base, context, generation, width);
				dispatch_output.write(dispatch);
				++accepted_count;
				++exit_count[(unsigned)allocation_level(width)];
				command_valid = false;
			} else {
				++blocked_cycles;
			}
		} else if (command_done && !dispatch_done) {
			// This branch is mutually exclusive with numerical dispatch above.
			// Keeping a single possible write to dispatch_output per iteration
			// lets HLS retain the accepted allocator's II=1 control loop.
			dispatch_output.write(allocator_dispatch_terminal());
			dispatch_done = true;
		}
		const id_t live_contexts
			= allocator_merge_output_allocator_popcount8(busy);
		const id_t live_leaves = 8
			- allocator_merge_output_allocator_popcount8(free_leaf);
		if (live_contexts > peak_contexts) peak_contexts = live_contexts;
		if (live_leaves > peak_leaves) peak_leaves = live_leaves;
	}

	MergeAllocator8AckToken ack_end = 0;
	ack_end[64] = 1;
	ack_output.write(ack_end);
	ap_uint<384> statistics = 0;
	statistics.range(31, 0) = accepted_count;
	statistics.range(63, 32) = completed_count;
	statistics.range(95, 64) = peak_contexts;
	statistics.range(127, 96) = peak_leaves;
	statistics.range(159, 128) = exit_count[0];
	statistics.range(191, 160) = exit_count[1];
	statistics.range(223, 192) = exit_count[2];
	statistics.range(255, 224) = exit_count[3];
	statistics.range(287, 256) = stale_count;
	statistics.range(319, 288) = blocked_cycles;
	statistics.range(351, 320) = free_leaf;
	statistics.range(383, 352) = busy;
	statistics_output.write(statistics);
}

void unified_allocator_dispatch_feed(
		tapa::istream<MergeAllocator8DispatchToken>& dispatch_input,
		tapa::mmap<const id_t> global_bases,
		tapa::mmap<const ap_uint<64> > source0,
		tapa::mmap<const ap_uint<64> > source1,
		tapa::mmap<const ap_uint<64> > source2,
		tapa::mmap<const ap_uint<64> > source3,
		tapa::mmap<const ap_uint<64> > source4,
		tapa::mmap<const ap_uint<64> > source5,
		tapa::mmap<const ap_uint<64> > source6,
		tapa::mmap<const ap_uint<64> > source7,
		tapa::ostream<MergeForwardTree8Token>& leaf0,
		tapa::ostream<MergeForwardTree8Token>& leaf1,
		tapa::ostream<MergeForwardTree8Token>& leaf2,
		tapa::ostream<MergeForwardTree8Token>& leaf3,
		tapa::ostream<MergeForwardTree8Token>& leaf4,
		tapa::ostream<MergeForwardTree8Token>& leaf5,
		tapa::ostream<MergeForwardTree8Token>& leaf6,
		tapa::ostream<MergeForwardTree8Token>& leaf7,
		tapa::ostream<BaseToken>& base0, tapa::ostream<BaseToken>& base1,
		tapa::ostream<BaseToken>& base2, tapa::ostream<BaseToken>& base3,
		tapa::ostream<BaseToken>& base4, tapa::ostream<BaseToken>& base5,
		tapa::ostream<BaseToken>& base6, tapa::ostream<BaseToken>& base7,
		tapa::ostream<BaseToken>& base8, tapa::ostream<BaseToken>& base9,
		tapa::ostream<BaseToken>& base10, tapa::ostream<BaseToken>& base11,
		tapa::ostream<BaseToken>& base12, tapa::ostream<BaseToken>& base13,
		tapa::ostream<BaseToken>& base14) {
	id_t cursor[8] = {};
#pragma HLS ARRAY_PARTITION variable=cursor complete
	ap_uint<16> sequence = 0;
	bool done = false;
	while (!done) {
		const MergeAllocator8DispatchToken dispatch_token
			= dispatch_input.read();
		if (dispatch_token[128]) {
			const MergeForwardTree8Token end = terminal_token();
			leaf0.write(end); leaf1.write(end); leaf2.write(end); leaf3.write(end);
			leaf4.write(end); leaf5.write(end); leaf6.write(end); leaf7.write(end);
			const BaseToken base_end = end_base();
			base0.write(base_end); base1.write(base_end); base2.write(base_end);
			base3.write(base_end); base4.write(base_end); base5.write(base_end);
			base6.write(base_end); base7.write(base_end); base8.write(base_end);
			base9.write(base_end); base10.write(base_end); base11.write(base_end);
			base12.write(base_end); base13.write(base_end); base14.write(base_end);
			done = true;
			continue;
		}
		const ap_uint<128> dispatch = dispatch_token.range(127, 0);
		const id_t row = dispatch.range(31, 0);
		const id_t descriptor = dispatch.range(63, 32);
		const ap_uint<8> source_mask = dispatch.range(71, 64);
		const ap_uint<8> leaf_mask = dispatch.range(79, 72);
		const ap_uint<3> context = dispatch.range(82, 80);
		const ap_uint<8> generation = dispatch.range(90, 83);
		const ap_uint<2> level = dispatch.range(92, 91);
		const ap_uint<3> leaf_base = dispatch.range(95, 93) << level;
		const ap_uint<24> mapping = dispatch.range(120, 97);
		const ap_uint<4> exit = level == 0 ? (ap_uint<4>)leaf_base
			: level == 1 ? (ap_uint<4>)(8 + (leaf_base >> 1))
			: level == 2 ? (ap_uint<4>)(12 + (leaf_base >> 2))
			: (ap_uint<4>)14;
		BaseToken base = 0;
		base.range(31, 0) = row;
		base.range(63, 32) = global_bases[descriptor];
		write_base(base, exit, base0, base1, base2, base3, base4, base5,
			base6, base7, base8, base9, base10, base11, base12, base13,
			base14);

		id_t count[8] = {};
#pragma HLS ARRAY_PARTITION variable=count complete
		if (source_mask[0]) count[0] = source0[cursor[0]++];
		if (source_mask[1]) count[1] = source1[cursor[1]++];
		if (source_mask[2]) count[2] = source2[cursor[2]++];
		if (source_mask[3]) count[3] = source3[cursor[3]++];
		if (source_mask[4]) count[4] = source4[cursor[4]++];
		if (source_mask[5]) count[5] = source5[cursor[5]++];
		if (source_mask[6]) count[6] = source6[cursor[6]++];
		if (source_mask[7]) count[7] = source7[cursor[7]++];
		id_t maximum = 0;
		for (int source = 0; source < 8; ++source) {
#pragma HLS UNROLL
			if (count[source] > maximum) maximum = count[source];
		}
		for (id_t position = 0; position < maximum; ++position) {
#pragma HLS PIPELINE II=1
#define EMIT_SOURCE(S, MEMORY) \
			if (position < count[S]) { \
				const ap_uint<64> numerical = MEMORY[cursor[S]++]; \
				MergeForwardTree8Token token = 0; \
				token.range(127, 0) = make_tree_item(numerical, row, context, \
					generation, level, false, sequence); \
				write_leaf(token, mapping.range(S * 3 + 2, S * 3), \
					leaf0, leaf1, leaf2, leaf3, leaf4, leaf5, leaf6, leaf7); \
			}
			EMIT_SOURCE(0, source0)
			EMIT_SOURCE(1, source1)
			EMIT_SOURCE(2, source2)
			EMIT_SOURCE(3, source3)
			EMIT_SOURCE(4, source4)
			EMIT_SOURCE(5, source5)
			EMIT_SOURCE(6, source6)
			EMIT_SOURCE(7, source7)
#undef EMIT_SOURCE
		}
		MergeForwardTree8Token eor = 0;
		eor.range(127, 0) = make_tree_item(0, row, context, generation,
			level, true, sequence);
		if (leaf_mask[0]) leaf0.write(eor);
		if (leaf_mask[1]) leaf1.write(eor);
		if (leaf_mask[2]) leaf2.write(eor);
		if (leaf_mask[3]) leaf3.write(eor);
		if (leaf_mask[4]) leaf4.write(eor);
		if (leaf_mask[5]) leaf5.write(eor);
		if (leaf_mask[6]) leaf6.write(eor);
		if (leaf_mask[7]) leaf7.write(eor);
		++sequence;
	}
}

static ap_uint<3> unified_mapped_leaf(ap_uint<128> dispatch,
		ap_uint<3> source) {
#pragma HLS INLINE
	switch ((unsigned)source) {
	case 0: return dispatch.range(99, 97);
	case 1: return dispatch.range(102, 100);
	case 2: return dispatch.range(105, 103);
	case 3: return dispatch.range(108, 106);
	case 4: return dispatch.range(111, 109);
	case 5: return dispatch.range(114, 112);
	case 6: return dispatch.range(117, 115);
	default: return dispatch.range(120, 118);
	}
}

// The low-rate command/allocator stage broadcasts one tagged dispatch to
// every independent source reader.  It also sends exactly one column-base
// token to the row's selected physical tree exit.
void unified_allocator_dispatch_broadcast(
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
		tapa::ostream<BaseToken>& base14) {
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
			done = true;
			continue;
		}
		const ap_uint<128> dispatch = dispatch_token.range(127, 0);
		const id_t row = dispatch.range(31, 0);
		const id_t descriptor = dispatch.range(63, 32);
		const ap_uint<2> level = dispatch.range(92, 91);
		const ap_uint<3> leaf_base = dispatch.range(95, 93) << level;
		const ap_uint<4> exit = level == 0 ? (ap_uint<4>)leaf_base
			: level == 1 ? (ap_uint<4>)(8 + (leaf_base >> 1))
			: level == 2 ? (ap_uint<4>)(12 + (leaf_base >> 2))
			: (ap_uint<4>)14;
		BaseToken base = 0;
		base.range(31, 0) = row;
		base.range(63, 32) = global_bases[descriptor];
		write_base(base, exit, base0, base1, base2, base3, base4, base5,
			base6, base7, base8, base9, base10, base11, base12, base13,
			base14);
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

// One instance owns one physical HBM shard.  It emits one item per cycle to
// exactly one leaf-destination FIFO.  Rounded buddy allocations are closed by
// assigning their empty leaf EORs to otherwise inactive source instances.
void unified_allocator_source_read(
		tapa::mmap<const ap_uint<64> > source_memory,
		ap_uint<3> source_index,
		tapa::istream<UnifiedSourceDispatchToken>& dispatch_input,
		tapa::ostream<UnifiedRoutedLeafToken>& output) {
	id_t cursor = 0;
	bool done = false;
	while (!done) {
		const UnifiedSourceDispatchToken source_dispatch
			= dispatch_input.read();
		if (source_dispatch[144]) {
			UnifiedRoutedLeafToken end = 0;
			end[132] = 1;
			output.write(end);
			done = true;
			continue;
		}
		const ap_uint<128> dispatch = source_dispatch.range(127, 0);
		const ap_uint<16> sequence = source_dispatch.range(143, 128);
		const id_t row = dispatch.range(31, 0);
		const ap_uint<8> source_mask = dispatch.range(71, 64);
		const ap_uint<3> context = dispatch.range(82, 80);
		const ap_uint<8> generation = dispatch.range(90, 83);
		const ap_uint<2> level = dispatch.range(92, 91);
		const ap_uint<3> leaf_base = dispatch.range(95, 93) << level;
		const ap_uint<4> active_count
			= allocator_merge_output_allocator_popcount8(source_mask);
		const ap_uint<4> width = allocation_width(source_mask);
		const bool active = source_mask[source_index];
		ap_uint<4> inactive_rank = 0;
		for (int source = 0; source < 8; ++source) {
#pragma HLS UNROLL
			if (source < source_index && !source_mask[source]) ++inactive_rank;
		}
		const bool padding = !active && inactive_rank < width - active_count;
		const ap_uint<3> target_leaf = active
			? unified_mapped_leaf(dispatch, source_index)
			: (ap_uint<3>)(leaf_base + active_count + inactive_rank);
		if (active) {
			const id_t count = source_memory[cursor++];
			for (id_t position = 0; position < count; ++position) {
#pragma HLS PIPELINE II=1
				const ap_uint<64> numerical = source_memory[cursor++];
				MergeForwardTree8Token token = 0;
				token.range(127, 0) = make_tree_item(numerical, row, context,
					generation, level, false, sequence);
				UnifiedRoutedLeafToken routed = 0;
				routed.range(128, 0) = token;
				routed.range(131, 129) = target_leaf;
				output.write(routed);
			}
		}
		if (active || padding) {
			MergeForwardTree8Token eor = 0;
			eor.range(127, 0) = make_tree_item(0, row, context, generation,
				level, true, sequence);
			UnifiedRoutedLeafToken routed = 0;
			routed.range(128, 0) = eor;
			routed.range(131, 129) = target_leaf;
			output.write(routed);
		}
	}
}

// All active allocations own disjoint leaves.  Therefore at most one held
// source token can target any leaf; the crossbar can commit up to eight tokens
// per cycle without arbitration between numerical producers.
void unified_allocator_source_crossbar8(
		tapa::istream<UnifiedRoutedLeafToken>& source0,
		tapa::istream<UnifiedRoutedLeafToken>& source1,
		tapa::istream<UnifiedRoutedLeafToken>& source2,
		tapa::istream<UnifiedRoutedLeafToken>& source3,
		tapa::istream<UnifiedRoutedLeafToken>& source4,
		tapa::istream<UnifiedRoutedLeafToken>& source5,
		tapa::istream<UnifiedRoutedLeafToken>& source6,
		tapa::istream<UnifiedRoutedLeafToken>& source7,
		tapa::ostream<MergeForwardTree8Token>& leaf0,
		tapa::ostream<MergeForwardTree8Token>& leaf1,
		tapa::ostream<MergeForwardTree8Token>& leaf2,
		tapa::ostream<MergeForwardTree8Token>& leaf3,
		tapa::ostream<MergeForwardTree8Token>& leaf4,
		tapa::ostream<MergeForwardTree8Token>& leaf5,
		tapa::ostream<MergeForwardTree8Token>& leaf6,
		tapa::ostream<MergeForwardTree8Token>& leaf7) {
	UnifiedRoutedLeafToken held[8] = {};
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
#define CROSSBAR_READ(S, STREAM) \
		if (!ended[S] && !valid[S]) { \
			UnifiedRoutedLeafToken token; \
			if (STREAM.try_read(token)) { \
				if (token[132]) ended[S] = true; \
				else { held[S] = token; valid[S] = true; } \
			} \
		}
		CROSSBAR_READ(0, source0)
		CROSSBAR_READ(1, source1)
		CROSSBAR_READ(2, source2)
		CROSSBAR_READ(3, source3)
		CROSSBAR_READ(4, source4)
		CROSSBAR_READ(5, source5)
		CROSSBAR_READ(6, source6)
		CROSSBAR_READ(7, source7)
#undef CROSSBAR_READ
		bool emitted[8] = {};
#pragma HLS ARRAY_PARTITION variable=emitted complete
#define CROSSBAR_WRITE(LEAF, STREAM) \
		{ \
			bool have = false; \
			ap_uint<3> selected = 0; \
			for (int source = 0; source < 8; ++source) { \
 _Pragma("HLS UNROLL") \
				if (!have && valid[source] \
						&& held[source].range(131, 129) == LEAF) { \
					have = true; selected = source; \
				} \
			} \
			if (have && STREAM.try_write(held[selected].range(128, 0))) \
				emitted[selected] = true; \
		}
		CROSSBAR_WRITE(0, leaf0)
		CROSSBAR_WRITE(1, leaf1)
		CROSSBAR_WRITE(2, leaf2)
		CROSSBAR_WRITE(3, leaf3)
		CROSSBAR_WRITE(4, leaf4)
		CROSSBAR_WRITE(5, leaf5)
		CROSSBAR_WRITE(6, leaf6)
		CROSSBAR_WRITE(7, leaf7)
#undef CROSSBAR_WRITE
		for (int source = 0; source < 8; ++source) {
#pragma HLS UNROLL
			if (emitted[source]) valid[source] = false;
		}
	}
	const MergeForwardTree8Token end = terminal_token();
	leaf0.write(end); leaf1.write(end); leaf2.write(end); leaf3.write(end);
	leaf4.write(end); leaf5.write(end); leaf6.write(end); leaf7.write(end);
}

// One stage of an 8x8 self-routing network.  Inputs are passed in four
// adjacent pairs by the top-level wiring.  Each pair routes on one destination
// bit and buffers a same-direction conflict for a later cycle.  Three such
// stages replace the timing-heavy eight-way priority/barrel crossbar.
void unified_allocator_crossbar_stage8(
		tapa::istream<UnifiedRoutedLeafToken>& input0,
		tapa::istream<UnifiedRoutedLeafToken>& input1,
		tapa::istream<UnifiedRoutedLeafToken>& input2,
		tapa::istream<UnifiedRoutedLeafToken>& input3,
		tapa::istream<UnifiedRoutedLeafToken>& input4,
		tapa::istream<UnifiedRoutedLeafToken>& input5,
		tapa::istream<UnifiedRoutedLeafToken>& input6,
		tapa::istream<UnifiedRoutedLeafToken>& input7,
		ap_uint<2> route_bit,
		tapa::ostream<UnifiedRoutedLeafToken>& output0,
		tapa::ostream<UnifiedRoutedLeafToken>& output1,
		tapa::ostream<UnifiedRoutedLeafToken>& output2,
		tapa::ostream<UnifiedRoutedLeafToken>& output3,
		tapa::ostream<UnifiedRoutedLeafToken>& output4,
		tapa::ostream<UnifiedRoutedLeafToken>& output5,
		tapa::ostream<UnifiedRoutedLeafToken>& output6,
		tapa::ostream<UnifiedRoutedLeafToken>& output7) {
	UnifiedRoutedLeafToken held[8] = {};
	bool valid[8] = {};
	bool ended[8] = {};
	bool turn[4] = {};
#pragma HLS ARRAY_PARTITION variable=held complete
#pragma HLS ARRAY_PARTITION variable=valid complete
#pragma HLS ARRAY_PARTITION variable=ended complete
#pragma HLS ARRAY_PARTITION variable=turn complete
	while (!(ended[0] && ended[1] && ended[2] && ended[3]
			&& ended[4] && ended[5] && ended[6] && ended[7]
			&& !valid[0] && !valid[1] && !valid[2] && !valid[3]
			&& !valid[4] && !valid[5] && !valid[6] && !valid[7])) {
#pragma HLS PIPELINE II=1
#define STAGE_READ(S, STREAM) \
		if (!ended[S] && !valid[S]) { \
			UnifiedRoutedLeafToken token; \
			if (STREAM.try_read(token)) { \
				if (token[132]) ended[S] = true; \
				else { held[S] = token; valid[S] = true; } \
			} \
		}
		STAGE_READ(0, input0)
		STAGE_READ(1, input1)
		STAGE_READ(2, input2)
		STAGE_READ(3, input3)
		STAGE_READ(4, input4)
		STAGE_READ(5, input5)
		STAGE_READ(6, input6)
		STAGE_READ(7, input7)
#undef STAGE_READ
		bool emitted[8] = {};
#pragma HLS ARRAY_PARTITION variable=emitted complete
#define ROUTE_PAIR(PAIR, A, B, LOW, HIGH) \
		{ \
			const ap_uint<3> dest_a = held[A].range(131, 129); \
			const ap_uint<3> dest_b = held[B].range(131, 129); \
			const bool a_low = valid[A] && !dest_a[route_bit]; \
			const bool b_low = valid[B] && !dest_b[route_bit]; \
			const bool a_high = valid[A] && dest_a[route_bit]; \
			const bool b_high = valid[B] && dest_b[route_bit]; \
			const bool select_a_low = a_low && (!b_low || !turn[PAIR]); \
			const bool select_a_high = a_high && (!b_high || !turn[PAIR]); \
			if ((a_low || b_low) \
					&& LOW.try_write(select_a_low ? held[A] : held[B])) { \
				if (select_a_low) emitted[A] = true; else emitted[B] = true; \
				if (a_low && b_low) turn[PAIR] = !turn[PAIR]; \
			} \
			if ((a_high || b_high) \
					&& HIGH.try_write(select_a_high ? held[A] : held[B])) { \
				if (select_a_high) emitted[A] = true; else emitted[B] = true; \
				if (a_high && b_high) turn[PAIR] = !turn[PAIR]; \
			} \
		}
		ROUTE_PAIR(0, 0, 1, output0, output1)
		ROUTE_PAIR(1, 2, 3, output2, output3)
		ROUTE_PAIR(2, 4, 5, output4, output5)
		ROUTE_PAIR(3, 6, 7, output6, output7)
#undef ROUTE_PAIR
		for (int source = 0; source < 8; ++source) {
#pragma HLS UNROLL
			if (emitted[source]) valid[source] = false;
		}
	}
	UnifiedRoutedLeafToken end = 0;
	end[132] = 1;
	output0.write(end); output1.write(end); output2.write(end); output3.write(end);
	output4.write(end); output5.write(end); output6.write(end); output7.write(end);
}

void unified_allocator_crossbar_finish8(
		tapa::istream<UnifiedRoutedLeafToken>& input0,
		tapa::istream<UnifiedRoutedLeafToken>& input1,
		tapa::istream<UnifiedRoutedLeafToken>& input2,
		tapa::istream<UnifiedRoutedLeafToken>& input3,
		tapa::istream<UnifiedRoutedLeafToken>& input4,
		tapa::istream<UnifiedRoutedLeafToken>& input5,
		tapa::istream<UnifiedRoutedLeafToken>& input6,
		tapa::istream<UnifiedRoutedLeafToken>& input7,
		tapa::ostream<MergeForwardTree8Token>& output0,
		tapa::ostream<MergeForwardTree8Token>& output1,
		tapa::ostream<MergeForwardTree8Token>& output2,
		tapa::ostream<MergeForwardTree8Token>& output3,
		tapa::ostream<MergeForwardTree8Token>& output4,
		tapa::ostream<MergeForwardTree8Token>& output5,
		tapa::ostream<MergeForwardTree8Token>& output6,
		tapa::ostream<MergeForwardTree8Token>& output7) {
	bool ended[8] = {};
#pragma HLS ARRAY_PARTITION variable=ended complete
	while (!(ended[0] && ended[1] && ended[2] && ended[3]
			&& ended[4] && ended[5] && ended[6] && ended[7])) {
#pragma HLS PIPELINE II=1
#define FINISH_LANE(L, INPUT, OUTPUT) \
		if (!ended[L]) { \
			UnifiedRoutedLeafToken token; \
			if (INPUT.try_read(token)) { \
				if (token[132]) ended[L] = true; \
				else OUTPUT.write(token.range(128, 0)); \
			} \
		}
		FINISH_LANE(0, input0, output0)
		FINISH_LANE(1, input1, output1)
		FINISH_LANE(2, input2, output2)
		FINISH_LANE(3, input3, output3)
		FINISH_LANE(4, input4, output4)
		FINISH_LANE(5, input5, output5)
		FINISH_LANE(6, input6, output6)
		FINISH_LANE(7, input7, output7)
#undef FINISH_LANE
	}
	const MergeForwardTree8Token end = terminal_token();
	output0.write(end); output1.write(end); output2.write(end); output3.write(end);
	output4.write(end); output5.write(end); output6.write(end); output7.write(end);
}

void unified_allocator_leaf_mux8(
		tapa::istream<MergeForwardTree8Token>& source0,
		tapa::istream<MergeForwardTree8Token>& source1,
		tapa::istream<MergeForwardTree8Token>& source2,
		tapa::istream<MergeForwardTree8Token>& source3,
		tapa::istream<MergeForwardTree8Token>& source4,
		tapa::istream<MergeForwardTree8Token>& source5,
		tapa::istream<MergeForwardTree8Token>& source6,
		tapa::istream<MergeForwardTree8Token>& source7,
		tapa::ostream<MergeForwardTree8Token>& output) {
	MergeForwardTree8Token held[8] = {};
	bool valid[8] = {};
	bool ended[8] = {};
#pragma HLS ARRAY_PARTITION variable=held complete
#pragma HLS ARRAY_PARTITION variable=valid complete
#pragma HLS ARRAY_PARTITION variable=ended complete
	ap_uint<3> turn = 0;
	while (!(ended[0] && ended[1] && ended[2] && ended[3]
			&& ended[4] && ended[5] && ended[6] && ended[7]
			&& !valid[0] && !valid[1] && !valid[2] && !valid[3]
			&& !valid[4] && !valid[5] && !valid[6] && !valid[7])) {
#pragma HLS PIPELINE II=1
#define READ_SOURCE(S, STREAM) \
		if (!ended[S] && !valid[S]) { \
			MergeForwardTree8Token token; \
			if (STREAM.try_read(token)) { \
				if (token[128]) ended[S] = true; \
				else { held[S] = token; valid[S] = true; } \
			} \
		}
		READ_SOURCE(0, source0)
		READ_SOURCE(1, source1)
		READ_SOURCE(2, source2)
		READ_SOURCE(3, source3)
		READ_SOURCE(4, source4)
		READ_SOURCE(5, source5)
		READ_SOURCE(6, source6)
		READ_SOURCE(7, source7)
#undef READ_SOURCE
		ap_uint<3> selected = turn;
		bool have = false;
		for (int offset = 0; offset < 8; ++offset) {
#pragma HLS UNROLL
			const ap_uint<3> candidate = turn + offset;
			if (!have && valid[candidate]) {
				selected = candidate;
				have = true;
			}
		}
		if (have && output.try_write(held[selected])) {
			valid[selected] = false;
			turn = selected + 1;
		}
	}
	output.write(terminal_token());
}

void unified_allocator_ack_drain(
		tapa::istream<MergeAllocator8AckToken>& input) {
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		done = input.read()[64];
	}
}

void unified_allocator_completion_broadcast(
		tapa::istream<CompletionToken>& input,
		tapa::ostream<MergeAllocator8EorToken>& allocator_output,
		tapa::ostream<CompletionToken>& memory_output) {
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		CompletionToken token;
		if (!input.try_read(token)) continue;
		MergeAllocator8EorToken eor = 0;
		if (token[64]) eor[64] = 1;
		else eor.range(42, 0) = token.range(42, 0);
		allocator_output.write(eor);
		memory_output.write(token);
		done = token[64];
	}
}

void unified_allocator_empty_output(
		tapa::ostream<PacketBundle>& packet,
		tapa::ostream<CompletionToken>& completion) {
	packet.write(end_bundle());
	completion.write(end_completion());
}

void adaptive_hbm_unified_allocator_merge_output_tapa(
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
		tapa::mmap<id_t> allocator_stats) {
	tapa::stream<MergeAllocator8CommandToken, 16> command_stream("command_stream");
	tapa::stream<MergeAllocator8DispatchToken, 16> dispatch_stream("dispatch_stream");
	tapa::stream<MergeAllocator8EorToken, 16> allocator_eors("allocator_eors");
	tapa::stream<MergeAllocator8AckToken, 16> allocator_acks("allocator_acks");
	tapa::stream<CompletionToken, 16> memory_completions("memory_completions");
	tapa::stream<ap_uint<384>, 2> allocator_statistics("allocator_statistics");
	tapa::streams<UnifiedSourceDispatchToken, 8, 16> source_dispatches("source_dispatches");
	tapa::streams<UnifiedRoutedLeafToken, 8, 16> routed_sources("routed_sources");
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
	tapa::streams<BaseToken, 15, 4> bases("bases");
	tapa::streams<PacketBundle, 15, 4> exit_packets("exit_packets");
	tapa::streams<CompletionToken, 15, 4> exit_completions("exit_completions");
	tapa::stream<PacketBundle, 2> empty_packet("empty_packet");
	tapa::stream<CompletionToken, 2> empty_completion("empty_completion");
	tapa::streams<PacketBundle, 4, 8> port_packets("port_packets");
	tapa::streams<CompletionToken, 4, 8> port_completions("port_completions");
	tapa::stream<CompletionToken, 8> merged_completions("merged_completions");
	tapa::streams<RecordToken, 4, 8> records("records");
	tapa::stream<RecordBatch, 4> record_batches("record_batches");
	tapa::stream<WordToken, 4> meta_words("meta_words");
	tapa::stream<StatsToken, 2> meta_summary("meta_summary");

	tapa::task()
		.invoke(merge_allocator8_command_read, commands, command_count,
			command_stream)
		.invoke(unified_allocator_live_control, command_stream,
			allocator_eors, dispatch_stream, allocator_acks,
			allocator_statistics)
		.invoke(unified_allocator_ack_drain, allocator_acks)
		.invoke(unified_allocator_dispatch_broadcast, dispatch_stream,
			global_bases, source_dispatches[0], source_dispatches[1],
			source_dispatches[2], source_dispatches[3], source_dispatches[4],
			source_dispatches[5], source_dispatches[6], source_dispatches[7],
			bases[0], bases[1], bases[2],
			bases[3], bases[4], bases[5], bases[6], bases[7], bases[8], bases[9],
			bases[10], bases[11], bases[12], bases[13], bases[14])
#define UNIFIED_ALLOCATOR_SOURCE_READER(S, MEMORY) \
		.invoke(unified_allocator_source_read, MEMORY, S, source_dispatches[S], \
			routed_sources[S])
		UNIFIED_ALLOCATOR_SOURCE_READER(0, source0)
		UNIFIED_ALLOCATOR_SOURCE_READER(1, source1)
		UNIFIED_ALLOCATOR_SOURCE_READER(2, source2)
		UNIFIED_ALLOCATOR_SOURCE_READER(3, source3)
		UNIFIED_ALLOCATOR_SOURCE_READER(4, source4)
		UNIFIED_ALLOCATOR_SOURCE_READER(5, source5)
		UNIFIED_ALLOCATOR_SOURCE_READER(6, source6)
		UNIFIED_ALLOCATOR_SOURCE_READER(7, source7)
#undef UNIFIED_ALLOCATOR_SOURCE_READER
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
#define UNIFIED_ALLOCATOR_PACKETIZE(STREAM, EXIT, PORT) \
		.invoke(merge15_exit_packetize, STREAM, bases[EXIT], EXIT, PORT, \
			exit_packets[EXIT], exit_completions[EXIT])
		UNIFIED_ALLOCATOR_PACKETIZE(leaf_exit[0], 0, 0)
		UNIFIED_ALLOCATOR_PACKETIZE(leaf_exit[1], 1, 1)
		UNIFIED_ALLOCATOR_PACKETIZE(leaf_exit[2], 2, 2)
		UNIFIED_ALLOCATOR_PACKETIZE(leaf_exit[3], 3, 3)
		UNIFIED_ALLOCATOR_PACKETIZE(leaf_exit[4], 4, 0)
		UNIFIED_ALLOCATOR_PACKETIZE(leaf_exit[5], 5, 1)
		UNIFIED_ALLOCATOR_PACKETIZE(leaf_exit[6], 6, 2)
		UNIFIED_ALLOCATOR_PACKETIZE(leaf_exit[7], 7, 3)
		UNIFIED_ALLOCATOR_PACKETIZE(level1_exit[0], 8, 0)
		UNIFIED_ALLOCATOR_PACKETIZE(level1_exit[1], 9, 1)
		UNIFIED_ALLOCATOR_PACKETIZE(level1_exit[2], 10, 2)
		UNIFIED_ALLOCATOR_PACKETIZE(level1_exit[3], 11, 3)
		UNIFIED_ALLOCATOR_PACKETIZE(level2_exit[0], 12, 0)
		UNIFIED_ALLOCATOR_PACKETIZE(level2_exit[1], 13, 1)
		UNIFIED_ALLOCATOR_PACKETIZE(root_exit, 14, 2)
#undef UNIFIED_ALLOCATOR_PACKETIZE
		.invoke(unified_allocator_empty_output, empty_packet, empty_completion)
		.invoke(merge15_bundle_mux4, exit_packets[0], exit_packets[4],
			exit_packets[8], exit_packets[12], port_packets[0])
		.invoke(merge15_bundle_mux4, exit_packets[1], exit_packets[5],
			exit_packets[9], exit_packets[13], port_packets[1])
		.invoke(merge15_bundle_mux4, exit_packets[2], exit_packets[6],
			exit_packets[10], exit_packets[14], port_packets[2])
		.invoke(merge15_bundle_mux4, exit_packets[3], exit_packets[7],
			exit_packets[11], empty_packet, port_packets[3])
		.invoke(merge15_completion_mux4, exit_completions[0],
			exit_completions[4], exit_completions[8], exit_completions[12],
			port_completions[0])
		.invoke(merge15_completion_mux4, exit_completions[1],
			exit_completions[5], exit_completions[9], exit_completions[13],
			port_completions[1])
		.invoke(merge15_completion_mux4, exit_completions[2],
			exit_completions[6], exit_completions[10], exit_completions[14],
			port_completions[2])
		.invoke(merge15_completion_mux4, exit_completions[3],
			exit_completions[7], exit_completions[11], empty_completion,
			port_completions[3])
		.invoke(merge15_completion_mux4, port_completions[0],
			port_completions[1], port_completions[2], port_completions[3],
			merged_completions)
		.invoke(unified_allocator_completion_broadcast, merged_completions,
			allocator_eors, memory_completions)
		.invoke(merge15_port_write, port_packets[0],
			output_data_word_capacity_per_port, output_item0,
			output_port_stats0, records[0])
		.invoke(merge15_port_write, port_packets[1],
			output_data_word_capacity_per_port, output_item1,
			output_port_stats1, records[1])
		.invoke(merge15_port_write, port_packets[2],
			output_data_word_capacity_per_port, output_item2,
			output_port_stats2, records[2])
		.invoke(merge15_port_write, port_packets[3],
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
		.invoke(merge_allocator8_statistics_write, allocator_statistics,
			allocator_stats);
}
