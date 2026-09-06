#include "adaptive_hbm_merge_allocator8_tapa.h"

namespace {

static ap_uint<4> popcount8(ap_uint<8> value) {
#pragma HLS INLINE
	ap_uint<3> low = value[0] + value[1] + value[2] + value[3];
	ap_uint<3> high = value[4] + value[5] + value[6] + value[7];
	return low + high;
}

static ap_uint<4> allocation_width(ap_uint<8> source_mask) {
#pragma HLS INLINE
	const ap_uint<4> count = popcount8(source_mask);
	return count <= 1 ? (ap_uint<4>)1
		: count <= 2 ? (ap_uint<4>)2
		: count <= 4 ? (ap_uint<4>)4 : (ap_uint<4>)8;
}

static ap_uint<2> allocation_level(ap_uint<4> width) {
#pragma HLS INLINE
	return width == 1 ? (ap_uint<2>)0
		: width == 2 ? (ap_uint<2>)1
		: width == 4 ? (ap_uint<2>)2 : (ap_uint<2>)3;
}

static ap_uint<8> block_mask(ap_uint<4> width, ap_uint<3> base) {
#pragma HLS INLINE
	const ap_uint<9> ones = width == 8 ? (ap_uint<9>)255
		: width == 4 ? (ap_uint<9>)15
		: width == 2 ? (ap_uint<9>)3 : (ap_uint<9>)1;
	return (ap_uint<8>)(ones << base);
}

// Buddy-style aligned allocation maps directly onto the 8-leaf binary tree.
// It guarantees that width=1/2/4 rows can leave at a leaf/level-1/level-2
// node, respectively, without touching the root.
static bool find_leaf_block(ap_uint<8> free_leaf, ap_uint<4> width,
		ap_uint<3>& base, ap_uint<8>& mask) {
#pragma HLS INLINE
	bool found = false;
	base = 0;
	mask = 0;
	if (width == 8) {
		if (free_leaf == 0xff) { found = true; mask = 0xff; }
	} else if (width == 4) {
		if ((free_leaf & 0x0f) == 0x0f) {
			found = true; base = 0; mask = 0x0f;
		} else if ((free_leaf & 0xf0) == 0xf0) {
			found = true; base = 4; mask = 0xf0;
		}
	} else if (width == 2) {
		if ((free_leaf & 0x03) == 0x03) {
			found = true; base = 0; mask = 0x03;
		} else if ((free_leaf & 0x0c) == 0x0c) {
			found = true; base = 2; mask = 0x0c;
		} else if ((free_leaf & 0x30) == 0x30) {
			found = true; base = 4; mask = 0x30;
		} else if ((free_leaf & 0xc0) == 0xc0) {
			found = true; base = 6; mask = 0xc0;
		}
	} else {
		for (int leaf = 0; leaf < 8; ++leaf) {
#pragma HLS UNROLL
			if (!found && free_leaf[leaf]) {
				found = true;
				base = leaf;
				mask = (ap_uint<8>)1 << leaf;
			}
		}
	}
	return found;
}

static bool find_context(ap_uint<8> busy, ap_uint<3>& context) {
#pragma HLS INLINE
	bool found = false;
	context = 0;
	for (int slot = 0; slot < 8; ++slot) {
#pragma HLS UNROLL
		if (!found && !busy[slot]) {
			found = true;
			context = slot;
		}
	}
	return found;
}

static ap_uint<24> make_source_to_leaf_map(ap_uint<8> source_mask,
		ap_uint<3> leaf_base) {
#pragma HLS INLINE
	ap_uint<24> mapping = 0;
	ap_uint<4> rank = 0;
	for (int source = 0; source < 8; ++source) {
#pragma HLS UNROLL
		const ap_uint<3> target = leaf_base + rank;
		mapping.range(source * 3 + 2, source * 3) = target;
		if (source_mask[source]) ++rank;
	}
	return mapping;
}

static ap_uint<128> make_dispatch(ap_uint<128> command,
		ap_uint<8> leaf_mask, ap_uint<3> leaf_base,
		ap_uint<3> context, ap_uint<8> generation,
		ap_uint<4> width) {
#pragma HLS INLINE
	ap_uint<128> record = 0;
	const ap_uint<8> source_mask = command.range(71, 64);
	const ap_uint<2> level = allocation_level(width);
	record.range(63, 0) = command.range(63, 0);
	record.range(71, 64) = source_mask;
	record.range(79, 72) = leaf_mask;
	record.range(82, 80) = context;
	record.range(90, 83) = generation;
	record.range(92, 91) = level;
	record.range(95, 93) = leaf_base >> level;
	record[96] = level != 3;
	record.range(120, 97) = make_source_to_leaf_map(source_mask, leaf_base);
	return record;
}

static ap_uint<64> make_ack(ap_uint<64> event, bool accepted,
		ap_uint<3> status, ap_uint<8> released_mask) {
#pragma HLS INLINE
	ap_uint<64> ack = 0;
	ack.range(42, 0) = event.range(42, 0);
	ack[43] = accepted;
	ack.range(46, 44) = status;
	ack.range(54, 47) = released_mask;
	return ack;
}

}  // namespace

void merge_allocator8_command_read(
		tapa::mmap<const ap_uint<128> > command_in, id_t count,
		tapa::ostream<MergeAllocator8CommandToken>& output) {
	for (id_t cursor = 0; cursor < count; ++cursor) {
#pragma HLS PIPELINE II=1
		MergeAllocator8CommandToken token = 0;
		token.range(127, 0) = command_in[cursor];
		output.write(token);
	}
	MergeAllocator8CommandToken terminal = 0;
	terminal[128] = 1;
	output.write(terminal);
}

void merge_allocator8_eor_read(
		tapa::mmap<const ap_uint<64> > eor_in, id_t count,
		tapa::ostream<MergeAllocator8EorToken>& output) {
	for (id_t cursor = 0; cursor < count; ++cursor) {
#pragma HLS PIPELINE II=1
		MergeAllocator8EorToken token = 0;
		token.range(63, 0) = eor_in[cursor];
		output.write(token);
	}
	MergeAllocator8EorToken terminal = 0;
	terminal[64] = 1;
	output.write(terminal);
}

void merge_allocator8_control(
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
	id_t accepted_count = 0;
	id_t completed_count = 0;
	id_t peak_contexts = 0;
	id_t peak_leaves = 0;
	id_t exit_count[4] = {0, 0, 0, 0};
#pragma HLS ARRAY_PARTITION variable=exit_count complete
	id_t stale_count = 0;
	id_t blocked_cycles = 0;

	while (!(command_done && eor_done && !command_valid && !eor_valid
			&& busy == 0)) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=1048576 avg=1024
		// Deliberately allocate from the start-of-cycle snapshot.  A context
		// released by EOR becomes reusable on the following cycle, cutting the
		// release/compare/priority-encode/generation feedback path while still
		// accepting one command per cycle whenever existing space is available.
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

		// Bits[58:43] are a gate-only causal watermark: the real numerical
		// fabric naturally cannot produce EOR before dispatch.  It lets the
		// memory-backed CSim reproduce that ordering without timing assumptions.
		if (eor_valid
				&& accepted_count >= (id_t)held_eor.range(58, 43)) {
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
			const bool have_leaves = find_leaf_block(allocation_free_leaf, width,
				leaf_base, leaf_mask);
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
		}

		const id_t live_contexts = popcount8(busy);
		const id_t live_leaves = 8 - popcount8(free_leaf);
		if (live_contexts > peak_contexts) peak_contexts = live_contexts;
		if (live_leaves > peak_leaves) peak_leaves = live_leaves;
	}

	MergeAllocator8DispatchToken dispatch_end = 0;
	dispatch_end[128] = 1;
	dispatch_output.write(dispatch_end);
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

void merge_allocator8_dispatch_write(
		tapa::istream<MergeAllocator8DispatchToken>& input,
		tapa::mmap<ap_uint<128> > dispatch_out) {
	id_t cursor = 0;
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		const MergeAllocator8DispatchToken token = input.read();
		if (token[128]) done = true;
		else dispatch_out[cursor++] = token.range(127, 0);
	}
}

void merge_allocator8_ack_write(
		tapa::istream<MergeAllocator8AckToken>& input,
		tapa::mmap<ap_uint<64> > ack_out) {
	id_t cursor = 0;
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		const MergeAllocator8AckToken token = input.read();
		if (token[64]) done = true;
		else ack_out[cursor++] = token.range(63, 0);
	}
}

void merge_allocator8_statistics_write(
		tapa::istream<ap_uint<384> >& input,
		tapa::mmap<id_t> statistics) {
	const ap_uint<384> packed = input.read();
	for (int word = 0; word < 12; ++word) {
#pragma HLS PIPELINE II=1
		statistics[word] = packed.range(word * 32 + 31, word * 32);
	}
}

void adaptive_hbm_merge_allocator8_tapa(
		tapa::mmap<const ap_uint<128> > commands,
		id_t command_count,
		tapa::mmap<const ap_uint<64> > eor_events,
		id_t eor_count,
		tapa::mmap<ap_uint<128> > dispatches,
		tapa::mmap<ap_uint<64> > acknowledgements,
		tapa::mmap<id_t> statistics) {
	tapa::stream<MergeAllocator8CommandToken, 16> command_stream("command_stream");
	tapa::stream<MergeAllocator8EorToken, 16> eor_stream("eor_stream");
	tapa::stream<MergeAllocator8DispatchToken, 16> dispatch_stream("dispatch_stream");
	tapa::stream<MergeAllocator8AckToken, 16> ack_stream("ack_stream");
	tapa::stream<ap_uint<384>, 2> statistics_stream("statistics_stream");
	tapa::task()
		.invoke(merge_allocator8_command_read, commands, command_count,
			command_stream)
		.invoke(merge_allocator8_eor_read, eor_events, eor_count, eor_stream)
		.invoke(merge_allocator8_control, command_stream, eor_stream,
			dispatch_stream, ack_stream, statistics_stream)
		.invoke(merge_allocator8_dispatch_write, dispatch_stream, dispatches)
		.invoke(merge_allocator8_ack_write, ack_stream, acknowledgements)
		.invoke(merge_allocator8_statistics_write, statistics_stream, statistics);
}
