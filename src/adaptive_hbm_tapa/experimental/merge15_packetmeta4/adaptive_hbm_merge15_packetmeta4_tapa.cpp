#include "adaptive_hbm_merge15_packetmeta4_tapa.h"

// CSim-only hook: real HBM can temporarily deassert write acceptance, whereas
// the default TAPA mmap model completes every store immediately.  The unified
// long-stream regression enables this delay to exercise bounded packet/meta
// FIFOs under downstream backpressure.  It is preprocessor-excluded from HLS.
#ifndef __SYNTHESIS__
#include <chrono>
#include <thread>
static unsigned g_merge15_csim_writer_delay_us = 0;
void merge15_csim_set_writer_delay_us(unsigned delay_us) {
	g_merge15_csim_writer_delay_us = delay_us;
}
#endif

namespace {

using MergeToken = ap_uint<129>;
using BaseToken = ap_uint<65>;
using PacketBundle = ap_uint<641>;
using RecordToken = ap_uint<129>;
using RecordBatch = ap_uint<516>;
using WordToken = ap_uint<513>;
using StatsToken = ap_uint<64>;
using CompletionToken = ap_uint<65>;

static PacketBundle end_bundle() {
#pragma HLS INLINE
	PacketBundle value = 0;
	value[640] = 1;
	return value;
}

static MergeToken end_merge() {
#pragma HLS INLINE
	MergeToken value = 0;
	value[128] = 1;
	return value;
}

static BaseToken end_base() {
#pragma HLS INLINE
	BaseToken value = 0;
	value[64] = 1;
	return value;
}

static CompletionToken end_completion() {
#pragma HLS INLINE
	CompletionToken value = 0;
	value[64] = 1;
	return value;
}

static ap_uint<3> popcount8(ap_uint<8> value) {
#pragma HLS INLINE
	const ap_uint<3> low = value[0] + value[1] + value[2] + value[3];
	const ap_uint<3> high = value[4] + value[5] + value[6] + value[7];
	return low + high;
}

static bool fp32_nonzero(ap_uint<32> value) {
#pragma HLS INLINE
	return value.range(30, 0) != 0;
}

static ap_uint<512> pack_lanes(const ap_uint<64> lane[8]) {
#pragma HLS INLINE
	ap_uint<512> word = 0;
	for (int index = 0; index < 8; ++index) {
#pragma HLS UNROLL
		word.range(index * 64 + 63, index * 64) = lane[index];
	}
	return word;
}

// Metadata is deliberately self-describing; Host reconstruction does not
// rely on the interleaving order of the four HBM streams.
//   row[31:0], row_nnz[63:32], valid[67:64], EOR[68], context[71:69],
//   route[73:72], logical exit[77:74], error[79:78], physical port[81:80],
//   generation[89:82], record-valid[127].
static ap_uint<128> make_record(id_t row, id_t row_nnz,
		ap_uint<4> valid, bool eor, ap_uint<3> context,
		ap_uint<4> exit, ap_uint<2> error, ap_uint<2> port,
		ap_uint<8> generation) {
#pragma HLS INLINE
	ap_uint<128> record = 0;
	record.range(31, 0) = row;
	record.range(63, 32) = row_nnz;
	record.range(67, 64) = valid;
	record[68] = eor;
	record.range(71, 69) = context;
	record.range(73, 72) = 1;  // MERGE_ROW; no new Host-visible mode.
	record.range(77, 74) = exit;
	record.range(79, 78) = error;
	record.range(81, 80) = port;
	record.range(89, 82) = generation;
	record[127] = 1;
	return record;
}

static PacketBundle make_bundle(ap_uint<512> word, ap_uint<128> record) {
#pragma HLS INLINE
	PacketBundle bundle = 0;
	bundle.range(511, 0) = word;
	bundle.range(639, 512) = record;
	return bundle;
}

static ap_uint<64> make_completion(id_t row, ap_uint<3> context,
		ap_uint<8> generation, ap_uint<4> exit) {
#pragma HLS INLINE
	// Bits [42:0] intentionally match MergeAllocator8EorToken.
	ap_uint<64> event = 0;
	event.range(2, 0) = context;
	event.range(10, 3) = generation;
	event.range(42, 11) = row;
	event.range(46, 43) = exit;
	return event;
}

static ap_uint<2> expected_level(ap_uint<4> exit) {
#pragma HLS INLINE
	return exit < 8 ? (ap_uint<2>)0
		: exit < 12 ? (ap_uint<2>)1
		: exit < 14 ? (ap_uint<2>)2 : (ap_uint<2>)3;
}

static void queue_write(ap_uint<128> queue[8], ap_uint<3> position,
		ap_uint<128> value) {
#pragma HLS INLINE
	switch ((unsigned)position) {
	case 0: queue[0] = value; break;
	case 1: queue[1] = value; break;
	case 2: queue[2] = value; break;
	case 3: queue[3] = value; break;
	case 4: queue[4] = value; break;
	case 5: queue[5] = value; break;
	case 6: queue[6] = value; break;
	default: queue[7] = value; break;
	}
}

}  // namespace

// One feeder models one physical-port group.  The real graph connects its
// four output streams directly to fixed tree exits.  A base token is emitted
// exactly once before every EOR-delimited row on an exit.
void merge15_source_feed(tapa::mmap<const Merge15PacketSourceRecord> source_in,
		id_t count, tapa::ostream<MergeToken>& output0,
		tapa::ostream<MergeToken>& output1,
		tapa::ostream<MergeToken>& output2,
		tapa::ostream<MergeToken>& output3,
		tapa::ostream<BaseToken>& base0,
		tapa::ostream<BaseToken>& base1,
		tapa::ostream<BaseToken>& base2,
		tapa::ostream<BaseToken>& base3) {
	bool active[4] = {};
#pragma HLS ARRAY_PARTITION variable=active complete
	for (id_t cursor = 0; cursor < count; ++cursor) {
#pragma HLS PIPELINE II=1
		const Merge15PacketSourceRecord source = source_in[cursor];
		MergeToken token = 0;
		token.range(127, 0) = source.range(127, 0);
		const ap_uint<2> slot = source.range(161, 160);
		BaseToken base = 0;
		base.range(31, 0) = token.range(79, 48);
		base.range(63, 32) = source.range(159, 128);
		if (!active[slot]) {
			switch ((unsigned)slot) {
			case 0: base0.write(base); break;
			case 1: base1.write(base); break;
			case 2: base2.write(base); break;
			default: base3.write(base); break;
			}
			active[slot] = true;
		}
		switch ((unsigned)slot) {
		case 0: output0.write(token); break;
		case 1: output1.write(token); break;
		case 2: output2.write(token); break;
		default: output3.write(token); break;
		}
		if (token[93]) active[slot] = false;
	}
	output0.write(end_merge());
	output1.write(end_merge());
	output2.write(end_merge());
	output3.write(end_merge());
	base0.write(end_base());
	base1.write(end_base());
	base2.write(end_base());
	base3.write(end_base());
}

// Each of the 15 exits has a private packetizer and row buffer.  Consequently
// unrelated rows never wait for a global row-order selector.  A completion is
// released only after the EOR bundle has entered the output FIFO.
void merge15_exit_packetize(tapa::istream<MergeToken>& input,
		tapa::istream<BaseToken>& bases, ap_uint<4> exit_id,
		ap_uint<2> port_id,
		tapa::ostream<PacketBundle>& output,
		tapa::ostream<CompletionToken>& completions) {
	bool active = false;
	id_t active_row = 0;
	id_t active_base = 0;
	ap_uint<3> active_context = 0;
	ap_uint<8> active_generation = 0;
	id_t active_nnz = 0;
	ap_uint<2> row_error = 0;
	ap_uint<64> lanes[8] = {};
#pragma HLS ARRAY_PARTITION variable=lanes complete
	ap_uint<4> pending = 0;
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1048576 avg=4096
		MergeToken input_token;
		if (!input.try_read(input_token)) continue;
		if (input_token[128]) {
			done = true;
			continue;
		}
		const ap_uint<128> token = input_token.range(127, 0);
		const id_t row = token.range(79, 48);
		const ap_uint<3> context = token.range(82, 80);
		const ap_uint<8> generation = token.range(90, 83);
		const bool eor = token[93];
		ap_uint<2> error = 0;
		if (token.range(92, 91) != expected_level(exit_id)) error = 1;
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
			pending = 0;
			for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
				lanes[lane] = 0;
			}
		} else if (active_row != row || active_context != context
				|| active_generation != generation) {
			error = 3;
		}
		if (row_error == 0 && error != 0) row_error = error;

		if (eor) {
			output.write(make_bundle(pack_lanes(lanes),
				make_record(row, active_nnz, pending, true, context, exit_id,
					row_error, port_id, generation)));
			CompletionToken completion = 0;
			completion.range(63, 0) = make_completion(row, context,
				generation, exit_id);
			completions.write(completion);
			active = false;
			pending = 0;
			row_error = 0;
			continue;
		}

		const ap_uint<32> fp32 = token.range(47, 16);
		const ap_uint<33> global_column
			= (ap_uint<33>)active_base + token.range(15, 0);
		if (!fp32_nonzero(fp32) || global_column[32]) {
			if (row_error == 0) row_error = 3;
			continue;
		}
		ap_uint<64> item = 0;
		item.range(31, 0) = global_column.range(31, 0);
		item.range(63, 32) = fp32;
		lanes[(unsigned)pending.range(2, 0)] = item;
		++pending;
		++active_nnz;
		if (pending == 8) {
			output.write(make_bundle(pack_lanes(lanes),
				make_record(row, 0, 8, false, context, exit_id,
					row_error, port_id, generation)));
			pending = 0;
			for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
				lanes[lane] = 0;
			}
		}
	}
	if (active || pending != 0) {
		output.write(make_bundle(pack_lanes(lanes),
			make_record(active_row, active_nnz, pending, true,
				active_context, exit_id, 3, port_id, active_generation)));
	}
	BaseToken base_end = bases.read();
	while (!base_end[64]) base_end = bases.read();
	output.write(end_bundle());
	completions.write(end_completion());
}

void merge15_bundle_mux4(tapa::istream<PacketBundle>& input0,
		tapa::istream<PacketBundle>& input1,
		tapa::istream<PacketBundle>& input2,
		tapa::istream<PacketBundle>& input3,
		tapa::ostream<PacketBundle>& output) {
	PacketBundle held[4] = {};
	bool valid[4] = {};
	bool ended[4] = {};
#pragma HLS ARRAY_PARTITION variable=held complete
#pragma HLS ARRAY_PARTITION variable=valid complete
#pragma HLS ARRAY_PARTITION variable=ended complete
	ap_uint<2> turn = 0;
	while (!(ended[0] && ended[1] && ended[2] && ended[3]
			&& !valid[0] && !valid[1] && !valid[2] && !valid[3])) {
#pragma HLS PIPELINE II=1
		PacketBundle next;
		if (!ended[0] && !valid[0] && input0.try_read(next)) {
			if (next[640]) ended[0] = true; else { held[0] = next; valid[0] = true; }
		}
		if (!ended[1] && !valid[1] && input1.try_read(next)) {
			if (next[640]) ended[1] = true; else { held[1] = next; valid[1] = true; }
		}
		if (!ended[2] && !valid[2] && input2.try_read(next)) {
			if (next[640]) ended[2] = true; else { held[2] = next; valid[2] = true; }
		}
		if (!ended[3] && !valid[3] && input3.try_read(next)) {
			if (next[640]) ended[3] = true; else { held[3] = next; valid[3] = true; }
		}
		ap_uint<2> selected = turn;
		bool have = false;
		for (int offset = 0; offset < 4; ++offset) {
#pragma HLS UNROLL
			const ap_uint<2> candidate = turn + offset;
			if (!have && valid[candidate]) { selected = candidate; have = true; }
		}
		if (have && output.try_write(held[selected])) {
			valid[selected] = false;
			turn = selected + 1;
		}
	}
	output.write(end_bundle());
}

void merge15_completion_mux4(tapa::istream<CompletionToken>& input0,
		tapa::istream<CompletionToken>& input1,
		tapa::istream<CompletionToken>& input2,
		tapa::istream<CompletionToken>& input3,
		tapa::ostream<CompletionToken>& output) {
	CompletionToken held[4] = {};
	bool valid[4] = {};
	bool ended[4] = {};
#pragma HLS ARRAY_PARTITION variable=held complete
#pragma HLS ARRAY_PARTITION variable=valid complete
#pragma HLS ARRAY_PARTITION variable=ended complete
	ap_uint<2> turn = 0;
	while (!(ended[0] && ended[1] && ended[2] && ended[3]
			&& !valid[0] && !valid[1] && !valid[2] && !valid[3])) {
#pragma HLS PIPELINE II=1
		CompletionToken next;
		if (!ended[0] && !valid[0] && input0.try_read(next)) {
			if (next[64]) ended[0] = true; else { held[0] = next; valid[0] = true; }
		}
		if (!ended[1] && !valid[1] && input1.try_read(next)) {
			if (next[64]) ended[1] = true; else { held[1] = next; valid[1] = true; }
		}
		if (!ended[2] && !valid[2] && input2.try_read(next)) {
			if (next[64]) ended[2] = true; else { held[2] = next; valid[2] = true; }
		}
		if (!ended[3] && !valid[3] && input3.try_read(next)) {
			if (next[64]) ended[3] = true; else { held[3] = next; valid[3] = true; }
		}
		ap_uint<2> selected = turn;
		bool have = false;
		for (int offset = 0; offset < 4; ++offset) {
#pragma HLS UNROLL
			const ap_uint<2> candidate = turn + offset;
			if (!have && valid[candidate]) { selected = candidate; have = true; }
		}
		if (have && output.try_write(held[selected])) {
			valid[selected] = false;
			turn = selected + 1;
		}
	}
	output.write(end_completion());
}

void merge15_port_write(tapa::istream<PacketBundle>& input, id_t capacity,
		tapa::mmap<ap_uint<512> > data_out, tapa::mmap<id_t> stats,
		tapa::ostream<RecordToken>& records) {
	id_t words = 0;
	id_t items = 0;
	id_t eors = 0;
	bool overflow = false;
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
#ifndef __SYNTHESIS__
		if (g_merge15_csim_writer_delay_us != 0)
			std::this_thread::sleep_for(std::chrono::microseconds(
				g_merge15_csim_writer_delay_us));
#endif
		const PacketBundle bundle = input.read();
		if (bundle[640]) {
			RecordToken end = 0;
			end[128] = 1;
			records.write(end);
			done = true;
		} else {
			const ap_uint<128> record = bundle.range(639, 512);
			const ap_uint<4> valid = record.range(67, 64);
			if (valid != 0) {
				if (words < capacity) data_out[words] = bundle.range(511, 0);
				else overflow = true;
				++words;
				items += valid;
			}
			if (record[68]) ++eors;
			RecordToken token = 0;
			token.range(127, 0) = record;
			records.write(token);
		}
	}
	stats[0] = words;
	stats[1] = items;
	stats[2] = eors;
	stats[3] = overflow ? 7 : 0;
}

void merge15_meta_collect(tapa::istream<RecordToken>& input0,
		tapa::istream<RecordToken>& input1,
		tapa::istream<RecordToken>& input2,
		tapa::istream<RecordToken>& input3,
		tapa::ostream<RecordBatch>& output,
		tapa::ostream<StatsToken>& summary) {
	bool done[4] = {};
#pragma HLS ARRAY_PARTITION variable=done complete
	id_t record_count = 0;
	id_t eor_count = 0;
	while (true) {
#pragma HLS PIPELINE II=1
		RecordToken incoming[4] = {};
		bool got[4] = {};
#pragma HLS ARRAY_PARTITION variable=incoming complete
#pragma HLS ARRAY_PARTITION variable=got complete
		if (!done[0]) got[0] = input0.try_read(incoming[0]);
		if (!done[1]) got[1] = input1.try_read(incoming[1]);
		if (!done[2]) got[2] = input2.try_read(incoming[2]);
		if (!done[3]) got[3] = input3.try_read(incoming[3]);
		const bool valid0 = got[0] && !incoming[0][128];
		const bool valid1 = got[1] && !incoming[1][128];
		const bool valid2 = got[2] && !incoming[2][128];
		const bool valid3 = got[3] && !incoming[3][128];
		if (got[0] && incoming[0][128]) done[0] = true;
		if (got[1] && incoming[1][128]) done[1] = true;
		if (got[2] && incoming[2][128]) done[2] = true;
		if (got[3] && incoming[3][128]) done[3] = true;
		const ap_uint<2> count01 = (ap_uint<2>)valid0 + valid1;
		const ap_uint<2> count23 = (ap_uint<2>)valid2 + valid3;
		const ap_uint<3> count = count01 + count23;
		const ap_uint<3> eors
			= (ap_uint<3>)(valid0 && incoming[0][68])
			+ (ap_uint<3>)(valid1 && incoming[1][68])
			+ (ap_uint<3>)(valid2 && incoming[2][68])
			+ (ap_uint<3>)(valid3 && incoming[3][68]);

		// Balanced 2+2 prefix compaction.  Avoiding a variable 512-bit range
		// select removes the 30k-LUT barrel mux generated by the first gate.
		const ap_uint<128> record0 = incoming[0].range(127, 0);
		const ap_uint<128> record1 = incoming[1].range(127, 0);
		const ap_uint<128> record2 = incoming[2].range(127, 0);
		const ap_uint<128> record3 = incoming[3].range(127, 0);
		const ap_uint<128> pair01_first = valid0 ? record0 : record1;
		const ap_uint<128> pair23_first = valid2 ? record2 : record3;
		const ap_uint<128> compact0
			= count01 != 0 ? pair01_first : pair23_first;
		const ap_uint<128> compact1 = count01 == 0 ? record3
			: (count01 == 1 ? pair23_first : record1);
		const ap_uint<128> compact2
			= count01 == 1 ? record3 : pair23_first;
		const ap_uint<128> compact3 = record3;
		if (count != 0) {
			RecordBatch batch = 0;
			batch.range(127, 0) = compact0;
			batch.range(255, 128) = compact1;
			batch.range(383, 256) = compact2;
			batch.range(511, 384) = compact3;
			batch.range(514, 512) = count;
			output.write(batch);
			record_count += count;
			eor_count += eors;
		}
		if (done[0] && done[1] && done[2] && done[3]) break;
	}
	RecordBatch end = 0;
	end[515] = 1;
	output.write(end);
	StatsToken stats = 0;
	stats.range(31, 0) = record_count;
	stats.range(63, 32) = eor_count;
	summary.write(stats);
}

void merge15_meta_pack(tapa::istream<RecordBatch>& input,
		tapa::ostream<WordToken>& output) {
	ap_uint<128> queue[8] = {};
#pragma HLS ARRAY_PARTITION variable=queue complete
	ap_uint<4> queue_count = 0;
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		const RecordBatch batch = input.read();
		if (batch[515]) { done = true; continue; }
		ap_uint<4> retained = queue_count;
		if (queue_count >= 4) {
			WordToken word = 0;
			for (int index = 0; index < 4; ++index) {
#pragma HLS UNROLL
				word.range(index * 128 + 127, index * 128) = queue[index];
				queue[index] = queue[index + 4];
				queue[index + 4] = 0;
			}
			output.write(word);
			retained = queue_count - 4;
		}
		const ap_uint<3> count = batch.range(514, 512);
		for (int index = 0; index < 4; ++index) {
#pragma HLS UNROLL
			if (index < count) queue_write(queue, retained + index,
				batch.range(index * 128 + 127, index * 128));
		}
		queue_count = retained + count;
	}
	while (queue_count != 0) {
#pragma HLS PIPELINE II=1
		WordToken word = 0;
		const ap_uint<3> emit = queue_count >= 4 ? (ap_uint<3>)4
			: (ap_uint<3>)queue_count;
		for (int index = 0; index < 4; ++index) {
#pragma HLS UNROLL
			if (index < emit)
				word.range(index * 128 + 127, index * 128) = queue[index];
		}
		output.write(word);
		for (int index = 0; index < 4; ++index) {
#pragma HLS UNROLL
			queue[index] = queue[index + 4];
			queue[index + 4] = 0;
		}
		queue_count -= emit;
	}
	WordToken end = 0;
	end[512] = 1;
	output.write(end);
}

void merge15_meta_write(tapa::istream<WordToken>& input,
		tapa::istream<StatsToken>& summary, id_t capacity,
		tapa::mmap<ap_uint<512> > meta_out, tapa::mmap<id_t> stats) {
	id_t words = 0;
	bool overflow = false;
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		const WordToken token = input.read();
		if (token[512]) done = true;
		else {
			if (words < capacity) meta_out[words] = token.range(511, 0);
			else overflow = true;
			++words;
		}
	}
	const StatsToken totals = summary.read();
	stats[0] = words;
	stats[1] = totals.range(31, 0);
	stats[2] = totals.range(63, 32);
	stats[3] = overflow ? 7 : 0;
}

void merge15_completion_write(tapa::istream<CompletionToken>& input,
		id_t capacity, tapa::mmap<ap_uint<64> > completion_out,
		tapa::mmap<id_t> stats) {
	id_t count = 0;
	bool overflow = false;
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		const CompletionToken token = input.read();
		if (token[64]) done = true;
		else {
			if (count < capacity) completion_out[count] = token.range(63, 0);
			else overflow = true;
			++count;
		}
	}
	stats[0] = count;
	stats[1] = overflow ? 7 : 0;
}

void adaptive_hbm_merge15_packetmeta4_tapa(
		tapa::mmap<const Merge15PacketSourceRecord> source0, id_t source_count0,
		tapa::mmap<const Merge15PacketSourceRecord> source1, id_t source_count1,
		tapa::mmap<const Merge15PacketSourceRecord> source2, id_t source_count2,
		tapa::mmap<const Merge15PacketSourceRecord> source3, id_t source_count3,
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
		tapa::mmap<id_t> output_completion_stats) {
	tapa::streams<MergeToken, 4, 8> token0("token0");
	tapa::streams<MergeToken, 4, 8> token1("token1");
	tapa::streams<MergeToken, 4, 8> token2("token2");
	tapa::streams<MergeToken, 4, 8> token3("token3");
	tapa::streams<BaseToken, 4, 4> base0("base0");
	tapa::streams<BaseToken, 4, 4> base1("base1");
	tapa::streams<BaseToken, 4, 4> base2("base2");
	tapa::streams<BaseToken, 4, 4> base3("base3");
	tapa::streams<PacketBundle, 4, 4> packet0("packet0");
	tapa::streams<PacketBundle, 4, 4> packet1("packet1");
	tapa::streams<PacketBundle, 4, 4> packet2("packet2");
	tapa::streams<PacketBundle, 4, 4> packet3("packet3");
	tapa::streams<CompletionToken, 4, 4> completion0("completion0");
	tapa::streams<CompletionToken, 4, 4> completion1("completion1");
	tapa::streams<CompletionToken, 4, 4> completion2("completion2");
	tapa::streams<CompletionToken, 4, 4> completion3("completion3");
	tapa::streams<PacketBundle, 4, 8> port_packets("port_packets");
	tapa::streams<CompletionToken, 4, 8> port_completions("port_completions");
	tapa::stream<CompletionToken, 8> all_completions("all_completions");
	tapa::streams<RecordToken, 4, 8> records("records");
	tapa::stream<RecordBatch, 4> record_batches("record_batches");
	tapa::stream<WordToken, 4> meta_words("meta_words");
	tapa::stream<StatsToken, 2> meta_summary("meta_summary");

	tapa::task()
		.invoke(merge15_source_feed, source0, source_count0,
			token0[0], token0[1], token0[2], token0[3],
			base0[0], base0[1], base0[2], base0[3])
		.invoke(merge15_source_feed, source1, source_count1,
			token1[0], token1[1], token1[2], token1[3],
			base1[0], base1[1], base1[2], base1[3])
		.invoke(merge15_source_feed, source2, source_count2,
			token2[0], token2[1], token2[2], token2[3],
			base2[0], base2[1], base2[2], base2[3])
		.invoke(merge15_source_feed, source3, source_count3,
			token3[0], token3[1], token3[2], token3[3],
			base3[0], base3[1], base3[2], base3[3])
#define PACKETIZER(GROUP, SLOT, EXIT, PORT) \
		.invoke(merge15_exit_packetize, token##GROUP[SLOT], base##GROUP[SLOT], \
			EXIT, PORT, packet##GROUP[SLOT], completion##GROUP[SLOT])
		PACKETIZER(0, 0, 0, 0)
		PACKETIZER(0, 1, 4, 0)
		PACKETIZER(0, 2, 8, 0)
		PACKETIZER(0, 3, 12, 0)
		PACKETIZER(1, 0, 1, 1)
		PACKETIZER(1, 1, 5, 1)
		PACKETIZER(1, 2, 9, 1)
		PACKETIZER(1, 3, 13, 1)
		PACKETIZER(2, 0, 2, 2)
		PACKETIZER(2, 1, 6, 2)
		PACKETIZER(2, 2, 10, 2)
		PACKETIZER(2, 3, 14, 2)
		PACKETIZER(3, 0, 3, 3)
		PACKETIZER(3, 1, 7, 3)
		PACKETIZER(3, 2, 11, 3)
		PACKETIZER(3, 3, 15, 3)
#undef PACKETIZER
#define MUX_GROUP(GROUP) \
		.invoke(merge15_bundle_mux4, packet##GROUP[0], packet##GROUP[1], \
			packet##GROUP[2], packet##GROUP[3], port_packets[GROUP]) \
		.invoke(merge15_completion_mux4, completion##GROUP[0], \
			completion##GROUP[1], completion##GROUP[2], completion##GROUP[3], \
			port_completions[GROUP])
		MUX_GROUP(0)
		MUX_GROUP(1)
		MUX_GROUP(2)
		MUX_GROUP(3)
#undef MUX_GROUP
		.invoke(merge15_completion_mux4, port_completions[0],
			port_completions[1], port_completions[2], port_completions[3],
			all_completions)
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
		.invoke(merge15_completion_write, all_completions,
			output_completion_capacity, output_completion,
			output_completion_stats);
}
