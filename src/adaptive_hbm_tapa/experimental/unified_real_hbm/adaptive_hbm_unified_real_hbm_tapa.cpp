#include "adaptive_hbm_unified_real_hbm_tapa.h"

// Reuse the accepted production reader/local-merge implementation verbatim.
// Keep it at global scope: tapacc 0.1.20250709 segfaults while traversing the
// same large translation unit when the implementation is nested in a private
// namespace, even if namespace-free task wrappers are added around it.  The
// production top functions have distinct names and are not instantiated by
// this graph, so global inclusion does not duplicate any hardware.
#include "../../adaptive_hbm_tapa_scalable.h"
#include "../../adaptive_hbm_tapa_scalable_windowed.cpp"

// The accepted allocator, flexible 15-exit tree, shared segmented reducer,
// keyed carry, dual DENSE contexts and four-port output tasks remain unchanged.
#include "../unified_adaptive_full_output/adaptive_hbm_unified_adaptive_full_output_tapa.cpp"

namespace {

using UnifiedRealProductPacket = TapaProductPacket;
using UnifiedRealRawPacket = TapaBReaderRawPacket;

// One token is broadcast once per physical output fragment to each reader.
// The reader obtains its own shard-local descriptor count/list from the tail
// of its B HBM bank, so a hot shard can never fill a central per-task command
// FIFO before the other shards receive their row boundary.
//   allocator dispatch[127:0], dense_base[159:128], span[191:160], done[192].
using UnifiedRealReaderDispatch = ap_uint<193>;

}  // namespace

// The per-fragment completion array is a host-side diagnostic trace.  It is
// redundant with the EOR metadata that already drives numerical validation.
// Do not put that HBM write on the allocator's resource-release path: a
// delayed diagnostic write can otherwise stop EOR delivery and form a cyclic
// backpressure loop through the scheduler.  The packetizer emits a completion
// only after it has emitted the final packet for the row, so forwarding it to
// the scheduler alone is sufficient to safely retire the context.
void unified_real_completion_to_scheduler(
		tapa::istream<CompletionToken>& input,
		tapa::ostream<MergeAllocator8EorToken>& allocator_output) {
	bool done = false;
	while (!done) {
#pragma HLS PIPELINE II=1
		const CompletionToken token = input.read();
		MergeAllocator8EorToken eor = 0;
		if (token[64]) eor[64] = 1;
		else eor.range(42, 0) = token.range(42, 0);
		allocator_output.write(eor);
		done = token[64];
	}
}

// Keep the existing kernel ABI and its completion-stat argument, while making
// the trace state explicit to the Host.  2 means "trace intentionally
// disabled" (0 is success with a materialized trace; 1 is overflow/error).
void unified_real_completion_trace_disabled(id_t capacity,
		tapa::mmap<ap_uint<64> > completion, tapa::mmap<id_t> stats) {
	// Preserve the deployed AXI ABI without creating a per-row HBM write
	// stream.  This independent, single diagnostic store is deliberately not
	// connected to the allocator feedback path.
	if (capacity != 0) completion[0] = 0;
	stats[0] = 0;
	stats[1] = 2;
}

// Translate the deployed route/profile arrays into the allocator's compact
// command ABI.  EMPTY rows use a phantom source-zero boundary so they still
// traverse the normal completion fabric and produce one tagged zero-NNZ EOR.
void unified_real_command_read(
		tapa::mmap<const ap_uint<32> > route,
		tapa::mmap<const ap_uint<32> > row_source_mask,
		tapa::mmap<const ap_uint<64> > row_geometry,
		tapa::mmap<const id_t> row_logical,
		id_t fragment_count,
		tapa::ostream<MergeAllocator8CommandToken>& output) {
	for (id_t fragment = 0; fragment < fragment_count; ++fragment) {
#pragma HLS PIPELINE II=1
		const ap_uint<32> route_word = route[fragment];
		const ap_uint<2> host_route = route_word.range(1, 0);
		const id_t span = row_geometry[fragment].range(63, 32);
		const bool legal_fragment = span != 0 && span <= 65536;
		const bool dense = host_route == ADAPT_FP32_DENSE && legal_fragment;
		const bool wide = dense && span > 32768;
		ap_uint<8> source_mask = row_source_mask[fragment].range(7, 0);
		if (source_mask == 0) source_mask = 1;
		ap_uint<128> command = 0;
		command.range(31, 0) = row_logical[fragment];
		command.range(63, 32) = fragment;
		command.range(71, 64) = source_mask;
		command.range(73, 72) = !legal_fragment ? (ap_uint<2>)0 : dense
			? (ap_uint<2>)UNIFIED_FULL_ROUTE_DENSE
			: (ap_uint<2>)UNIFIED_FULL_ROUTE_MERGE;
		command.range(105, 74) = route_word.range(30, 2);
		command[106] = wide;
		MergeAllocator8CommandToken token = 0;
		token.range(127, 0) = command;
		output.write(token);
	}
	MergeAllocator8CommandToken terminal = 0;
	terminal[128] = 1;
	output.write(terminal);
}

// One accepted allocation broadcasts only row metadata.  Each physical HBM
// reader owns a shard-local command list appended to its B buffer and emits
// its own row boundary after consuming that list.  This removes the cyclic
// head-of-line dependency in which a central issuer could block on the 11th
// command of one shard before sending boundaries to the other seven shards.
void unified_real_dispatch_issue(
		tapa::istream<MergeAllocator8DispatchToken>& dispatch_input,
		tapa::mmap<const ap_uint<64> > row_geometry,
		tapa::ostream<UnifiedSourceDispatchToken>& source_dispatch0,
		tapa::ostream<UnifiedSourceDispatchToken>& source_dispatch1,
		tapa::ostream<UnifiedSourceDispatchToken>& source_dispatch2,
		tapa::ostream<UnifiedSourceDispatchToken>& source_dispatch3,
		tapa::ostream<UnifiedSourceDispatchToken>& source_dispatch4,
		tapa::ostream<UnifiedSourceDispatchToken>& source_dispatch5,
		tapa::ostream<UnifiedSourceDispatchToken>& source_dispatch6,
		tapa::ostream<UnifiedSourceDispatchToken>& source_dispatch7,
		tapa::ostream<UnifiedRealReaderDispatch>& reader_dispatch0,
		tapa::ostream<UnifiedRealReaderDispatch>& reader_dispatch1,
		tapa::ostream<UnifiedRealReaderDispatch>& reader_dispatch2,
		tapa::ostream<UnifiedRealReaderDispatch>& reader_dispatch3,
		tapa::ostream<UnifiedRealReaderDispatch>& reader_dispatch4,
		tapa::ostream<UnifiedRealReaderDispatch>& reader_dispatch5,
		tapa::ostream<UnifiedRealReaderDispatch>& reader_dispatch6,
		tapa::ostream<UnifiedRealReaderDispatch>& reader_dispatch7,
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
		const MergeAllocator8DispatchToken dispatch_token = dispatch_input.read();
		if (dispatch_token[128]) {
			UnifiedSourceDispatchToken source_terminal = 0;
			source_terminal[144] = 1;
			source_dispatch0.write(source_terminal);
			source_dispatch1.write(source_terminal);
			source_dispatch2.write(source_terminal);
			source_dispatch3.write(source_terminal);
			source_dispatch4.write(source_terminal);
			source_dispatch5.write(source_terminal);
			source_dispatch6.write(source_terminal);
			source_dispatch7.write(source_terminal);
			UnifiedRealReaderDispatch reader_terminal = 0;
			reader_terminal[192] = 1;
			reader_dispatch0.write(reader_terminal);
			reader_dispatch1.write(reader_terminal);
			reader_dispatch2.write(reader_terminal);
			reader_dispatch3.write(reader_terminal);
			reader_dispatch4.write(reader_terminal);
			reader_dispatch5.write(reader_terminal);
			reader_dispatch6.write(reader_terminal);
			reader_dispatch7.write(reader_terminal);
			const BaseToken base_terminal = end_base();
			base0.write(base_terminal); base1.write(base_terminal);
			base2.write(base_terminal); base3.write(base_terminal);
			base4.write(base_terminal); base5.write(base_terminal);
			base6.write(base_terminal); base7.write(base_terminal);
			base8.write(base_terminal); base9.write(base_terminal);
			base10.write(base_terminal); base11.write(base_terminal);
			base12.write(base_terminal); base13.write(base_terminal);
			base14.write(base_terminal); dense_base0.write(base_terminal);
			dense_base1.write(base_terminal); heavy_base.write(base_terminal);
			done = true;
			continue;
		}

		const ap_uint<128> dispatch = dispatch_token.range(127, 0);
		const id_t row = dispatch.range(31, 0);
		const id_t fragment = dispatch.range(63, 32);
		const bool dense = dispatch[121];
		const bool heavy = dispatch[123];
		const bool wide = dispatch[124];
		const ap_uint<64> geometry = row_geometry[fragment];
		const id_t dense_base = geometry.range(31, 0);
		const id_t span = geometry.range(63, 32);
		BaseToken base = 0;
		base.range(31, 0) = row;
		base.range(63, 32) = dense_base;
		if (dense) {
			if (wide) {
				dense_base0.write(base);
				BaseToken upper = base;
				upper.range(63, 32) = dense_base + 32768;
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

		UnifiedSourceDispatchToken source_token = 0;
		source_token.range(127, 0) = dispatch;
		source_token.range(143, 128) = sequence;
		source_dispatch0.write(source_token); source_dispatch1.write(source_token);
		source_dispatch2.write(source_token); source_dispatch3.write(source_token);
		source_dispatch4.write(source_token); source_dispatch5.write(source_token);
		source_dispatch6.write(source_token); source_dispatch7.write(source_token);

		UnifiedRealReaderDispatch reader_dispatch = 0;
		reader_dispatch.range(127, 0) = dispatch;
		reader_dispatch.range(159, 128) = dense_base;
		reader_dispatch.range(191, 160) = span;
		reader_dispatch0.write(reader_dispatch);
		reader_dispatch1.write(reader_dispatch);
		reader_dispatch2.write(reader_dispatch);
		reader_dispatch3.write(reader_dispatch);
		reader_dispatch4.write(reader_dispatch);
		reader_dispatch5.write(reader_dispatch);
		reader_dispatch6.write(reader_dispatch);
		reader_dispatch7.write(reader_dispatch);
		++sequence;
	}
}

// B bank layout for every shard:
//   [0, B_data_beats)                         packed B numerical data
//   [B_data_beats, +ceil(fragment_count/16))  32-bit task count per fragment
//   [after headers, ...]                      four 128-bit descriptors/beat
// Descriptor records are concatenated in physical-fragment order.  Because
// the scheduler preserves command order, every reader needs only a local
// sequential descriptor cursor; no per-row start table or deep on-chip queue
// is required.  The count header is cached for sixteen consecutive fragments.
void unified_real_b_reader_fetch_embedded(
		tapa::mmap<const ap_uint<512> > B, id_t B_data_beats,
		id_t fragment_count,
		tapa::istream<UnifiedRealReaderDispatch>& dispatches,
		tapa::ostream<UnifiedRealRawPacket>& raw_packets) {
	ap_uint<512> dense_cache_data[kTapaDenseBReaderCacheBeats];
	id_t dense_cache_offset[kTapaDenseBReaderCacheRows];
	ap_uint<4> dense_cache_beats[kTapaDenseBReaderCacheRows];
	bool dense_cache_valid[kTapaDenseBReaderCacheRows];
#pragma HLS BIND_STORAGE variable=dense_cache_data type=ram_t2p impl=bram latency=2
#pragma HLS ARRAY_PARTITION variable=dense_cache_offset complete dim=1
#pragma HLS ARRAY_PARTITION variable=dense_cache_beats complete dim=1
#pragma HLS ARRAY_PARTITION variable=dense_cache_valid complete dim=1
	for (int entry = 0; entry < kTapaDenseBReaderCacheRows; ++entry) {
#pragma HLS UNROLL
		dense_cache_offset[entry] = 0;
		dense_cache_beats[entry] = 0;
		dense_cache_valid[entry] = false;
	}
	ap_uint<4> dense_cache_replacement = 0;
	const id_t header_words = (fragment_count + 15) >> 4;
	id_t descriptor_cursor = 0;
	id_t cached_header_index = ~id_t(0);
	ap_uint<512> cached_header = 0;
	id_t cached_descriptor_index = ~id_t(0);
	ap_uint<512> cached_descriptors = 0;
	bool done = false;
	while (!done) {
		const UnifiedRealReaderDispatch reader_dispatch = dispatches.read();
		if (reader_dispatch[192]) {
			UnifiedRealRawPacket terminal = 0;
			terminal[552] = 1;
			raw_packets.write(terminal);
			done = true;
			continue;
		}

		const ap_uint<128> dispatch = reader_dispatch.range(127, 0);
		const id_t row = dispatch.range(31, 0);
		const id_t fragment = dispatch.range(63, 32);
		const bool dense = dispatch[121];
		const bool heavy = dispatch[123];
		const ap_uint<2> mode = dense
			? (ap_uint<2>)ADAPT_FP32_DENSE
			: (ap_uint<2>)ADAPT_FP32_MERGE;
		const id_t dense_base = reader_dispatch.range(159, 128);
		const id_t span = reader_dispatch.range(191, 160);
		const id_t header_index = fragment >> 4;
		if (header_index != cached_header_index) {
			cached_header = B[B_data_beats + header_index];
			cached_header_index = header_index;
		}
		const ap_uint<4> header_lane = fragment & 15;
		const id_t command_count = cached_header.range(
			(unsigned)header_lane * 32 + 31, (unsigned)header_lane * 32);

		for (id_t local_command = 0; local_command < command_count;
				++local_command) {
			const id_t descriptor_index = descriptor_cursor >> 2;
			if (descriptor_index != cached_descriptor_index) {
				cached_descriptors = B[B_data_beats + header_words
					+ descriptor_index];
				cached_descriptor_index = descriptor_index;
			}
			const ap_uint<2> descriptor_lane = descriptor_cursor & 3;
			const ap_uint<128> descriptor = cached_descriptors.range(
				(unsigned)descriptor_lane * 128 + 127,
				(unsigned)descriptor_lane * 128);
			++descriptor_cursor;

			const id_t offset = descriptor.range(31, 0);
			const id_t length = descriptor.range(55, 32);
			const id_t beats = (length + 7) >> 3;
			const bool cacheable = dense && beats != 0
				&& beats <= kTapaDenseBReaderCacheRowBeats;
			bool cache_hit = false;
			ap_uint<4> cache_hit_row = 0;
			for (int entry = 0; entry < kTapaDenseBReaderCacheRows; ++entry) {
#pragma HLS UNROLL
				if (cacheable && dense_cache_valid[entry]
						&& dense_cache_offset[entry] == offset
						&& dense_cache_beats[entry] == beats) {
					cache_hit = true;
					cache_hit_row = entry;
				}
			}
			const ap_uint<4> cache_fill_row = dense_cache_replacement;
#pragma HLS DEPENDENCE variable=dense_cache_data inter false
			for (id_t beat = 0; beat < beats; ++beat) {
#pragma HLS PIPELINE II=1
				const id_t address = offset + beat;
				ap_uint<512> source = 0;
				if (address < B_data_beats) {
					if (cache_hit) {
						const id_t cache_index
							= ((id_t)cache_hit_row << 3) + beat;
						source = dense_cache_data[cache_index];
					} else {
						source = B[address];
						if (cacheable) {
							const id_t cache_index
								= ((id_t)cache_fill_row << 3) + beat;
							dense_cache_data[cache_index] = source;
						}
					}
				}
				const id_t remaining = length - (beat << 3);
				const ap_uint<4> valid = remaining > 8 ? 8 : remaining;
				UnifiedRealRawPacket packet = 0;
				packet.range(511, 0) = source;
				packet.range(515, 512) = valid;
				packet.range(547, 516) = row;
				packet.range(549, 548) = mode;
				packet[550] = beat + 1 == beats
					|| (mode == ADAPT_FP32_MERGE
						&& (beat + 1) % kTapaMergeRunPackets == 0);
				packet.range(584, 553) = descriptor.range(127, 96);
				packet.range(616, 585) = dense_base;
				packet.range(648, 617) = span;
				packet[649] = heavy;
				raw_packets.write(packet);
			}
			if (cacheable && !cache_hit) {
				dense_cache_offset[cache_fill_row] = offset;
				dense_cache_beats[cache_fill_row] = beats;
				dense_cache_valid[cache_fill_row] = true;
				dense_cache_replacement = dense_cache_replacement + 1;
			}
		}

		UnifiedRealRawPacket boundary = 0;
		boundary.range(547, 516) = row;
		boundary.range(549, 548) = mode;
		boundary[551] = 1;
		boundary[649] = heavy;
		raw_packets.write(boundary);
	}
}

// Adapt one real post-local-merge shard stream to the accepted allocator/tree
// token protocols.  No numerical row is materialized: packets are consumed and
// routed as they arrive, and only a small dispatch identity remains live.
void unified_real_postlocal_adapter(
		tapa::istream<UnifiedRealProductPacket>& input,
		ap_uint<3> source_index,
		tapa::istream<UnifiedSourceDispatchToken>& dispatch_input,
		tapa::ostream<UnifiedRoutedLeafToken>& merge_output,
		tapa::ostream<UnifiedFullDenseSourceToken>& dense_output,
		tapa::ostream<UnifiedFullDenseSourceToken>& heavy_output) {
	bool done = false;
	while (!done) {
		const UnifiedSourceDispatchToken source_dispatch = dispatch_input.read();
		if (source_dispatch[144]) {
			// The fetch/local-merge chain terminates after all preceding row
			// boundaries.  Consume its terminal before closing all three routes.
			UnifiedRealProductPacket terminal = input.read();
			while (!terminal[552]) terminal = input.read();
			UnifiedRoutedLeafToken merge_terminal = 0;
			merge_terminal[132] = 1;
			merge_output.write(merge_terminal);
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

		bool row_done = false;
		bool protocol_error = false;
		while (!row_done) {
			const UnifiedRealProductPacket packet = input.read();
			if (packet[552]) {
				protocol_error = true;
				row_done = true;
				continue;
			}
			protocol_error |= packet[553]
				|| packet.range(547, 516) != row;
			if (packet[551]) {
				row_done = true;
				continue;
			}
			const ap_uint<4> count = packet.range(515, 512);
			for (int lane = 0; lane < 8; ++lane) {
#pragma HLS PIPELINE II=1
				if (lane >= count) continue;
				const ap_uint<64> numerical
					= packet.range(lane * 64 + 63, lane * 64);
				if (dense || heavy) {
					UnifiedFullDenseSourceToken token = 0;
					token.range(63, 0) = numerical;
					token.range(95, 64) = row;
					token.range(98, 96) = context;
					token.range(106, 99) = generation;
					token[107] = dispatch[122];
					token.range(125, 110) = sequence;
					token[126] = protocol_error;
					token[127] = heavy;
					token[128] = wide;
					if (heavy) heavy_output.write(token);
					else dense_output.write(token);
				} else if (active) {
					MergeForwardTree8Token tree = 0;
					tree.range(127, 0) = make_tree_item(numerical, row, context,
						generation, level, false, sequence);
					UnifiedRoutedLeafToken routed = 0;
					routed.range(128, 0) = tree;
					routed.range(131, 129) = target_leaf;
					merge_output.write(routed);
				}
			}
		}

		if (dense || heavy) {
			UnifiedFullDenseSourceToken eor = 0;
			eor.range(95, 64) = row;
			eor.range(98, 96) = context;
			eor.range(106, 99) = generation;
			eor[107] = dispatch[122];
			eor[108] = 1;
			eor.range(125, 110) = sequence;
			eor[126] = protocol_error;
			eor[127] = heavy;
			eor[128] = wide;
			if (heavy) heavy_output.write(eor);
			else dense_output.write(eor);
		} else if (active || padding) {
			MergeForwardTree8Token eor = 0;
			eor.range(127, 0) = make_tree_item(
				0, row, context, generation, level, true, sequence);
			UnifiedRoutedLeafToken routed = 0;
			routed.range(128, 0) = eor;
			routed.range(131, 129) = target_leaf;
			merge_output.write(routed);
		}
	}
}

void adaptive_hbm_unified_real_hbm_tapa(
		tapa::mmap<const ap_uint<32> > route,
		tapa::mmap<const ap_uint<32> > row_source_mask,
		tapa::mmap<const ap_uint<64> > row_geometry,
		tapa::mmap<const id_t> row_logical,
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
		id_t fragment_count, id_t heavy_merge_product_threshold,
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
	// These FIFOs are purely elastic boundaries between concurrent TAPA tasks.
	// Large 32/16-deep instances were implemented as SRL/LUTRAM and pushed the
	// U280 to 92% CLB-site use, even though arithmetic LUT use was only 55%.
	// Four entries retain one full packet-mux round of slack; the B-reader
	// boundary keeps eight entries to absorb an HBM burst without reintroducing
	// the previous physical congestion.
	tapa::stream<MergeAllocator8CommandToken, 4> command_stream("command_stream");
	tapa::stream<MergeAllocator8DispatchToken, 4> dispatch_stream("dispatch_stream");
	// Completion returns form the only cyclic backpressure path: a long output
	// burst must be able to retire completed rows while packet/meta writers are
	// temporarily serving HBM.  Four entries pass short CSim gates but deadlock
	// the routed design on sustained ex9 output.  Eight breaks that physical
	// feedback cycle without replicating any reducer or changing arithmetic.
	tapa::stream<MergeAllocator8EorToken, 8> scheduler_eors("scheduler_eors");
	tapa::stream<MergeAllocator8AckToken, 8> scheduler_acks("scheduler_acks");
	tapa::stream<ap_uint<512>, 2> scheduler_statistics("scheduler_statistics");

	tapa::streams<UnifiedSourceDispatchToken, 8, 4> source_dispatches("source_dispatches");
	tapa::streams<UnifiedRealReaderDispatch, 8, 4>
		reader_dispatches("reader_dispatches");
	tapa::streams<UnifiedRealRawPacket, 8, 8>
		raw_packets("raw_packets");
	tapa::streams<UnifiedRealProductPacket, 8, 8> products("products");
	tapa::streams<UnifiedRealProductPacket, 8, 8> local_products("local_products");
	tapa::streams<UnifiedRoutedLeafToken, 8, 4> routed_sources("routed_sources");
	tapa::streams<UnifiedFullDenseSourceToken, 8, 4> dense_sources("dense_sources");
	tapa::streams<UnifiedFullDenseSourceToken, 8, 4> heavy_sources("heavy_sources");
	tapa::streams<UnifiedHeavyVector4Token, 8, 4> heavy_vector_leaf("heavy_vector_leaf");
	tapa::streams<UnifiedHeavyVector4Token, 4, 4> heavy_vector_stage0("heavy_vector_stage0");
	tapa::streams<UnifiedHeavyVector4Token, 2, 4> heavy_vector_stage1("heavy_vector_stage1");
	tapa::stream<UnifiedHeavyVector4Token, 4> heavy_vector_root("heavy_vector_root");
	tapa::streams<UnifiedRoutedLeafToken, 8, 4> crossbar_stage0("crossbar_stage0");
	tapa::streams<UnifiedRoutedLeafToken, 8, 4> crossbar_stage1("crossbar_stage1");
	tapa::streams<UnifiedRoutedLeafToken, 8, 4> crossbar_stage2("crossbar_stage2");
	tapa::streams<MergeForwardTree8Token, 8, 4> memory_to_leaf("memory_to_leaf");

	tapa::streams<MergeForwardTree8Token, 8, 4> leaf_up("leaf_up");
	tapa::streams<MergeForwardTree8Token, 8, 4> leaf_exit("leaf_exit");
	tapa::streams<MergeForwardTree8Token, 4, 4> level1_up("level1_up");
	tapa::streams<MergeForwardTree8Token, 4, 4> level1_exit("level1_exit");
	tapa::streams<MergeForwardTree8Token, 2, 4> level2_up("level2_up");
	tapa::streams<MergeForwardTree8Token, 2, 4> level2_exit("level2_exit");
	tapa::stream<MergeForwardTree8Token, 4> root_exit("root_exit");
	tapa::stream<MergeForwardTree8Token, 4> root_up("root_up");
	tapa::streams<BaseToken, 15, 4> merge_bases("merge_bases");
	tapa::streams<BaseToken, 2, 4> dense_bases("dense_bases");
	tapa::stream<BaseToken, 4> heavy_base("heavy_base");

	tapa::stream<Segmented8SharedToken, 4> dense_raw("dense_raw");
	tapa::stream<UnifiedFullDenseTag, 4> dense_tags("dense_tags");
	tapa::stream<Segmented8SharedToken, 4> heavy_raw("heavy_raw");
	tapa::stream<UnifiedFullDenseTag, 4> heavy_tags("heavy_tags");
	tapa::stream<Segmented8SharedToken, 4> shared_raw("shared_raw");
	tapa::stream<UnifiedFullDenseTag, 4> shared_tags("shared_tags");
	tapa::stream<Segmented8SharedToken, 4> dense_reduced("dense_reduced");
	tapa::streams<DenseDualContextToken, 2, 4> dense_prebank("dense_prebank");
	tapa::streams<JointDenseLaneToken, 8, 4> dense_lane0("dense_lane0");
	tapa::streams<JointDenseLaneToken, 8, 4> dense_lane1("dense_lane1");
	tapa::streams<DenseDualContextToken, 2, 4> dense_accumulate("dense_accumulate");
	tapa::streams<DenseDualContextToken, 2, 4> dense_result("dense_result");
	tapa::streams<DenseDualContextToken, 2, 4> dense_compact2("dense_compact2");
	tapa::streams<DenseDualContextToken, 2, 4> dense_compact4("dense_compact4");
	tapa::streams<DenseDualContextToken, 2, 4> dense_compact8("dense_compact8");
	tapa::streams<ap_uint<192>, 2, 2> dense_statistics("dense_statistics");
	tapa::stream<MergeKeyedCarry8InputToken, 4> heavy_segmented("heavy_segmented");
	tapa::stream<CarryWorkToken, 4> heavy_carry_work("heavy_carry_work");
	tapa::stream<CarryResultToken, 4> heavy_carry_result("heavy_carry_result");
	tapa::stream<DenseDualContextToken, 4> heavy_result("heavy_result");
	tapa::stream<DenseDualContextToken, 4> heavy_compact2("heavy_compact2");
	tapa::stream<DenseDualContextToken, 4> heavy_compact4("heavy_compact4");
	tapa::stream<DenseDualContextToken, 4> heavy_compact8("heavy_compact8");
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
	tapa::stream<RecordBatch, 8> record_batches("record_batches");
	tapa::stream<WordToken, 8> meta_words("meta_words");
	tapa::stream<StatsToken, 4> meta_summary("meta_summary");

	tapa::task()
		.invoke(unified_real_command_read, route, row_source_mask, row_geometry,
			row_logical, fragment_count, command_stream)
		.invoke(unified_full_scheduler, command_stream, scheduler_eors,
			heavy_merge_product_threshold, dispatch_stream, scheduler_acks,
			scheduler_statistics)
		.invoke(unified_allocator_ack_drain, scheduler_acks)
		.invoke(unified_real_dispatch_issue, dispatch_stream, row_geometry,
			source_dispatches[0], source_dispatches[1], source_dispatches[2],
			source_dispatches[3], source_dispatches[4], source_dispatches[5],
			source_dispatches[6], source_dispatches[7],
			reader_dispatches[0], reader_dispatches[1], reader_dispatches[2],
			reader_dispatches[3], reader_dispatches[4], reader_dispatches[5],
			reader_dispatches[6], reader_dispatches[7],
			merge_bases[0], merge_bases[1], merge_bases[2], merge_bases[3],
			merge_bases[4], merge_bases[5], merge_bases[6], merge_bases[7],
			merge_bases[8], merge_bases[9], merge_bases[10], merge_bases[11],
			merge_bases[12], merge_bases[13], merge_bases[14],
			dense_bases[0], dense_bases[1], heavy_base)
#define UNIFIED_REAL_FRONTEND(S, MEMORY, BEATS) \
		.invoke(unified_real_b_reader_fetch_embedded, MEMORY, BEATS, \
			fragment_count, reader_dispatches[S], raw_packets[S]) \
		.invoke(tapa_b_reader_scale, raw_packets[S], \
			products[S]) \
		.invoke(tapa_shard_local_merge, products[S], \
			local_products[S]) \
		.invoke(unified_real_postlocal_adapter, local_products[S], S, \
			source_dispatches[S], routed_sources[S], dense_sources[S], \
			heavy_sources[S])
		UNIFIED_REAL_FRONTEND(0, B0, B0_beats)
		UNIFIED_REAL_FRONTEND(1, B1, B1_beats)
		UNIFIED_REAL_FRONTEND(2, B2, B2_beats)
		UNIFIED_REAL_FRONTEND(3, B3, B3_beats)
		UNIFIED_REAL_FRONTEND(4, B4, B4_beats)
		UNIFIED_REAL_FRONTEND(5, B5, B5_beats)
		UNIFIED_REAL_FRONTEND(6, B6, B6_beats)
		UNIFIED_REAL_FRONTEND(7, B7, B7_beats)
#undef UNIFIED_REAL_FRONTEND
#define UNIFIED_REAL_HEAVY_PACK4(S) \
		.invoke(unified_full_heavy_source_pack4, heavy_sources[S], \
			heavy_vector_leaf[S])
		UNIFIED_REAL_HEAVY_PACK4(0)
		UNIFIED_REAL_HEAVY_PACK4(1)
		UNIFIED_REAL_HEAVY_PACK4(2)
		UNIFIED_REAL_HEAVY_PACK4(3)
		UNIFIED_REAL_HEAVY_PACK4(4)
		UNIFIED_REAL_HEAVY_PACK4(5)
		UNIFIED_REAL_HEAVY_PACK4(6)
		UNIFIED_REAL_HEAVY_PACK4(7)
#undef UNIFIED_REAL_HEAVY_PACK4
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
		// The port order is exactly the prior 8-wide stage's four pairs.
		.invoke(unified_allocator_crossbar_switch2, routed_sources[0], routed_sources[4], 2, crossbar_stage0[0], crossbar_stage0[4])
		.invoke(unified_allocator_crossbar_switch2, routed_sources[1], routed_sources[5], 2, crossbar_stage0[1], crossbar_stage0[5])
		.invoke(unified_allocator_crossbar_switch2, routed_sources[2], routed_sources[6], 2, crossbar_stage0[2], crossbar_stage0[6])
		.invoke(unified_allocator_crossbar_switch2, routed_sources[3], routed_sources[7], 2, crossbar_stage0[3], crossbar_stage0[7])
		.invoke(unified_allocator_crossbar_switch2, crossbar_stage0[0], crossbar_stage0[2], 1, crossbar_stage1[0], crossbar_stage1[2])
		.invoke(unified_allocator_crossbar_switch2, crossbar_stage0[1], crossbar_stage0[3], 1, crossbar_stage1[1], crossbar_stage1[3])
		.invoke(unified_allocator_crossbar_switch2, crossbar_stage0[4], crossbar_stage0[6], 1, crossbar_stage1[4], crossbar_stage1[6])
		.invoke(unified_allocator_crossbar_switch2, crossbar_stage0[5], crossbar_stage0[7], 1, crossbar_stage1[5], crossbar_stage1[7])
		.invoke(unified_allocator_crossbar_switch2, crossbar_stage1[0], crossbar_stage1[1], 0, crossbar_stage2[0], crossbar_stage2[1])
		.invoke(unified_allocator_crossbar_switch2, crossbar_stage1[2], crossbar_stage1[3], 0, crossbar_stage2[2], crossbar_stage2[3])
		.invoke(unified_allocator_crossbar_switch2, crossbar_stage1[4], crossbar_stage1[5], 0, crossbar_stage2[4], crossbar_stage2[5])
		.invoke(unified_allocator_crossbar_switch2, crossbar_stage1[6], crossbar_stage1[7], 0, crossbar_stage2[6], crossbar_stage2[7])
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
			(id_t)32768, dense_result[0], dense_statistics[0])
		.invoke(dense_dual_context_engine1, dense_accumulate[1],
			(id_t)32768, dense_result[1], dense_statistics[1])
		.invoke(unified_full_dense_compact2, dense_result[0], dense_compact2[0])
		.invoke(unified_full_dense_compact2, dense_result[1], dense_compact2[1])
		.invoke(unified_full_dense_compact4, dense_compact2[0], dense_compact4[0])
		.invoke(unified_full_dense_compact4, dense_compact2[1], dense_compact4[1])
		.invoke(unified_full_dense_compact8, dense_compact4[0], dense_compact8[0])
		.invoke(unified_full_dense_compact8, dense_compact4[1], dense_compact8[1])
		.invoke(unified_full_dense_compact2, heavy_result, heavy_compact2)
		.invoke(unified_full_dense_compact4, heavy_compact2, heavy_compact4)
		.invoke(unified_full_dense_compact8, heavy_compact4, heavy_compact8)
#define UNIFIED_REAL_PACKETIZE(STREAM, EXIT, PORT) \
		.invoke(merge15_exit_packetize, STREAM, merge_bases[EXIT], EXIT, PORT, \
			exit_packets[EXIT], exit_completions[EXIT])
		UNIFIED_REAL_PACKETIZE(leaf_exit[0], 0, 0)
		UNIFIED_REAL_PACKETIZE(leaf_exit[1], 1, 1)
		UNIFIED_REAL_PACKETIZE(leaf_exit[2], 2, 2)
		UNIFIED_REAL_PACKETIZE(leaf_exit[3], 3, 3)
		UNIFIED_REAL_PACKETIZE(leaf_exit[4], 4, 0)
		UNIFIED_REAL_PACKETIZE(leaf_exit[5], 5, 1)
		UNIFIED_REAL_PACKETIZE(leaf_exit[6], 6, 2)
		UNIFIED_REAL_PACKETIZE(leaf_exit[7], 7, 3)
		UNIFIED_REAL_PACKETIZE(level1_exit[0], 8, 0)
		UNIFIED_REAL_PACKETIZE(level1_exit[1], 9, 1)
		UNIFIED_REAL_PACKETIZE(level1_exit[2], 10, 2)
		UNIFIED_REAL_PACKETIZE(level1_exit[3], 11, 3)
		UNIFIED_REAL_PACKETIZE(level2_exit[0], 12, 0)
		UNIFIED_REAL_PACKETIZE(level2_exit[1], 13, 1)
		UNIFIED_REAL_PACKETIZE(root_exit, 14, 2)
#undef UNIFIED_REAL_PACKETIZE
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
		.invoke(unified_real_completion_to_scheduler, merged_completions,
			scheduler_eors)
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
		.invoke(unified_real_completion_trace_disabled,
			output_completion_capacity, output_completion, output_completion_stats)
		.invoke(unified_full_scheduler_statistics_write, scheduler_statistics,
			scheduler_stats)
		.invoke(unified_full_dense_statistics_write, dense_statistics[0],
			dense_statistics[1], dense_stats)
		.invoke(unified_full_heavy_statistics_write, heavy_statistics,
			heavy_merge_stats);
}
